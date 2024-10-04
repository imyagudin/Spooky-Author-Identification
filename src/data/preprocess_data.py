import os
import string
from collections import Counter
from typing import Callable, Dict, Optional, List

import nltk
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class Vocabulary:
  PAD, UNK, SOS, EOS = '<pad>', '<unk>', '<sos>', '<eos>'

  def __init__(self,
                text: Optional[List[str]],
                minimal_frequency: int = 1,
                maximum_size: Optional[int] = None,
                special_tokens: Optional[List[str]] = None,
                tokenizer: Optional[Callable[[str], List[str]]] = None,
                preprocessor: Optional[Callable[[str], str]] = 'lowercase',
                lowercase: Optional[bool]=True,
                remove_punctuation: Optional[bool]=None
                ) -> None:
    self.minimal_frequency = minimal_frequency
    self.maximum_size = maximum_size
    self.special_tokens = special_tokens or []
    self.tokenizer = tokenizer or nltk.tokenize.word_tokenize
    self.preprocessor = preprocessor
    self.remove_punctuation = remove_punctuation
    self.tokens = self._tokenize(text)
    self.token_counts = Counter(self.tokens)
    self.UNK_INDEX, self.PAD_INDEX, self.token2id, self.id2token = self._preprocess()

  def add(self, token: str):
    if token not in self.token2id:
      self.token_counts[token] += 1
      if self.token_counts[token] >= self.minimal_frequency:
        self.token2id[token] = len(self.token2id)
        self.id2token.append(token)
    return self

  def remove(self, token: str):
    if token in self.token2id:
      del self.token2id[token]
      self.id2token.remove(token)
      del self.token_counts[token]
    else:
      raise KeyError(f"There isn't token \"{token}\" in vocabulary")
    return self

  def update(self, new_text: str):
    new_tokens = self._tokenize(new_text)
    new_token_counts = Counter(new_tokens)
    self.token_counts.update(new_token_counts)

    self.token2id = {token: idx for idx, token in enumerate(self.special_tokens + sorted(token for token, count in self.token_counts.items() if count >= self.minimal_frequency))}
    self.id2token = list(self.token2id.keys())

    if self.maximum_size is not None and len(self.token2id) > self.maximum_size:
      most_common_tokens = [token for token, _ in self.token_counts.most_common(self.maximum_size)]
      self.token2id = {token: idx for idx, token in enumerate(self.special_tokens + most_common_tokens)}
      self.id2token = list(self.token2id.keys())

    self.UNK_INDEX, self.PAD_INDEX = self.token2id[self.UNK], self.token2id[self.PAD]

  def _tokenize(self, text: Optional[List[str]] = None) -> List[str]:
    tokens = list()
    if self.preprocessor == 'lowercase':
      text = [item.lower() for item in text]
    for item in text:
      tokens.extend(self.tokenizer(item))
    if self.remove_punctuation is True:
      tokens = [token for token in tokens if token not in string.punctuation]
    return tokens

  def _preprocess(self):
    self.tokens = [self.UNK, self.PAD] + self.special_tokens + sorted(token for token, count in self.token_counts.items() if count >= self.minimal_frequency)
    token2id = {token: idx for idx, token in enumerate(self.tokens)}
    id2token = self.tokens
    UNK_INDEX, PAD_INDEX = token2id[self.UNK], token2id[self.PAD]
    return UNK_INDEX, PAD_INDEX, token2id, id2token

  def __len__(self):
      return len(self.token2id)

  def __getitem__(self, token: str):
    return self.token2id.get(token, self.UNK_INDEX)

  def get_token(self, index: int):
      return self.id2token[index]

class SpookyAuthors(Dataset):
    def __init__(self,
                 root=None,
                 train: Optional[bool]=True,
                 max_length: Optional[int]=None,
                 padding: Optional[str]='max_length',
                 truncation: Optional[bool]=True,
                 vocab: Optional[Dict[str, int]]=None,
                 stopwords: Optional[List[str]]=None,
                 label_map: Optional[Dict[str, int]]=None,
                 transform: Optional[Callable]=None
                ) -> None:
        super().__init__()
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.stopwords = stopwords
        self.label_map = label_map
        self.transform = transform
        self.train = train

        self.data_path = os.path.join(root, 'train.csv' if self.train is True else 'test.csv')
        if not os.path.isdir(root):
            raise NotADirectoryError(f"The path '{root}' is not a directory.")
        self.data = pd.read_csv(self.data_path)
        self.text = self.data['text']
        self.author = self.data['author'].factorize()[0] if train is True else np.zeros(len(self.text))
        self.vocab = vocab or Vocabulary(self.text)
        self.features = self.__text2vect__(self.text)

    def __text2vect__(self, sequences):
        if isinstance(sequences[0], str):
            sequences = list(map(str.split, sequences))
        if self.max_length is None:
            self.max_length = min(max(map(len, sequences)), self.max_length or float('inf'))
        matrix = np.full((len(sequences), self.max_length), np.int32(self.vocab.PAD_INDEX))
        for i, seq in enumerate(sequences):
            row_ix =[self.vocab.token2id.get(word, self.vocab.UNK_INDEX) for word in seq[:self.max_length]]
            matrix[i, :len(row_ix)] = row_ix
        return matrix

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        item = torch.tensor(self.features[index], dtype=torch.long)
        target = torch.tensor(self.author[index], dtype=torch.long)
        target = torch.nn.functional.one_hot(target, num_classes=3)
        return item, target