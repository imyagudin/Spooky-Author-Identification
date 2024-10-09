import os
import sys
from typing import Union

import pandas as pd
from sklearn.metrics import train_test_split

project_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(project_path)

from src.data.load_data import read_file
from src.utils.logging_utils import get_logger

def split_data(data: Union[str, pd.DataFrame],
               test_size: float=0.2,
               val_size: float=0.2,
               shuffle: bool=False,
               random_state: int=42):
    logger = get_logger(__name__)
    if isinstance(data, str):
        data = read_file(data)
    temp_data, test_data = train_test_split(data,
                                            test_size=test_size,
                                            shuffle=shuffle,
                                            random_state=random_state)
    train_data, val_data = train_test_split(temp_data,
                                            test_size=val_size / (1 - test_size),
                                            shuffle=shuffle,
                                            random_state=random_state)
    logger.info(f"Data successfully split. Training set shape: {train_data.shape}, validation set shape: {val_data.shape}, test set shape: {test_data.shape}.")
    return train_data, val_data, test_data