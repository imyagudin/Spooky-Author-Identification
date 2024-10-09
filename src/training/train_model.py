import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from IPython.display import clear_output

project_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(project_path)

from src.utils.logging_utils import get_logger

def train_model(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        epochs: int=None,
        optimizer: torch.optim=None,
        loss_function: nn.Module=None,
        device: torch.device=None
):
    logger = get_logger(__name__)
    logger.info("Start training the model")
    model.train()
    logger.info("The model has been switched to training mode.")
    device = device if device is not None else torch.device("cpu")
    logger.info(f"Device selected: {device}")
    model.to(device)

    epochs = epochs if epochs is not None else 1

    optimizer = optimizer if optimizer is not None else torch.optim.Adam(model.parameters())
    loss_function = loss_function if loss_function is not None else nn.CrossEntropyLoss()

    history = []
    for epoch in range(epochs):
        for idx, (batch, target) in enumerate(dataloader):
            optimizer.zero_grad()
            target = target.float().to(device)
            predictions = model(batch)
            loss = loss_function(predictions, target)
            loss.backward()
            optimizer.step()
            history.append(loss.item())
            if (idx + 1) % 10 == 0:
                clear_output(True)
                plt.plot(history, label="loss")
                plt.legend()
                plt.show()
        logger.info(f"Epoch #1: Train {loss_function}: {np.mean(history[-1 * len(dataloader):])}")