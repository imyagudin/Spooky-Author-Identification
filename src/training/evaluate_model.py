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

def test_model(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_function: nn.Module=None,
        device: torch.device=None
):
    logger = get_logger(__name__)
    logger.info("Start test the model")

    model.eval()
    logger.info("The model has been switched to evaluate mode.")
    device = device if device is not None else torch.device("cpu")
    logger.info(f"Device selected: {device}")
    model.to(device)

    loss_function = loss_function if loss_function is not None else nn.CrossEntropyLoss()

    history = []
    for idx, (batch, target) in enumerate(dataloader):
        with torch.no_grad():
            batch = batch.to(device)
            target = target.float().to(device)
            predictions = model(batch)
            loss = loss_function(predictions, target)
            history.append(loss.item())
            if (idx + 1) % 10 == 0:
                clear_output(True)
                plt.plot(history, label="loss")
                plt.legend()
                plt.show()
    logger.info(f"Test {loss_function}: {np.mean(history)}")