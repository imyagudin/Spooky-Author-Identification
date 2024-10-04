import matplotlib.pyplot as plt
import torch
from torch import nn
from IPython.display import clear_output

def test_model(
        model,
        dataloader,
        device=None
):
    device = device if device is not None else torch.device("cpu")
    model.to(device)
    model.eval()

    logloss = num_samples = 0.0
    epsilon = 1e-15

    for batch, targets in dataloader:
        with torch.no_grad():
            targets = targets.float().to(device)
            predictions = model(batch)
            predictions = nn.functional.softmax(predictions, dim=1)
            predictions = torch.clamp(predictions, min=epsilon, max=1-epsilon)
            logloss += -torch.sum(targets * torch.log(predictions)).item()
            num_samples += len(batch)

    return logloss / num_samples