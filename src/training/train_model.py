import matplotlib.pyplot as plt
import torch
from torch import nn
from IPython.display import clear_output

def train_model(
        model,
        dataloader,
        epochs=None,
        optimizer=None,
        loss_function=None,
        device=None
):
    model.train()
    device = device if device is not None else torch.device("cpu")
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