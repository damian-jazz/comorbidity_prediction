import torch
from torch.nn import functional as F

##### generic loss_fn #####

def train(dataloader, device, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device) 
        
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (batch+1 == len(dataloader)):
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, device, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    model.eval()
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            
            test_loss += loss_fn(pred, y).item()

            pred[pred < 0.5] = 0
            pred[pred >= 0.5] = 1

            correct += (pred == y).type(torch.float).sum().item()
            
    test_loss /= num_batches
    correct /= dataloader.dataset.labels.shape[1] # to account for multiple labels
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def eval(dataloader, device, model):
    y_prob = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for batch, (X, _) in enumerate(dataloader):
            X = X.to(device)
            out = model(X)

            prob = torch.sigmoid(out).detach().cpu()  
            pred = torch.sigmoid((out)).detach().cpu()
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0

            if batch == 0:
                y_prob = prob
                y_pred = pred
            else:
                y_prob = torch.cat((y_prob, prob), 0)
                y_pred = torch.cat((y_pred, pred), 0)

    return y_prob, y_pred

##### focal loss #####

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

def train_focal(dataloader, device, model, optimizer, gamma=2):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device) 
        
        # Compute prediction error
        pred = model(X)
        loss = sigmoid_focal_loss(pred, y, gamma=gamma)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (batch+1 == len(dataloader)):
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_focal(dataloader, device, model, gamma=2):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    model.eval()
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            
            test_loss += sigmoid_focal_loss(pred, y, gamma=gamma).item()

            pred[pred < 0.5] = 0
            pred[pred >= 0.5] = 1

            correct += (pred == y).type(torch.float).sum().item()
            
    test_loss /= num_batches
    correct /= dataloader.dataset.labels.shape[1] # to account for multiple labels
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")