import torch

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

        if (batch+1 % 1 == 0):
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

def eval(dataloader, device, model, loss_fn):
    y_prob = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        counter = 0
        for X, _ in dataloader:
            X = X.to(device)
            out = model(X)

            prob = torch.sigmoid(out).detach().cpu()  
            pred = torch.sigmoid((out)).detach().cpu()
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0

            if counter == 0:
                y_prob = prob
                y_pred = pred
                counter += 1
            else:
                y_prob = torch.cat((y_prob, prob), 0)
                y_pred = torch.cat((y_pred, pred), 0)

    return y_prob, y_pred