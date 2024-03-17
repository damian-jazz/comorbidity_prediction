import torch
from cnn_utils import sigmoid_focal_loss

##### bce loss #####

def train(dataloader, device, model, loss_fn, optimizer, logging, period=20):
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

        if (batch % period == 0):
            loss, current = loss.item(), (batch + 1) * len(X)
            logging.info(f"\tTrain loss: {loss:.4f}  [{current:>5d}/{size:>5d}]") 
    
    logging.info(f"\tTrain loss: {loss:.4f}") 

def test(dataloader, device, model, loss_fn, logging):
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
    logging.info(f"\tTest accuracy: {(100*correct):>0.1f}")
    logging.info(f"\tTest avg loss: {test_loss:>4f}")

##### focal loss #####

def train_focal(dataloader, device, model, optimizer, logging, period=20):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = sigmoid_focal_loss(pred, y)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (batch % period == 0):
            loss, current = loss.item(), (batch + 1) * len(X)
            logging.info(f"\tTrain loss: {loss:.4f}  [{current:>5d}/{size:>5d}]") 

    logging.info(f"\tTrain loss: {loss:.4f}") 

def test_focal(dataloader, device, model, logging):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            
            test_loss += sigmoid_focal_loss(pred, y).item()

            pred[pred < 0.5] = 0
            pred[pred >= 0.5] = 1

            correct += (pred == y).type(torch.float).sum().item()
            
    test_loss /= num_batches
    correct /= dataloader.dataset.labels.shape[1] # to account for multiple labels
    correct /= size
    logging.info(f"\tTest accuracy: {(100*correct):>0.1f}")
    logging.info(f"\tTest avg loss: {test_loss:>4f}")

##### eval #####

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