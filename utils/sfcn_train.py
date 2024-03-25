import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.sfcn_utils import DatasetBrainImages
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, average_precision_score, roc_auc_score, brier_score_loss, f1_score, hamming_loss, balanced_accuracy_score, accuracy_score

##### bce loss #####

def train(dataloader, device, model, loss_fn, optimizer, logging, period=20):
    size = len(dataloader.dataset)
    model.train()
    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (batch % period == 0):
            loss, current = loss.item(), (batch + 1) * len(X)
            logging.info(f"\tTrain loss: {loss:.4f}  [{current:>5d}/{size:>5d}]") 
    
    train_loss /= len(dataloader)
    logging.info(f"\tAverage loss: {train_loss:.4f}") 

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

def train_focal(dataloader, device, model, optimizer, logging, period=20):
    size = len(dataloader.dataset)
    model.train()
    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = sigmoid_focal_loss(pred, y)
        train_loss += loss
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (batch % period == 0):
            loss, current = loss.item(), (batch + 1) * len(X)
            logging.info(f"\tTrain loss: {loss:.4f}  [{current:>5d}/{size:>5d}]") 
    
    train_loss /= len(dataloader)
    logging.info(f"\tAverage loss: {loss:.4f}") 

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

def compute_atomic(X_test, Y_test, device, model, modality, source_path, batch_size, iteration):
        
        X_test_resampled, y_test_resampled = resample(X_test, Y_test, replace=True, n_samples=len(Y_test), random_state=0+iteration)
        
        eval_set = DatasetBrainImages(X_test_resampled, y_test_resampled, modality=modality, source_path=source_path)
        eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)
        Y_prob, Y_pred  = eval(eval_loader, device, model)
        
        # Compute brier score
        brier_w = 0
        acc_w = 0
        brier_scores = np.zeros(Y_test.shape[1])
        acc_scores = np.zeros(Y_test.shape[1])

        for i in range(Y_test.shape[1]):    
            brier_scores[i] = brier_score_loss(y_test_resampled.iloc[:,i], Y_prob[:, i])
            acc_scores[i] = balanced_accuracy_score(y_test_resampled.iloc[:,i], Y_pred[:, i])
            
            brier_w += brier_scores[i] * (Y_test.iloc[:,i].sum() / Y_test.shape[0])
            acc_w += acc_scores[i] * (Y_test.iloc[:,i].sum() / Y_test.shape[0])

        # Store results
        score_dict = {
               'auprc_macro': average_precision_score(y_test_resampled, Y_prob, average='macro'),
               'auprc_weighted': average_precision_score(y_test_resampled, Y_prob, average='weighted'),
               'auroc_macro': roc_auc_score(y_test_resampled, Y_prob, average='macro'),
               'auroc_weighted': roc_auc_score(y_test_resampled, Y_prob, average='weighted'),
               'brier_macro': brier_scores.mean(),
               'brier_weighted': brier_w / Y_test.shape[1],
               'balanced_accuracy_macro': acc_scores.mean(),
               'balanced_accuracy_weighted': acc_w / Y_test.shape[1],
               'f1_micro': f1_score(y_test_resampled, Y_pred, average='micro'),
               'hamming': hamming_loss(y_test_resampled, Y_pred),
               'subset_accuracy': accuracy_score(y_test_resampled, Y_pred),
        }

        return score_dict
  
def compute_scores(X_test, Y_test, device, model, modality, source_path, batch_size, logging, boot_iter):

    score_dict = {
            'auprc_macro': [],
            'auprc_weighted': [],
            'auroc_macro': [],
            'auroc_weighted': [],
            'brier_macro': [],
            'brier_weighted': [],
            'balanced_accuracy_macro': [],
            'balanced_accuracy_weighted': [],
            'f1_micro': [],
            'hamming': [],
            'subset_accuracy': [],
    }
    
    scores = []
    for i in range(boot_iter):
        logging.info(f"Bootstrapping iteration {i}")
        scores.append(compute_atomic(X_test, Y_test, device, model, modality, source_path, batch_size, i))

    # Aggregate scores
    for k,_ in score_dict.items():
        for dict in scores:
            score_dict[k].append(dict[k])

    logging.info(f"Mean scores with SE and 95% confidence intervals:\n")

    for k,v in score_dict.items():
        logging.info(f"{(k + ':').ljust(30)}{np.mean(v):.2f} ({np.std(v):.2f}) [{np.percentile(v, 2.5):.2f}, {np.percentile(v, 97.5):.2f}]")
    