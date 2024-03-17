import sys
import os
import argparse
import logging

import pandas as pd
import numpy as np

from utils.utils import load_data
from utils.cnn_utils import datasetT1
from utils.cnn_train import eval
from utils.cnn_model import SFCN

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, f1_score, hamming_loss

import torch
from torch.utils.data import DataLoader


# Parsing
parser = argparse.ArgumentParser()

parser.add_argument("-device_index", type=int, default=0)
parser.add_argument("-run", type=int, default=1)
parser.add_argument("-epoch", type=int, default=5)
parser.add_argument("-modality", type=str, default="T1w")
parser.add_argument("-loss", type=str, default="bce") 
parser.add_argument("-source_path", type=str, default="/t1images/")

# Parse the arguments
args = parser.parse_args()

device_index =  args.device_index
run = args.run # run required for checkpoint loading
epoch = args.epoch # epoch required for checkpoint loading
modality = args.modality
loss = args.loss # loss required for checkpoint loading
source_path = args.source_path

# Set up paths
base_path = os.path.expanduser("~") + "/comorbidity_prediction/"
run_path = base_path + "runs/"
checkpoints_path = base_path + "checkpoints/"

# Configure logging settings
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=f'{run_path}evaluation_run_{run}_epoch_{epoch}_modality_{modality}_loss_{loss}.log',
                    filemode='w')
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# Log path variables
logging.info(f"base_path: {base_path}")
logging.info(f"source_path: {source_path}")
logging.info(f"run_path: {run_path}")
logging.info(f"checkpoints_path: {checkpoints_path}")
logging.info(f"run: {run}")
logging.info(f"epoch: {epoch}")
logging.info(f"modality: {modality}")
logging.info(f"loss: {loss}")

# Device
device = "cuda:" + str(device_index)
logging.info(f"device: {device}")

# Load and split data
X, _, Y = load_data('classification_t1')
_, X_test, _, Y_test = train_test_split(X.iloc[:,0], Y.iloc[:,1:], test_size=0.25, random_state=0)

# Create dataset
test_data = datasetT1(X_test, Y_test, modality=modality, source_path=source_path)

# Set batch size
batch_size = 8
logging.info(f"batch size: {batch_size}")

# Instantiate model and load params from checkpoint
model = SFCN(output_dim=13)
model.to(device)
model.load_state_dict(torch.load(checkpoints_path + f"run_{run}_sfcn_{modality}_{loss}_epoch_{epoch}.pth"))

# Evaluation
auprc = []
auroc = []
brier = []
hamm = []
f1 = []

for i in range(100):
    X_test_resampled, y_test_resampled = resample(X_test, Y_test, replace=True, n_samples=len(Y_test), random_state=0+i)

    eval_data = datasetT1(X_test_resampled, y_test_resampled, modality=modality, source_path=source_path)
    eval_dataloader = DataLoader(eval_data, batch_size=batch_size, shuffle=False)
    y_prob, y_pred  = eval(eval_dataloader, device, model)

    # Compute brier score
    brier_scores = np.zeros(y_prob.shape[1])
    for i in range(y_prob.shape[1]):
        brier_scores[i] = brier_score_loss(y_test_resampled.iloc[:,i], y_prob[:,i])
    brier.append(brier_scores.mean())
    
    # Other metrics
    auprc.append(average_precision_score(y_test_resampled, y_prob, average='macro'))
    auroc.append(roc_auc_score(y_test_resampled, y_prob, average='macro'))
    f1.append(f1_score(y_test_resampled, y_pred, average='micro'))
    hamm.append(hamming_loss(y_test_resampled, y_pred))

logging.info(f"Mean scores for 3D-CNN with 95% confidence intervals:")
logging.info("AUPRC macro: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(auprc), np.percentile(auprc, 2.5), np.percentile(auprc, 97.5)))
logging.info("AUROC macro: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(auroc), np.percentile(auroc, 2.5), np.percentile(auroc, 97.5)))
logging.info("Brier score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(brier), np.percentile(brier, 2.5), np.percentile(brier, 97.5)))
logging.info("Hamming loss: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(hamm), np.percentile(hamm, 2.5), np.percentile(hamm, 97.5)))
logging.info("Micro Avg F1 score: {:.2f} [{:.2f}, {:.2f}]".format(np.mean(f1), np.percentile(f1, 2.5), np.percentile(f1, 97.5)))