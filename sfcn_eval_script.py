import sys
import os
import argparse
import logging

import pandas as pd
import numpy as np

from utils.utils import load_data
from utils.sfcn_utils import DatasetBrainImages
from utils.sfcn_train import compute_scores, eval
from utils.sfcn_model import SFCN

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, average_precision_score

import torch
from torch.utils.data import DataLoader


# Parsing
parser = argparse.ArgumentParser()

parser.add_argument("-device_index", type=int, default=0)
parser.add_argument("-run", type=int, default=1)
parser.add_argument("-epoch", type=int, default=5)
parser.add_argument("-modality", type=str, default="T1w")
parser.add_argument("-loss", type=str, default="bce")
parser.add_argument("-sampling", type=str, default="none")
parser.add_argument("-boot_iter", type=int, default=1)
parser.add_argument("-eval_mode", type=bool, default="multi")
parser.add_argument("-source_path", type=str, default="/t1images/")

# Parse the arguments
args = parser.parse_args()

device_index =  args.device_index
run = args.run # run required for checkpoint loading
epoch = args.epoch # epoch required for checkpoint loading
modality = args.modality
loss = args.loss # loss required for checkpoint loading
sampling = args.sampling
boot_iter = args.boot_iter
eval_mode = args.eval_mode
source_path = args.source_path

# Set up paths
base_path = os.path.expanduser("~") + "/comorbidity_prediction/"
logs_path = base_path + "logs/"
checkpoints_path = base_path + "checkpoints/"

# Configure logging settings
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=f'{logs_path}evaluation_run_{run}_epoch_{epoch}__{modality}_{loss}_{sampling}_{boot_iter}_{eval_mode}.log',
                    filemode='w')

console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# Log variables
logging.info(f"base_path: {base_path}")
logging.info(f"source_path: {source_path}")
logging.info(f"logs_path: {logs_path}")
logging.info(f"checkpoints_path: {checkpoints_path}")
logging.info(f"run: {run}")
logging.info(f"epoch: {epoch}")
logging.info(f"modality: {modality}")
logging.info(f"loss: {loss}")
logging.info(f"sampling: {sampling}")
logging.info(f"eval_mode: {eval_mode}")
logging.info(f"boot_iter: {boot_iter}")

# Device
device = "cuda:" + str(device_index)
logging.info(f"device: {device}")

# Load and split data
X, _, Y = load_data('classification_t1')
X, Y = X.iloc[:,0], Y.iloc[:,1:]
_, X_test, _, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Set batch size
batch_size = 8
logging.info(f"batch size: {batch_size}")

# Instantiate model and load params from checkpoint
model = SFCN(output_dim=13)
model.to(device)
model.load_state_dict(torch.load(checkpoints_path + f"run_{run}_sfcn_{modality}_{loss}_{sampling}_epoch_{epoch}.pth"))

# Compute scores
if eval_mode == 'multi':
    compute_scores(X_test, Y_test, device, model, modality, source_path, batch_size, logging, boot_iter)
elif eval_mode == 'binary':
    auprc_scores = {}
    auroc_scores = {}

    for label in Y_test.columns:
         auprc_scores[label] = []
         auroc_scores[label] = []

    for i in range(boot_iter):
        logging.info(f"Bootstrapping iteration {i}")
        X_test_resampled, Y_test_resampled = resample(X_test, Y_test, replace=True, n_samples=len(Y_test), random_state=0+i)

        eval_set = DatasetBrainImages(X_test_resampled, Y_test_resampled, modality=modality, source_path=source_path)
        eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)
        Y_prob, _  = eval(eval_loader, device, model)
    
        for i, label in enumerate(Y_test.columns):
             auprc_scores[label].append(average_precision_score(Y_test_resampled.iloc[:, i], Y_prob[:, i]))
             auroc_scores[label].append(roc_auc_score(Y_test_resampled.iloc[:, i], Y_prob[:, i]))

    logging.info(f"Mean scores with SE and 95% confidence intervals:")
    logging.info(f"AUPRC:")
    for k,v in auprc_scores.items():
        logging.info(f"{(k + ':').ljust(50)}{np.mean(v):.2f} ({np.std(v):.2f}) [{np.percentile(v, 2.5):.2f}, {np.percentile(v, 97.5):.2f}]")
    logging.info(f"AUROC:")
    for k,v in auroc_scores.items():
        logging.info(f"{(k + ':').ljust(50)}{np.mean(v):.2f} ({np.std(v):.2f}) [{np.percentile(v, 2.5):.2f}, {np.percentile(v, 97.5):.2f}]")