import sys
import os
import argparse
import logging

from utils.utils import load_data, generate_undersampled_set, generate_oversampled_set
from utils.sfcn_utils import DatasetBrainImages
from utils.sfcn_train import compute_scores
from utils.sfcn_model import SFCN

from sklearn.model_selection import train_test_split

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
source_path = args.source_path

# Set up paths
base_path = os.path.expanduser("~") + "/comorbidity_prediction/"
logs_path = base_path + "logs/"
checkpoints_path = base_path + "checkpoints/"

# Configure logging settings
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=f'{logs_path}evaluation_run_{run}_epoch_{epoch}__{modality}_{loss}_{sampling}_{boot_iter}.log',
                    filemode='w')
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# Log path variables
logging.info(f"base_path: {base_path}")
logging.info(f"source_path: {source_path}")
logging.info(f"logs_path: {logs_path}")
logging.info(f"checkpoints_path: {checkpoints_path}")
logging.info(f"run: {run}")
logging.info(f"epoch: {epoch}")
logging.info(f"modality: {modality}")
logging.info(f"loss: {loss}")
logging.info(f"sampling: {sampling}")
logging.info(f"boot_iter: {boot_iter}")

# Device
device = "cuda:" + str(device_index)
logging.info(f"device: {device}")

# Load and split data
X, _, Y = load_data('classification_t1')
X, Y = X.iloc[:,0], Y.iloc[:,1:]

if sampling == "none":
    _, X_test, _, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
elif sampling == "under":
    X_under, Y_under = generate_undersampled_set(X, Y)
    _, X_test, _, Y_test = train_test_split(X_under, Y_under, test_size=0.25, random_state=0)
elif sampling == "over":
    X_over, Y_over = generate_oversampled_set(X, Y)
    _, X_test, _, Y_test = train_test_split(X_over, Y_over, test_size=0.25, random_state=0)
else:
    pass

# Create dataset
test_data = DatasetBrainImages(X_test, Y_test, modality=modality, source_path=source_path)

# Set batch size
batch_size = 8
logging.info(f"batch size: {batch_size}")

# Instantiate model and load params from checkpoint
model = SFCN(output_dim=13)
model.to(device)
model.load_state_dict(torch.load(checkpoints_path + f"run_{run}_sfcn_{modality}_{loss}_{sampling}_epoch_{epoch}.pth"))

# Compute scores
compute_scores(X_test, Y_test, device, model, batch_size, logging, boot_iter)