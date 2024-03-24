import sys
import os
import argparse
import logging

from utils.utils import load_data, generate_undersampled_set, generate_oversampled_set
from utils.sfcn_utils import DatasetBrainImages
from utils.sfcn_train import train, test, train_focal, test_focal
from utils.sfcn_model import SFCN

from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import DataLoader

# Parsing
parser = argparse.ArgumentParser()

parser.add_argument("-device_index", type=int, default=0)
parser.add_argument("-epochs", type=int, default=1)
parser.add_argument("-run", type=int, default=1)
parser.add_argument("-modality", type=str, default="T1w")
parser.add_argument("-loss", type=str, default="bce")
parser.add_argument("-sampling", type=str, default="none")
parser.add_argument("-source_path", type=str, default="/t1images/")

# Parse the arguments
args = parser.parse_args()

device_index =  args.device_index
epochs = args.epochs
run = args.run
modality = args.modality
loss = args.loss
sampling = args.sampling
source_path = args.source_path

# Set up paths
base_path = os.path.expanduser("~") + "/comorbidity_prediction/"
logs_path = base_path + "logs/"
checkpoints_path = base_path + "checkpoints/"

# Configure logging settings
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=f'{logs_path}training_run_{run}_sfcn_{modality}_{loss}_{sampling}.log',
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
logging.info(f"modality: {modality}")
logging.info(f"loss: {loss}")
logging.info(f"sampling: {sampling}")

# Device
device = "cuda:" + str(device_index)
logging.info(f"device: {device}")

# Load and split data
X, _, Y = load_data("classification_t1")
X, Y = X.iloc[:,0], Y.iloc[:,1:]

if sampling == "none":
    X_train, _, Y_train, _ = train_test_split(X, Y, test_size=0.25, random_state=0)
elif sampling == "under":
    X_under, Y_under = generate_undersampled_set(X, Y)
    X_train, _, Y_train, _ = train_test_split(X_under, Y_under, test_size=0.25, random_state=0)
elif sampling == "over":
    X_train, _, Y_train, _ = train_test_split(X, Y, test_size=0.25, random_state=0)
    X_over, Y_over = generate_oversampled_set(X_train, Y_train)
    X_train, Y_train = X_over, Y_over
else:
    pass

# Create dataset and dataloader objects
training_data = DatasetBrainImages(X_train, Y_train, modality=modality, source_path=source_path) 

batch_size = 8
logging.info(f"batch size: {batch_size}")
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True) 

# Instantiate model-related objects
model = SFCN(output_dim=13)
model.to(device)

if loss == "bce":
    loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(epochs):
    logging.info(f"Epoch {epoch + 1}/{epochs}")

    if loss == "bce":
        train(train_dataloader, device, model, loss_fn, optimizer, logging)
    elif loss == "focal":
        train_focal(train_dataloader, device, model, optimizer, logging)

    torch.save(model.state_dict(), checkpoints_path +  f"run_{run}_sfcn_{modality}_{loss}_{sampling}_epoch_{epoch}.pth")