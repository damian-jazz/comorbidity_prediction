import sys
import os
import argparse
import logging

import pandas as pd
import numpy as np

from utils.utils import load_data
from utils.sfcn_utils import datasetT1
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
parser.add_argument("-source_path", type=str, default="/t1images/")

# Parse the arguments
args = parser.parse_args()

device_index =  args.device_index
epochs = args.epochs
run = args.run
modality = args.modality
loss = args.loss
source_path = args.source_path

# Set up paths
base_path = os.path.expanduser("~") + "/comorbidity_prediction/"
logs_path = base_path + "logs/"
checkpoints_path = base_path + "checkpoints/"

# Configure logging settings
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=f'{logs_path}training_run_{run}_sfcn_{modality}_{loss}.log',
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

# Device
device = "cuda:" + str(device_index)
logging.info(f"device: {device}")

# Load and split data
X, _, Y = load_data('classification_t1')
X_train, _, Y_train, _ = train_test_split(X.iloc[:,0], Y.iloc[:,1:], test_size=0.25, random_state=0)
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=1)

# Create dataset and dataloader objects
training_data = datasetT1(X_train, Y_train, modality=modality, source_path=source_path) 
test_data = datasetT1(X_test, Y_test, modality=modality, source_path=source_path) 

batch_size = 8
logging.info(f"batch size: {batch_size}")
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True) 
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Instantiate model-related objects
model = SFCN(output_dim=13)
model.to(device)

if loss == 'bce':
    loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(epochs):
    logging.info(f"Epoch {epoch + 1}/{epochs}")

    if loss == 'bce':
        train(train_dataloader, device, model, loss_fn, optimizer, logging)
        test(test_dataloader, device, model, loss_fn, logging)
    elif loss == 'focal':
        train_focal(train_dataloader, device, model, optimizer, logging)
        test_focal(test_dataloader, device, model, logging)
    
    torch.save(model.state_dict(), checkpoints_path +  f"run_{run}_sfcn_{modality}_{loss}_epoch_{epoch}.pth")