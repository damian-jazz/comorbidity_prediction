import sys
import os
import argparse
import logging

import pandas as pd
import numpy as np

from utils.utils import load_data
from utils.cnn_utils import datasetT1
from utils.cnn_train import train_with_logging, test_with_logging
from utils.cnn_model import SFCN

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
parser.add_argument("-source_path", type=str, default="/t1images/")

# Parse the arguments
args = parser.parse_args()

device_index =  args.device_index
modality = args.modality
epochs = args.epochs
run = args.run
source_path = args.source_path

# Set up paths
base_path = os.path.expanduser("~") + "/comorbidity_prediction/"
run_path = base_path + "runs/"
checkpoints_path = base_path + "checkpoints/"

# Configure logging settings
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=f'{run_path}training_run_{run}_modality_{modality}.log',
                    filemode='w')
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# Log path variables
logging.info(f"Using base_path: {base_path}")
logging.info(f"Using source_path: {source_path}")
logging.info(f"Using run_path: {run_path}")
logging.info(f"Using checkpoints_path: {checkpoints_path}")

# Device
device = "cuda:" + str(device_index)
logging.info(f"Using {device} device")

# Load and split data
X, _, Y = load_data('classification_t1')
X_train, _, Y_train, _ = train_test_split(X.iloc[:,0], Y.iloc[:,1:], test_size=0.25, random_state=0)
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=1)

# Create dataset and dataloader objects
training_data = datasetT1(X_train, Y_train, modality=modality, source_path=source_path) 
test_data = datasetT1(X_test, Y_test, modality=modality, source_path=source_path) 

batch_size = 8
logging.info(f"Using batch size {batch_size}")
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True) 
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Instantiate model-related objects
model = SFCN(output_dim=13)
model.to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(epochs):
    logging.info(f"Epoch {epoch + 1}/{epochs}")

    train_with_logging(train_dataloader, device, model, loss_fn, optimizer, logging)
    test_with_logging(test_dataloader, device, model, loss_fn, logging)

    torch.save(model.state_dict(), checkpoints_path +  f"run_{run}_cnn_{modality}_epoch_{epoch}.pth")