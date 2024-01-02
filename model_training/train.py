from cnn import CNN

import numpy as np

from torch.optim import Adam
from torch.nn import NLLLoss

from torch.utils.data import DataLoader
from torch import float as torch_float
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
import random

class Train:

    def __init__(self, train_data: list, batch_size:int=5, epochs: int=4):

        random.shuffle(train_data)
        VAL_BOUNDRY = len(train_data) // 5
        self.train_data = DataLoader(train_data[VAL_BOUNDRY:], batch_size=batch_size)
        self.validation_data = DataLoader(train_data[0:VAL_BOUNDRY], batch_size=1)

        self.batch_size = batch_size
        self.epochs = epochs
        self. device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_model(self, name: str, model, opt, lossFn):
        train_loss_list = []
        validation_loss_list = []
        epoch_list = np.arange(self.epochs)
        print(f"Beginning training: {name}\n-------------------------------")

        for epoch in range(self.epochs):
            model.train()
            train_loss = 0
            train_correct = 0
            val_loss = 0
            val_correct = 0
            print(f"Starting Epoch: {epoch}\n-------------------------------")

            for x, y in tqdm(self.train_data):

                (x, y) = (x.to(self.device), y.to(self.device))

                pred = model(x)

                loss = lossFn(pred, y) # negative log likelihood loss will penalize exponentially for lower probabilities on the right category
                opt.zero_grad()
                loss.backward()
                opt.step()

                train_loss += (loss)

                train_correct += (torch.argmax(pred, dim=1) == y).type(
                    torch_float).sum().item()

            # append total loss to train_loss_list     
            train_loss_list[epoch] = train_loss

            print("Checking validation loss")

            with torch.no_grad():
                model.eval() #set model to evaluation mode

                for x, y in tqdm(self.validation_data):
                    x, y = x.to(self.device), y.to(self.device)

                    pred = model(x)
                    val_loss += float(lossFn(pred, y))
                    val_correct += (torch.argmax(pred, 1) == y).type(
                    torch_float).sum().item()

                #append total loss to validation_loss_list
                validation_loss_list[epoch] = (val_loss)


            print(f"Epoch: {epoch} --- train loss: {train_loss} --- train correct: {train_correct} / {len(self.train_data) * self.batch_size}\n")
            print(f"Epoch: {epoch} --- validation loss: {val_loss} --- validation correct: {val_correct} / {len(self.validation_data)}\n")

            plt.plot(epoch_list, train_loss_list, label = "training loss")
            plt.plot(epoch_list, validation_loss_list, label = "validation loss")
            plt.title(f"{name} validation and training error")
            plt.xlabel("Epoch")
            plt.ylabel("loss")
            plt.savefig(f"training_plots/training_{name}.png")
        
            print("Completed training\n-------------------------------")

        return validation_loss_list[-1]

# test_loss = 0
# test_correct = 0

# print("Beginning Testing\n-------------------------------")
# with torch.no_grad():
#     model.eval()
#     for x, y in tqdm(test_load):
#         pred = model(x)
#         loss = lossFn(pred, y)
#         test_loss += loss
#         test_correct += (torch.argmax(pred, 1) == y).type(
# 			torch_float).sum().item()
        
#     print(f"Testing --- test loss: {test_loss} --- test correct: {test_correct}\n")
#     print(f"Accuracy: {test_correct}/{len(test)}\n{test_correct / len(test)}%")


    

# model = CNN(filter_size1=4, filter_size2=4, pooling2=3, flattened_data_in=1568, 
#             flattened_data_out=800, classes=len(class_dict))
# opt = Adam(params=model.parameters(), lr=LR) # method to perform gradient descent

# lossFn = NLLLoss() # use negative log-likelihood as the loss function



    