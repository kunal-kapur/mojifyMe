import os
from PIL import Image
import random
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from train import Train
from cnn import CNN
from tqdm import tqdm
from torch import float as torch_float



category_dict = {}

# indicate which 
CWD = os.getcwd()
BATCH_SIZE = 16
LR = 0.001
EPOCHS=15

train = []
num = 0

pil_to_tensor = ToTensor()


print("Loading test and training images...........")
for root, dirs, files in os.walk(f"{CWD}/model_training/archive/train"):
    for dir in dirs:
        if dir not in ["happy", "sad"]:
            continue
        path = f"{root}/{dir}"
        for file in os.listdir(path):
            with Image.open(f"{path}/{file}") as img:
                train.append((pil_to_tensor(img), num))

        
        # increment the counter for the 
        category_dict[num] = dir
        num += 1

# get mapping for actual categories to values
inv_map = {v: k for k, v in category_dict.items()}

test = []   
for root, dirs, files in os.walk(f"{CWD}/model_training/archive/test"):
    for dir in dirs:
        if dir not in ["happy", "sad"]:
            continue
        path = f"{root}/{dir}"
        for file in os.listdir(path):
            with Image.open(f"{path}/{file}") as img:
                test.append((pil_to_tensor(img), inv_map[dir]))

print("Done")

random.shuffle(test)

train_instance = Train(train_data=train, batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LR)


model = CNN(num_classes=len(category_dict))

opt, lossFn, validation_loss = train_instance.train_model(name="model1", model=model)

#We load the test data separately
test_load = DataLoader(test, batch_size=1)

test_loss = 0
test_correct = 0

print("Beginning Testing\n--------------------------------------\n")
with torch.no_grad():
    model.eval()
    for x, y in tqdm(test_load):
        pred = model(x)
        loss = lossFn(pred, y)
        test_loss += loss
        test_correct += (torch.argmax(pred, 1) == y).type(
			torch_float).sum().item()
        
    print(f"Testing --- test loss: {test_loss} --- test correct: {test_correct}\n")
    print(f"Accuracy: {test_correct}/{len(test)}\n{test_correct / len(test)}%")


#torch.save(model.state_dict(), f"{os.getcwd()}/model.pt")















