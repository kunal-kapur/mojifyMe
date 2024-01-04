import os
from PIL import Image
import random
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from train import Train
from cnn import CNN


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
            img = Image.open(f"{path}/{file}")
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
            img = Image.open(f"{path}/{file}")
            test.append((pil_to_tensor(img), inv_map[dir]))
print("Done")

random.shuffle(test)

train_instance = Train(train_data=train, batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LR)


model1 = CNN(num_classes=len(category_dict))

train_instance.train_model(name="model1", model=model1)

#We load the test data separately
test_load = DataLoader(test, batch_size=1)











