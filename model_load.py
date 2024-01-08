from model_training.cnn import CNN
import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from PIL import ImageOps
import random


emojii_dict = {"happy": ["&#128512;", "&#128513;", "&#128515;", "&#128516;"],
               "sad": ["&#128554;", "&#128557;", "128560"]}

mapping = {0: "happy", 1: "sad"}

MODEL_PATH =  "model.pt"
pil_to_tensor = ToTensor()



def transform_and_predict(path_to_image):
    image_transformed = None
    with Image.open(path_to_image) as img:
        img = img.resize((48, 48))
        img = ImageOps.grayscale(img)
        image_transformed = pil_to_tensor(img)
        image_transformed = image_transformed[None, :, :, :]

        print(image_transformed.size())
    with torch.no_grad():
        pred_arr = model(image_transformed)
        pred = int(torch.argmax(pred_arr, 1))
        feeling = mapping[pred]
        return feeling
        # return random.choice(emojii_dict[feeling])


model = CNN(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()


happy = 0
sad = 0
for img_name in os.listdir("model_training/archive/test/happy"):
    feeling = transform_and_predict(f"model_training/archive/test/happy/{img_name}")
    if feeling == "happy":
        happy += 1
    else:
        sad += 1
print("Happy", happy)
print("Sad", sad)
    

