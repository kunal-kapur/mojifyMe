from flask import Flask, render_template, flash, redirect, url_for
from flask import request
import os
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps
import random
from model_training.cnn import CNN

import torch
from torchvision.transforms import ToTensor



UPLOAD_FOLDER = f"{os.getcwd()}/upload_folder/"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH =  "model.pt"

pil_to_tensor = ToTensor()


app = Flask(__name__)

app.config['upload_folder'] = UPLOAD_FOLDER

model = CNN(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

emojii_dict = {"happy": ["128512"],
               "sad": ['&#128554;', '&#128557;', '128560']}

mapping = {1: "happy", 0: "sad"}



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def transform_and_predict(path_to_image):
    image_transformed = None
    with Image.open(path_to_image) as img:
        img = img.resize((48, 48))
        img = ImageOps.grayscale(img)
        image_transformed = pil_to_tensor(img)
        image_transformed = image_transformed[None, :, :, :]

    with torch.no_grad():
        pred_arr = model(image_transformed)
        pred = int(torch.argmax(pred_arr, 1))
        feeling = mapping[pred]
        return random.choice(emojii_dict[feeling])



@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print("Filename", filename)
            file.save(os.path.join(app.config['upload_folder'], filename))
            return redirect(url_for('result', name=filename))

    return render_template("index.html")

@app.route('/result/<name>', methods=['GET', 'POST'])
def result(name):
    res = transform_and_predict(f"{UPLOAD_FOLDER}/{name}")
    # if request.method == 'GET':
    #     return redirect('index')
    
    return render_template("result.html", icon=res).encode( "utf-8" )


if __name__ == '__main__':
    app.run()
    