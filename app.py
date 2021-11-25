import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename, send_from_directory
# import model
from PIL import Image
import base64
import io
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

def model(img):
    print("model")
    
    image, result = model(img)
    print(result)
    return image, result
    # plt.imshow(image)

@app.route('/image')
def image():
    img = './input/images/test/0.png'
    # model.ownimage(img)
    image, result = model(img)
    im = Image.open(img)
    data = io.BytesIO()
    im.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())

    return render_template("image.html", img_data=encoded_img_data.decode('utf-8'), result=result)

if __name__ == '__main__':
    app.run(debug=True)