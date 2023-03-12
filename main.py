from flask import  Flask, render_template, request, redirect, url_for
import torch
from model_settings import MODEL_POOL
from generator import StyleGANGenerator
import numpy as np
import cv2
import random


app = Flask(__name__)

generator = StyleGANGenerator()
model_name = "stylegan_ffhq"
latent_space_type = "W"
attr = ""
seed = 0
latent_vectors = 0
dic = {}

def sample_vectors(generator, num, latent_space_type='Z', seed=0):
  np.random.seed(seed)
  vectors = generator.sample(num, latent_space_type)
  if latent_space_type == 'W':
    vectors = torch.from_numpy(vectors).type(torch.FloatTensor).to(generator.run_device)
    vectors = generator.get_value(generator.model.mapping(vectors))
  return vectors

def save_image(images, col, size=512):
    num, height, width, channels = images.shape
    assert num % col == 0
    row = num // col

    fused_image = np.zeros((size * row, size * col, channels), dtype=np.uint8)

    for idx, image in enumerate(images):
        i, j = divmod(idx, col)
        y = i * size
        x = j * size
        if height != size or width != size:
            image = cv2.resize(image, (size, size))
        fused_image[y:y + size, x:x + size] = image

    fused_image = np.asarray(fused_image, dtype=np.uint8)

    cv2.imwrite(".\static\image.jpg",  cv2.cvtColor( fused_image, cv2.COLOR_RGB2BGR))


@app.route('/', methods=['GET', 'POST'])
def home():
    global seed
    if request.method == 'POST':
        # Get the values of the buttons
        seed = request.form.get("seed")
        button1 = request.form.get('button1') == 'True'
        button2 = request.form.get('button2') == 'True'
        button3 = request.form.get('button3') == 'True'
        button4 = request.form.get('button4') == 'True'
        button5 = request.form.get('button5') == 'True'

        global attr
        global latent_vectors
        global dic
        dic = {}
        attr = ""
        print(seed, button1, button2,button3, button4, button5)
        if button1 :
            dic["age"] = 0
            attr += "age,"
        if button2 :
            dic["eyeglasses"] =0
            attr += "eyeglasses,"
        if button3 :
            dic["gender"] = 0
            attr += "gender,"
        if button4 :
            dic["pose"] =0
            attr += "pose,"
        if button5 :
            dic["smile"] =0
            attr += "smile,"
        attr = attr[:-1]
        latent_vectors = sample_vectors(generator, 1, latent_space_type, int(seed))

        if latent_space_type == 'W':
            kwargs = {'latent_space_type': 'W', 'generate_style' : True}
        else:
            kwargs = {}
        images = generator.synthesize(latent_vectors, **kwargs)['image']
        save_image(images, 1)
        print(attr, dic)
        return redirect(url_for('synthesis', attr = attr, dic = dic))
    
    return render_template('home.html')

@app.route('/synthesis', methods = ['GET', 'POST'])
def synthesis():
    boundaries = {}
    global attr
    global seed
    global latent_vectors
    global dic
    attr_list = list(dic.keys())
    for i, attr_name in enumerate(attr_list):
        boundary_name = f'{model_name}_{attr_name}'
        if  latent_space_type == 'W':
            boundaries[attr_name] = np.load(f'.\\boundaries\{boundary_name}_w_boundary.npy')
        else:
            boundaries[attr_name] = np.load(f'.\\boundaries\{boundary_name}_boundary.npy')

    if request.method == 'POST':
        # Get the values of the buttons
        buttons = {}
        for attr_name in attr_list:
            val = float(request.form.get(attr_name))
            if val != 0:
                buttons[attr_name] = float(request.form.get(attr_name))
            else :
                buttons[attr_name] = dic[attr_name]
        print(buttons)
        new_vector = latent_vectors.copy()
        for i, attr_name in enumerate(attr_list):
            new_vector += boundaries[attr_name]*buttons[attr_name]

        if latent_space_type == 'W':
            kwargs = {'latent_space_type': 'W', 'generate_style' : True}
        else:
            kwargs = {}
        images = generator.synthesize(new_vector, **kwargs)['image']
        save_image(images, 1)
        dic = buttons
        print(dic)
        return redirect(url_for('synthesis', attr = attr, dic = buttons))


    return render_template('synthesis.html', attr = attr, dic = dic)



if __name__ == '__main__':
    app.run(debug=True)