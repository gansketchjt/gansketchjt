from bottle import Bottle, run, static_file
import random
import os

ROOT = os.path.join('.', os.path.dirname(__file__))

# filter out weights that generate cats
cats_weight = []
for _, _, files in os.walk('weights'):
    for filename in files:
        if 'cat' in filename and '.pt' in filename:
            cats_weight.append(filename)


def folder(folder_name: str):
    return os.path.join(ROOT, folder_name)


def html(hname: str):
    return static_file(hname + '.html', root=folder('html'))


def css(name: str):
    return static_file(name, folder('css'))


app = Bottle()


@app.route('/')
def gan_hello():
    return html('index')


@app.route('/generate')
def gan_generate():
    # random generate a cat figure into the static folder
    # shuffle the weight we are going to use

    return html('generate')


@app.route('/generate/image')
def gan_generate_image():
    target, origin = random.choice([('standing_cat', 'cat'), ('gabled_church','church'), ('horse_on_a_side', 'horse')])
    os.system(
        f'python generate.py --ckpt ./weights/photosketch_{target}_aug.pt --origin_ckpt ./pretrained/stylegan2-{origin}/netG.pth --save_dir ./web/static --samples 1')
    return static_file('000000after.png', root=os.path.join(ROOT, 'static'))

@app.route('/generate/origin')
def gan_get_generate_origin():
    return static_file('000000before.png', root=os.path.join(ROOT, 'static'))


@app.route('/feature')
def gan_feature():
    return html('feature')


@app.route('/feature/image/<scale>')
def gan_generate_feature(scale):
    os.system(
        f'python ganspace.py --obj cat --comp_id 27 --scalar {scale} --layers 2,4 --samples 1 --ckpt weights/photosketch_standing_cat_aug.pt --save_dir web/static')
    return static_file('after_0.png', root=os.path.join(ROOT, 'static'))

@app.route('/feature/origin')
def gan_get_origin():
    return static_file('before_0.png', root=os.path.join(ROOT, 'static'))


@app.route('/css/<filename>')
def serve_css(filename: str):
    return css(filename)

@app.route('/sketch/<obj>')
def serve_sketch(obj:str):
    return static_file(obj, root=os.path.join(ROOT, 'static'))


run(app, reloader=True, debug=True, host='0.0.0.0', port=8080)
