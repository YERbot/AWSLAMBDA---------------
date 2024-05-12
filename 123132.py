import tflite_runtime.interpreter as tflite
import numpy as np
from keras_image_helper import create_preprocessor

preprocessor = create_preprocessor('xception', target_size=(150, 150))

interpreter = tflite.Interpreter(model_path='clothing-model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]
url = 'https://www.bing.com/images/search?view=detailV2&ccid=7sh2X7py&id=6DC74D0CA30F2C36A64D5E914F7CD163CA7C88E7&thid=OIP.7sh2X7py31QXpwnhjc6jbgHaHa&mediaurl=https%3a%2f%2fth.bing.com%2fth%2fid%2fR.eec8765fba72df5417a709e18dcea36e%3frik%3d54h8ymPRfE%252bRXg%26riu%3dhttp%253a%252f%252fwww.rusultras.ru%252fimage%252fphoto%252fkargo-shtany-vintage-industries-pesochniy-kupit-v-moskve.jpg%26ehk%3drz1uLNNIPNoBVI%252bxzOKdbTi92cldwrQIc%252fOMRtKR%252f%252fM%253d%26risl%3d%26pid%3dImgRaw%26r%3d0&exph=900&expw=900&q=%d1%88%d1%82%d0%b0%d0%bd%d1%8b&simid=608022315916482520&FORM=IRPRST&ck=3C112EE78E43BAFDC09F22413464994B&selectedIndex=6&itb=0знерщт'

def predict(url): # функция предсказания и возвращения списка
    X = preprocessor.from_url(url)
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    return dict(zip(classes, preds[0]))


def lambda_handler(event, context):
    url = event['url']
    preds = predict(url)
    return {
        'StatusCode':200,
        'body':str(preds)
    }
