import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Загрузка модели TensorFlow Lite
interpreter = tflite.Interpreter(model_path='dida-model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# Функция для предварительной обработки входных данных
def preprocess_input(x):
    x /= 255
    return x

def predict(url):
    # Загрузка изображения по URL
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((150, 150), Image.NEAREST)
    
    # Предварительная обработка изображения
    x = np.array(img, dtype='float32')
    X = np.array([x])
    X = preprocess_input(X)
    
    # Установка входного тензора
    interpreter.set_tensor(input_index, X)
    
    # Выполнение вывода
    interpreter.invoke()
    
    # Получение предсказаний
    preds = interpreter.get_tensor(output_index)
    
    # Определение класса на основе порога (0.5)
    if preds[0] < 0.5:
        prediction = "0 (дино)"
    else:
        prediction = "1 (дракон)"
        
    return prediction

def lambda_handler(event, context):
    url = event['url']
    prediction = predict(url)
    
    return {
        'statusCode': 200,
        'body': prediction
    }
