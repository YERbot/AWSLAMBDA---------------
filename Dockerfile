FROM public.ecr.aws/lambda/python:3.9.2024.04.17.17-x86_64
COPY dida-model.tflite .
COPY lambda_function.py .
COPY tflite_runtime-2.7.0-cp39-cp39-manylinux2014_x86_64.whl .
RUN pip install Pillow
RUN pip install tflite
RUN pip install keras_image_helper
RUN pip install tflite_runtime-2.7.0-cp39-cp39-manylinux2014_x86_64.whl
RUN pip install requests

CMD ["lambda_function.lambda_handler"]
