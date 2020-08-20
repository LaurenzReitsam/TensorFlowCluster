FROM tensorflow/tensorflow:latest

COPY train.py /usr/src/app/

RUN pip install -q -U tfds-nightly
# RUN curl https://s3.amazonaws.com/img-datasets/mnist.npz --output /usr/src/app/mnist.npz
