FROM tensorflow/tensorflow

COPY train.py /usr/src/app/

RUN curl https://s3.amazonaws.com/img-datasets/mnist.npz --output /usr/src/app/mnist.npz
