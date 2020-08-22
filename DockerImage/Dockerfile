FROM tensorflow/tensorflow:latest

COPY train.py /usr/src/app/

RUN pip install -q -U tfds-nightly

# Download the mnist dataset
RUN python3 -c "import tensorflow_datasets; tensorflow_datasets.builder('mnist').download_and_prepare()"
