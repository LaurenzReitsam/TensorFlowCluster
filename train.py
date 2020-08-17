"""
Code with training definition here
"""

import tensorflow as tf
import numpy as np
import datetime
import socket

hostname = socket.gethostname()
starting_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logfile_name = '/data_volume/{}_train_logs_{}.txt'.format(starting_time, hostname)

def logging(text):
    with open(logfile_name, 'a') as f:
        print("{}  -  {}".format(datetime.datetime.now().strftime("%H:%M:%S"), text), file=f)

def load_data(path='/usr/src/app/mnist.npz'):
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    return (x_train, y_train), (x_test, y_test)

def mnist_dataset(batch_size):
    (x_train, y_train), _ = load_data()
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(1000).repeat().batch(batch_size)
    return train_dataset

def build_and_compile_cnn_model():
    model = tf.keras.Sequential([
      tf.keras.Input(shape=(28, 28)),
      tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10)
      ])

    model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])
    return model

#---------------------------------------------------------------------------#

tic = datetime.datetime.now()
logging("Starting script.")

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

per_worker_batch_size = 32
num_workers = 2

global_batch_size = per_worker_batch_size * num_workers

logging("Loading data.")
multi_worker_dataset = mnist_dataset(global_batch_size)

logging("Building Model")
with strategy.scope():
    multi_worker_model = build_and_compile_cnn_model()

print("Create TensorBoard Callback")
log_dir = "/data_volume/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

logging("Fitting Model")
multi_worker_model.fit(multi_worker_dataset,
                      epochs=60,
                      steps_per_epoch=60,
                      callbacks=[tensorboard_callback])

toc = datetime.datetime.now()
logging("Done!")

running_time = datetime.timedelta(toc-tic).total_seconds()
logging("Total calculation time: {} sec".format(running_time))
