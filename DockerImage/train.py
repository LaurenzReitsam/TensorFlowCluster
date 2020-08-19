"""
Code with training definition here
"""

import tensorflow as tf
import numpy as np
import datetime
import socket

DATAPATH = "/usr/src/app"
LOGPATH  = "/data_volume"
TBOARDPATH = "/data_volume/logs/fit"


hostname = socket.gethostname()
starting_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logfile_name = '{}/{}_train_logs_{}.txt'.format(LOGPATH, starting_time, hostname)

def logging(text):
    with open(logfile_name, 'a') as f:
        print("{}  -  {}".format(datetime.datetime.now().strftime("%H:%M:%S"), text), file=f)

def load_data(path):
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    return (x_train, y_train), (x_test, y_test)

def mnist_dataset():
    (x_train, y_train), _ = load_data('{}/mnist.npz'.format(DATAPATH))
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))\
                                   .shuffle(BUFFER_SIZE)\
                                   .batch(BATCH_SIZE)
    return train_dataset

def build_and_compile_cnn_model():
    model = tf.keras.Sequential([
      tf.keras.Input(shape=(28, 28)),
      tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.Conv2D(64, 3, activation='relu'),
      tf.keras.layers.Conv2D(128, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(64, activation='relu'),
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

#-----------------------------#
# setting constants:

BUFFER_SIZE = 10000
BATCH_SIZE_PER_REPLICA = 64
EPOCHS = 5

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

logging("Number of parallel workers: {}".format(strategy.num_replicas_in_sync))

#-----------------------------#
# building and running the model

logging("Loading data.")
multi_worker_dataset = mnist_dataset()

STEPS_PER_EPOCH = len(multi_worker_dataset)

logging("Building Model")
with strategy.scope():
    multi_worker_model = build_and_compile_cnn_model()

# print("Create TensorBoard Callback")
# log_dir = TBOARDPATH + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

logging("Fitting Model")
multi_worker_model.fit(multi_worker_dataset,
                      epochs=EPOCHS,
                      steps_per_epoch=STEPS_PER_EPOCH,
                      callbacks=[])


#-----------------------------#
# close session

toc = datetime.datetime.now()
running_time = toc-tic

hours, _rem = divmod(running_time.seconds, 3600)
minutes, seconds = divmod(_rem, 60)

logging("Done!")
logging("Total calculation time: {}:{}:{}".format(hours, minutes, seconds))
