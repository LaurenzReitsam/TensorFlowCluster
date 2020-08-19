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
EPOCHS = 60

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

STEPS_PER_EPOCH = BUFFER_SIZE // BATCH_SIZE

logging("Number of parallel workers: {}".format(strategy.num_replicas_in_sync))

#-----------------------------#
# loading the data

datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
mnist_train, mnist_test = datasets['train'], datasets['test']

# num_train_examples = info.splits['train'].num_examples
# num_test_examples = info.splits['test'].num_examples

def scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255
  return image, label

train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)

#-----------------------------#
# building and running the model

logging("Loading data.")
multi_worker_dataset = mnist_dataset(BATCH_SIZE)

logging("Building Model")
with strategy.scope():
    multi_worker_model = build_and_compile_cnn_model()

print("Create TensorBoard Callback")
log_dir = TBOARDPATH + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

logging("Fitting Model")
multi_worker_model.fit(train_dataset,
                      epochs=EPOCHS,
                      callbacks=[tensorboard_callback])


#-----------------------------#
# close session

toc = datetime.datetime.now()
running_time = toc-tic

hours, _rem = divmod(running_time.seconds, 3600)
minutes, seconds = divmod(_rem, 60)

logging("Done!")
logging("Total calculation time: {}:{}:{}".format(hours, minutes, seconds))
