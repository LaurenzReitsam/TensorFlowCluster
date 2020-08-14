"""
Code with training definition here
"""

import tensorflow as tf
import numpy as np
import datetime

try:

    f = open('/usr/src/app/logs.txt', 'w')

    print("Starting procedure", file=f)

    def mnist_dataset(batch_size):
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
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


    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    per_worker_batch_size = 32
    num_workers = 2

    global_batch_size = per_worker_batch_size * num_workers

    print("Loading Data", file=f)
    multi_worker_dataset = mnist_dataset(global_batch_size)

    print("Building Model", file=f)
    with strategy.scope():
        multi_worker_model = build_and_compile_cnn_model()

    # print("Create TensorBoard Callback")
    # log_dir = "/data_volume/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    print("Fitting Model", file=f)
    multi_worker_model.fit(multi_worker_dataset,
                          epochs=60,
                          steps_per_epoch=60,
                          callbacks=[])

    print("Done!", file=f)

except Exception as e:
    print("---------------------")
    print(e, file=f)


f.close()
