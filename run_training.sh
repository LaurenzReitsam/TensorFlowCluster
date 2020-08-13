docker build -t tensorflow_env .

docker run --name test_run -v C:/Users/Laurenz.Reitsam/Desktop/TensorFlowCluster/data_volume:/data_volume -it tensorflow_env

tensorboard --logdir=C:\Users\Laurenz.Reitsam\Desktop\TensorFlowCluster\data_volume\logs