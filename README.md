# TensorFlowCluster
Run TensorFlow on distributed workers

This is an example project to illustrate how to setup a TensorFlow model on distributed machines using Kubernetes and Docker.

The kubernetes setup is in the **setup_workers.yaml** file.

Every worker is running the same docker image that is defined **DockerImage** and available on Dockerhub. Every new commit pushes a new version to Dockerhub using Github Actions.
