## Docker Image Definition

This directory defines the docker image, which is executed by every worker in the Kubernetes cluster.

The **train.py** file contains the model definition and the training definition.

For local testing the **docker-compose.yml** can be used to start several workers on a local machine instead running Kubernetes.
