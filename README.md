# Federated-Learning
A simple implementation of Federated Learning using MNIST data using Flower Framework.

First you must install Flower environment (https://flower.dev/docs/).

Run serverad.py this will start Federated Learning based on initial parameters (accuracy should be v low).

Then run client files. I am using my own model but you can use any other model such as "convnet".

after each federated round accuracy of global model(on serverad.py) will start improving. You can tinker with parameters as per your requirement
