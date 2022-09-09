from keras.models import load_model
import os
import tensorflow as tf
import flwr as fl
import numpy as np
from tensorflow import keras
input_shape = (28, 28, 1)
# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class My_Client(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.1,
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["acc"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_acc"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}


def main() -> None:

    # Load and compile Keras model


    #model=load_model('my_models1.h5')
    model=load_model('my_models1.h5')

    # Load a subset of data to simulate the local data partition
    (x_train, y_train), (x_test, y_test) = load_partition()

    # Start Flower client
    client = My_Client(model, x_train, y_train, x_test, y_test)

    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client,
        #root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
    )


def load_partition():
    """Load 1/10th of the training and test data to simulate a partition."""
    

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    x_train=x_train[:5000]
    y_train=y_train[:5000]
   
    x_test=x_test[:1000]
    y_test=y_test[:1000]
    
    
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    return (
        x_train,
        y_train,
    ), (
        x_test,
        y_test,
    )

if __name__ == "__main__":
    main()
    
