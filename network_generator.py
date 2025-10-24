#!/usr/bin/env python3
"""
Dynamic Artificial Neural Network Framework
A flexible framework for creating, training, and using neural networks with dynamic architecture.
"""

import numpy as np
import pickle
import json
from typing import List, Tuple, Optional, Callable


class ActivationFunction:
    """Base class for activation functions."""
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid function."""
        s = ActivationFunction.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU function."""
        return (x > 0).astype(float)
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Tanh activation function."""
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of tanh function."""
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Softmax activation function."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    @staticmethod
    def linear(x: np.ndarray) -> np.ndarray:
        """Linear activation function (identity)."""
        return x
    
    @staticmethod
    def linear_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of linear function."""
        return np.ones_like(x)


class Layer:
    """Base class for neural network layers."""
    
    def __init__(self, input_size: int, output_size: int, activation: str = 'sigmoid'):
        """
        Initialize a neural network layer.
        
        Args:
            input_size: Number of input neurons
            output_size: Number of output neurons
            activation: Activation function name ('sigmoid', 'relu', 'tanh', 'softmax', 'linear')
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        # Initialize weights using He initialization for ReLU, Xavier for others
        if activation == 'relu':
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        else:
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)
        
        self.biases = np.zeros((1, output_size))
        
        # Cache for backpropagation
        self.input = None
        self.z = None
        self.output = None
        
        # Gradients
        self.dweights = None
        self.dbiases = None
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Forward propagation through the layer.
        
        Args:
            input_data: Input data of shape (batch_size, input_size)
        
        Returns:
            Output of the layer after activation
        """
        self.input = input_data
        self.z = np.dot(input_data, self.weights) + self.biases
        
        # Apply activation function
        if self.activation == 'sigmoid':
            self.output = ActivationFunction.sigmoid(self.z)
        elif self.activation == 'relu':
            self.output = ActivationFunction.relu(self.z)
        elif self.activation == 'tanh':
            self.output = ActivationFunction.tanh(self.z)
        elif self.activation == 'softmax':
            self.output = ActivationFunction.softmax(self.z)
        elif self.activation == 'linear':
            self.output = ActivationFunction.linear(self.z)
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")
        
        return self.output
    
    def backward(self, d_output: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward propagation through the layer.
        
        Args:
            d_output: Gradient of loss with respect to layer output
            learning_rate: Learning rate for weight updates
        
        Returns:
            Gradient of loss with respect to layer input
        """
        batch_size = self.input.shape[0]
        
        # Compute gradient with respect to activation
        if self.activation == 'sigmoid':
            d_z = d_output * ActivationFunction.sigmoid_derivative(self.z)
        elif self.activation == 'relu':
            d_z = d_output * ActivationFunction.relu_derivative(self.z)
        elif self.activation == 'tanh':
            d_z = d_output * ActivationFunction.tanh_derivative(self.z)
        elif self.activation == 'softmax':
            d_z = d_output  # Assumes softmax with cross-entropy loss
        elif self.activation == 'linear':
            d_z = d_output * ActivationFunction.linear_derivative(self.z)
        else:
            d_z = d_output
        
        # Compute gradients
        self.dweights = np.dot(self.input.T, d_z) / batch_size
        self.dbiases = np.sum(d_z, axis=0, keepdims=True) / batch_size
        d_input = np.dot(d_z, self.weights.T)
        
        # Update weights and biases
        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases
        
        return d_input
    
    def get_config(self) -> dict:
        """Get layer configuration."""
        return {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'activation': self.activation
        }


class NeuralNetwork:
    """Dynamic Artificial Neural Network class."""
    
    def __init__(self):
        """Initialize an empty neural network."""
        self.layers: List[Layer] = []
        self.loss_history: List[float] = []
    
    def add_layer(self, layer: Layer) -> None:
        """
        Add a layer to the network.
        
        Args:
            layer: Layer object to add
        """
        if self.layers and self.layers[-1].output_size != layer.input_size:
            raise ValueError(
                f"Layer input size ({layer.input_size}) must match "
                f"previous layer output size ({self.layers[-1].output_size})"
            )
        self.layers.append(layer)
    
    def remove_layer(self, index: int = -1) -> Layer:
        """
        Remove a layer from the network.
        
        Args:
            index: Index of layer to remove (default: last layer)
        
        Returns:
            Removed layer
        """
        if not self.layers:
            raise ValueError("No layers to remove")
        return self.layers.pop(index)
    
    def insert_layer(self, index: int, layer: Layer) -> None:
        """
        Insert a layer at a specific position.
        
        Args:
            index: Position to insert the layer
            layer: Layer object to insert
        """
        if index > 0 and self.layers[index-1].output_size != layer.input_size:
            raise ValueError(
                f"Layer input size ({layer.input_size}) must match "
                f"previous layer output size ({self.layers[index-1].output_size})"
            )
        if index < len(self.layers) and layer.output_size != self.layers[index].input_size:
            raise ValueError(
                f"Layer output size ({layer.output_size}) must match "
                f"next layer input size ({self.layers[index].input_size})"
            )
        self.layers.insert(index, layer)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation through the entire network.
        
        Args:
            X: Input data of shape (batch_size, input_features)
        
        Returns:
            Network output
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray, learning_rate: float) -> None:
        """
        Backward propagation through the entire network.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            learning_rate: Learning rate for weight updates
        """
        # Compute initial gradient (derivative of loss)
        d_output = y_pred - y_true
        
        # Backpropagate through layers in reverse order
        for layer in reversed(self.layers):
            d_output = layer.backward(d_output, learning_rate)
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, 
              learning_rate: float = 0.01, batch_size: Optional[int] = None,
              verbose: bool = True) -> None:
        """
        Train the neural network.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Training labels of shape (n_samples, n_outputs)
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            batch_size: Batch size for mini-batch training (None for full batch)
            verbose: Whether to print training progress
        """
        if batch_size is None:
            batch_size = X.shape[0]
        
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            n_batches = 0
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Compute loss (mean squared error)
                loss = np.mean((y_pred - y_batch) ** 2)
                epoch_loss += loss
                n_batches += 1
                
                # Backward pass
                self.backward(y_batch, y_pred, learning_rate)
            
            # Record average loss for the epoch
            avg_loss = epoch_loss / n_batches
            self.loss_history.append(avg_loss)
            
            # Print progress
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input data of shape (n_samples, n_features)
        
        Returns:
            Predictions of shape (n_samples, n_outputs)
        """
        return self.forward(X)
    
    def save(self, filename: str) -> None:
        """
        Save the neural network to a file.
        
        Args:
            filename: Path to save the model
        """
        model_data = {
            'layers': [
                {
                    'config': layer.get_config(),
                    'weights': layer.weights,
                    'biases': layer.biases
                }
                for layer in self.layers
            ],
            'loss_history': self.loss_history
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filename}")
    
    def load(self, filename: str) -> None:
        """
        Load a neural network from a file.
        
        Args:
            filename: Path to load the model from
        """
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.layers = []
        for layer_data in model_data['layers']:
            config = layer_data['config']
            layer = Layer(config['input_size'], config['output_size'], config['activation'])
            layer.weights = layer_data['weights']
            layer.biases = layer_data['biases']
            self.layers.append(layer)
        
        self.loss_history = model_data['loss_history']
        
        print(f"Model loaded from {filename}")
    
    def summary(self) -> None:
        """Print a summary of the network architecture."""
        print("\n" + "="*60)
        print("Neural Network Architecture Summary")
        print("="*60)
        
        if not self.layers:
            print("No layers in the network")
            return
        
        total_params = 0
        
        for i, layer in enumerate(self.layers):
            layer_params = layer.weights.size + layer.biases.size
            total_params += layer_params
            
            print(f"\nLayer {i+1}: Dense Layer")
            print(f"  Input Size:  {layer.input_size}")
            print(f"  Output Size: {layer.output_size}")
            print(f"  Activation:  {layer.activation}")
            print(f"  Parameters:  {layer_params}")
        
        print("\n" + "="*60)
        print(f"Total Parameters: {total_params}")
        print("="*60 + "\n")


def create_network(architecture: List[Tuple[int, str]]) -> NeuralNetwork:
    """
    Create a neural network with the specified architecture.
    
    Args:
        architecture: List of tuples (neurons, activation) for each layer
                     First tuple should only have (input_size,)
    
    Returns:
        Configured NeuralNetwork object
    
    Example:
        network = create_network([
            (4,),              # Input layer with 4 features
            (10, 'relu'),      # Hidden layer with 10 neurons and ReLU
            (3, 'softmax')     # Output layer with 3 neurons and softmax
        ])
    """
    if len(architecture) < 2:
        raise ValueError("Architecture must have at least input and output layer")
    
    network = NeuralNetwork()
    
    # First element is input size
    input_size = architecture[0][0]
    
    # Add layers
    for i in range(1, len(architecture)):
        output_size, activation = architecture[i]
        layer = Layer(input_size, output_size, activation)
        network.add_layer(layer)
        input_size = output_size
    
    return network


if __name__ == "__main__":
    # Example usage
    print("Dynamic Artificial Neural Network Framework")
    print("=" * 60)
    
    # Generate XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Create a neural network for XOR problem
    print("\nCreating a neural network for XOR problem...")
    network = create_network([
        (2,),              # 2 input features
        (4, 'relu'),       # Hidden layer with 4 neurons
        (1, 'sigmoid')     # Output layer with 1 neuron
    ])
    
    # Display network summary
    network.summary()
    
    # Train the network
    print("Training the network...")
    network.train(X, y, epochs=5000, learning_rate=0.1, verbose=True)
    
    # Make predictions
    print("\nPredictions:")
    predictions = network.predict(X)
    for i, (input_val, pred, true_val) in enumerate(zip(X, predictions, y)):
        print(f"Input: {input_val}, Predicted: {pred[0]:.4f}, True: {true_val[0]}")
    
    # Save the model
    network.save("xor_model.pkl")
    
    # Demonstrate loading
    print("\nLoading the saved model...")
    new_network = NeuralNetwork()
    new_network.load("xor_model.pkl")
    
    # Test loaded model
    print("\nTesting loaded model:")
    loaded_predictions = new_network.predict(X)
    for i, (input_val, pred) in enumerate(zip(X, loaded_predictions)):
        print(f"Input: {input_val}, Predicted: {pred[0]:.4f}")
