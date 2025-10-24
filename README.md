# Dynamic Artificial Neural Network Framework

A flexible and easy-to-use Python framework for creating, training, and using neural networks with dynamic architecture. This framework allows you to build custom neural networks by adding, removing, or modifying layers on the fly.

## Features

- **Dynamic Architecture**: Add, remove, or insert layers at any position in the network
- **Multiple Activation Functions**: Support for sigmoid, ReLU, tanh, softmax, and linear activations
- **Flexible Training**: Mini-batch and full-batch training with customizable learning rates
- **Save/Load Models**: Persist trained models to disk and load them later
- **Multiple Use Cases**: Binary classification, multi-class classification, regression, and more
- **Easy to Use**: Simple API with helper functions for quick network creation

## Installation

1. Clone this repository:
```bash
git clone https://github.com/SKurnella/Dynamic-Artificial-Neural-Network-Framework.git
cd Dynamic-Artificial-Neural-Network-Framework
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from network_generator import create_network
import numpy as np

# Create a simple dataset (XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create a neural network
network = create_network([
    (2,),              # 2 input features
    (4, 'relu'),       # Hidden layer with 4 neurons
    (1, 'sigmoid')     # Output layer with 1 neuron
])

# Train the network
network.train(X, y, epochs=5000, learning_rate=0.1)

# Make predictions
predictions = network.predict(X)
print(predictions)
```

### Manual Layer Construction

```python
from network_generator import NeuralNetwork, Layer

# Create an empty network
network = NeuralNetwork()

# Add layers manually
network.add_layer(Layer(input_size=2, output_size=5, activation='tanh'))
network.add_layer(Layer(input_size=5, output_size=3, activation='relu'))
network.add_layer(Layer(input_size=3, output_size=1, activation='sigmoid'))

# Display network architecture
network.summary()
```

### Dynamic Modification

```python
# Remove the last layer
network.remove_layer()

# Add a new layer
network.add_layer(Layer(input_size=3, output_size=1, activation='tanh'))

# Insert a layer at a specific position
network.insert_layer(1, Layer(input_size=5, output_size=4, activation='relu'))
```

### Save and Load Models

```python
# Save trained model
network.save('my_model.pkl')

# Load model
loaded_network = NeuralNetwork()
loaded_network.load('my_model.pkl')

# Use loaded model
predictions = loaded_network.predict(X)
```

## API Reference

### NeuralNetwork Class

#### Methods

- `add_layer(layer)`: Add a layer to the end of the network
- `remove_layer(index=-1)`: Remove a layer at the specified index
- `insert_layer(index, layer)`: Insert a layer at a specific position
- `train(X, y, epochs, learning_rate, batch_size, verbose)`: Train the network
- `predict(X)`: Make predictions on input data
- `forward(X)`: Forward propagation through the network
- `save(filename)`: Save the model to a file
- `load(filename)`: Load a model from a file
- `summary()`: Print network architecture summary

### Layer Class

#### Constructor

```python
Layer(input_size, output_size, activation='sigmoid')
```

**Parameters:**
- `input_size` (int): Number of input neurons
- `output_size` (int): Number of output neurons
- `activation` (str): Activation function ('sigmoid', 'relu', 'tanh', 'softmax', 'linear')

### Helper Functions

#### create_network(architecture)

Create a neural network with the specified architecture.

**Parameters:**
- `architecture` (list): List of tuples defining the network structure
  - First tuple: `(input_size,)`
  - Subsequent tuples: `(neurons, activation)`

**Example:**
```python
network = create_network([
    (4,),              # 4 input features
    (10, 'relu'),      # Hidden layer with 10 neurons
    (3, 'softmax')     # Output layer with 3 classes
])
```

## Supported Activation Functions

- **sigmoid**: Standard sigmoid function, output range (0, 1)
- **relu**: Rectified Linear Unit, output range [0, ∞)
- **tanh**: Hyperbolic tangent, output range (-1, 1)
- **softmax**: Softmax function for multi-class classification
- **linear**: Linear activation (identity function)

## Examples

Run the examples script to see various use cases:

```bash
python examples.py
```

The examples include:
1. XOR Problem - Simple binary classification
2. Manual Construction - Building networks layer by layer
3. Dynamic Modification - Adding/removing layers
4. Multi-class Classification - Using softmax for multiple classes
5. Save and Load - Persisting models to disk
6. Deep Networks - Creating networks with many layers

You can also run the main network generator script:

```bash
python network_generator.py
```

## Use Cases

### Binary Classification

```python
network = create_network([
    (n_features,),
    (16, 'relu'),
    (1, 'sigmoid')
])
```

### Multi-class Classification

```python
network = create_network([
    (n_features,),
    (32, 'relu'),
    (16, 'relu'),
    (n_classes, 'softmax')
])
```

### Regression

```python
network = create_network([
    (n_features,),
    (64, 'relu'),
    (32, 'relu'),
    (1, 'linear')
])
```

## Requirements

- Python 3.6+
- NumPy >= 1.19.0

## License

This project is open source and available for educational and research purposes.

## Contributing

Contributions are welcome! Feel free to submit pull requests or open issues for bugs and feature requests.

## Author

SKurnella

## Acknowledgments

This framework was built to provide an educational tool for understanding neural networks and their dynamic construction.