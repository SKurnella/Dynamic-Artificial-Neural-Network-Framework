#!/usr/bin/env python3
"""
Example usage of the Dynamic Artificial Neural Network Framework.
This script demonstrates various ways to create and train neural networks.
"""

import numpy as np
from network_generator import NeuralNetwork, Layer, create_network


def example_1_xor_problem():
    """Example 1: Solving XOR problem with a simple neural network."""
    print("\n" + "="*70)
    print("EXAMPLE 1: XOR Problem")
    print("="*70)
    
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Create network using the helper function
    network = create_network([
        (2,),              # 2 input features
        (4, 'relu'),       # Hidden layer with 4 neurons and ReLU activation
        (1, 'sigmoid')     # Output layer with 1 neuron and sigmoid activation
    ])
    
    network.summary()
    
    # Train the network
    network.train(X, y, epochs=5000, learning_rate=0.1, verbose=False)
    
    # Make predictions
    print("\nResults:")
    predictions = network.predict(X)
    for input_val, pred, true_val in zip(X, predictions, y):
        print(f"Input: {input_val}, Predicted: {pred[0]:.4f}, True: {true_val[0]}")


def example_2_manual_construction():
    """Example 2: Manually constructing a neural network layer by layer."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Manual Layer-by-Layer Construction")
    print("="*70)
    
    # Create an empty network
    network = NeuralNetwork()
    
    # Manually add layers one by one
    network.add_layer(Layer(input_size=2, output_size=5, activation='tanh'))
    network.add_layer(Layer(input_size=5, output_size=3, activation='relu'))
    network.add_layer(Layer(input_size=3, output_size=1, activation='sigmoid'))
    
    network.summary()
    
    # Generate simple dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Train
    network.train(X, y, epochs=3000, learning_rate=0.05, verbose=False)
    
    print("\nFinal predictions:")
    predictions = network.predict(X)
    for input_val, pred in zip(X, predictions):
        print(f"Input: {input_val}, Predicted: {pred[0]:.4f}")


def example_3_dynamic_modification():
    """Example 3: Dynamic modification of network architecture."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Dynamic Network Modification")
    print("="*70)
    
    # Create initial network
    network = create_network([
        (2,),
        (3, 'relu'),
        (1, 'sigmoid')
    ])
    
    print("Initial architecture:")
    network.summary()
    
    # Remove the last layer
    print("\nRemoving last layer...")
    network.remove_layer()
    
    # Add a new output layer with different configuration
    print("Adding new output layer with tanh activation...")
    network.add_layer(Layer(input_size=3, output_size=1, activation='tanh'))
    
    print("\nModified architecture:")
    network.summary()


def example_4_multi_class_classification():
    """Example 4: Multi-class classification with softmax."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Multi-class Classification")
    print("="*70)
    
    # Generate synthetic multi-class dataset
    np.random.seed(42)
    
    # Create 3 clusters of data
    n_samples_per_class = 30
    
    # Class 0: centered around (0, 0)
    X_class0 = np.random.randn(n_samples_per_class, 2) * 0.5 + np.array([0, 0])
    
    # Class 1: centered around (2, 2)
    X_class1 = np.random.randn(n_samples_per_class, 2) * 0.5 + np.array([2, 2])
    
    # Class 2: centered around (0, 2)
    X_class2 = np.random.randn(n_samples_per_class, 2) * 0.5 + np.array([0, 2])
    
    X = np.vstack([X_class0, X_class1, X_class2])
    
    # One-hot encoded labels
    y = np.zeros((n_samples_per_class * 3, 3))
    y[0:n_samples_per_class, 0] = 1
    y[n_samples_per_class:2*n_samples_per_class, 1] = 1
    y[2*n_samples_per_class:, 2] = 1
    
    # Create network for multi-class classification
    network = create_network([
        (2,),              # 2 input features
        (8, 'relu'),       # Hidden layer with 8 neurons
        (3, 'softmax')     # Output layer with 3 classes
    ])
    
    network.summary()
    
    # Train the network
    print("Training...")
    network.train(X, y, epochs=2000, learning_rate=0.01, batch_size=30, verbose=False)
    
    # Test predictions
    print("\nSample predictions:")
    test_samples = np.array([
        [0, 0],    # Should be class 0
        [2, 2],    # Should be class 1
        [0, 2]     # Should be class 2
    ])
    
    predictions = network.predict(test_samples)
    for i, (sample, pred) in enumerate(zip(test_samples, predictions)):
        predicted_class = np.argmax(pred)
        confidence = pred[predicted_class]
        print(f"Sample {sample}: Class {predicted_class} (confidence: {confidence:.4f})")
        print(f"  All probabilities: {pred}")


def example_5_save_and_load():
    """Example 5: Saving and loading models."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Save and Load Model")
    print("="*70)
    
    # Create and train a simple network
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    network = create_network([
        (2,),
        (4, 'relu'),
        (1, 'sigmoid')
    ])
    
    print("Training original network...")
    network.train(X, y, epochs=3000, learning_rate=0.1, verbose=False)
    
    # Save the model
    model_path = "example_model.pkl"
    network.save(model_path)
    
    # Load the model into a new network
    loaded_network = NeuralNetwork()
    loaded_network.load(model_path)
    
    # Compare predictions
    original_pred = network.predict(X)
    loaded_pred = loaded_network.predict(X)
    
    print("\nComparing predictions (should be identical):")
    print("Input | Original | Loaded")
    print("-" * 35)
    for inp, orig, load in zip(X, original_pred, loaded_pred):
        print(f"{inp} | {orig[0]:.4f}   | {load[0]:.4f}")


def example_6_deep_network():
    """Example 6: Creating a deep neural network."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Deep Neural Network")
    print("="*70)
    
    # Create a deep network with multiple hidden layers
    network = create_network([
        (10,),             # 10 input features
        (64, 'relu'),      # First hidden layer
        (32, 'relu'),      # Second hidden layer
        (16, 'relu'),      # Third hidden layer
        (8, 'relu'),       # Fourth hidden layer
        (1, 'sigmoid')     # Output layer
    ])
    
    network.summary()
    
    # Generate synthetic data for binary classification
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 10)
    # Target: 1 if sum of features > 0, else 0
    y = (X.sum(axis=1, keepdims=True) > 0).astype(float)
    
    # Train
    print("Training deep network...")
    network.train(X, y, epochs=1000, learning_rate=0.01, batch_size=32, verbose=False)
    
    # Evaluate
    predictions = network.predict(X)
    accuracy = np.mean((predictions > 0.5) == y)
    print(f"\nTraining accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("DYNAMIC ARTIFICIAL NEURAL NETWORK FRAMEWORK - EXAMPLES")
    print("="*70)
    
    # Run all examples
    example_1_xor_problem()
    example_2_manual_construction()
    example_3_dynamic_modification()
    example_4_multi_class_classification()
    example_5_save_and_load()
    example_6_deep_network()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70 + "\n")
