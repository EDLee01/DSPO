# Simplified DSPO Implementation

A simplified, educational implementation of **Differentiable Sensor Placement Optimization (DSPO)** using synthetic data. This implementation demonstrates the core bi-level optimization concept without requiring large external datasets.

## Overview

DSPO is a framework that simultaneously optimizes:
1. **Sensor positions** in a spatial domain
2. **Field reconstruction models** that predict the full field from sensor observations

The optimization uses a bi-level approach:
- **Lower level**: Train reconstruction model with fixed sensor positions
- **Upper level**: Optimize sensor positions with fixed reconstruction model

## Features

### Synthetic Data Generation
- **Gaussian blobs**: Moving Gaussian peaks with temporal variation
- **Vortex flow**: Analytical vortex solutions with moving centers
- **Wave patterns**: Sine/cosine based spatial patterns
- **Combined data**: Mixture of different patterns

### Reconstruction Models
- **MLP**: Multi-layer perceptron with configurable architecture
- **CNN**: Convolutional neural network for 2D field reconstruction
- **ResMLP**: Residual MLP with skip connections
- **MultiScale**: Multi-scale processing for different feature sizes

### Sensor Optimization
- **Differentiable optimization** using PyTorch autograd
- **Constraint handling** for spatial bounds and minimum distances
- **Multiple loss components**: reconstruction, diversity, information content, coverage
- **Adaptive optimization** with learning rate scheduling and early stopping

### Visualization and Analysis
- Sensor position evolution tracking
- Field reconstruction comparisons
- Optimization convergence plots
- Comprehensive metrics (RMSE, correlation, etc.)

## Installation

### Requirements
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- SciPy

### Install Dependencies
```bash
pip install torch numpy matplotlib scipy
```

## Usage

### Basic Example

```python
import torch
from simplified_dspo import DSPOSolver, SyntheticDataGenerator

# Generate synthetic data
generator = SyntheticDataGenerator(width=64, height=64)
field_data, coords = generator.generate_combined_data(n_samples=100)

# Convert to tensor
field_tensor = torch.tensor(field_data, dtype=torch.float32)

# Create DSPO solver
solver = DSPOSolver(n_sensors=8, field_shape=(64, 64), model_type='mlp')

# Train with bi-level optimization
history = solver.train(field_tensor, n_outer_iterations=5)

# Visualize results
figures = solver.visualize_results(field_tensor, sample_indices=[0, 10, 20])
```

### Run Complete Demonstration

```python
# Run the demonstration script
python demo.py --data_type combined --model_type mlp --n_sensors 8

# Compare different sensor numbers
python demo.py --comparison --data_type gaussian --model_type cnn
```

### Available Data Types
- `gaussian`: Moving Gaussian blobs
- `vortex`: Vortex flow patterns
- `waves`: Wave-based patterns
- `combined`: Mixture of all patterns

### Available Model Types
- `mlp`: Multi-layer perceptron
- `cnn`: Convolutional neural network
- `resmlp`: Residual MLP
- `multiscale`: Multi-scale MLP

## Code Structure

```
simplified_dspo/
├── __init__.py              # Package initialization
├── synthetic_data.py        # Synthetic data generation
├── models.py               # Reconstruction models
├── sensor_optimization.py  # Sensor placement optimization
├── dspo_solver.py          # Main bi-level optimization framework
├── utils.py                # Utility functions (RBF, visualization)
└── demo.py                 # Demonstration script
```

## Key Classes

### `SyntheticDataGenerator`
Generates various types of synthetic 2D flow field data.

```python
generator = SyntheticDataGenerator(width=64, height=64)

# Generate different data types
gaussian_data, coords = generator.generate_gaussian_blobs(n_samples=100)
vortex_data, coords = generator.generate_vortex_flow(n_samples=100)
wave_data, coords = generator.generate_wave_patterns(n_samples=100)
combined_data, coords = generator.generate_combined_data(n_samples=100)
```

### `DSPOSolver`
Main class implementing the bi-level optimization framework.

```python
solver = DSPOSolver(
    n_sensors=8,                    # Number of sensors
    field_shape=(64, 64),          # Field dimensions
    model_type='mlp',              # Reconstruction model type
    spatial_bounds=(0, 1, 0, 1)    # Domain bounds
)

# Training
history = solver.train(
    field_data,
    n_outer_iterations=5,           # Bi-level iterations
    model_epochs_per_iteration=50,  # Epochs per model training
    sensor_iterations_per_iteration=100  # Steps per sensor optimization
)

# Prediction
sensor_obs, reconstructed = solver.predict(field_data)
```

### `SensorOptimizer`
Handles differentiable sensor position optimization.

```python
sensor_opt = SensorOptimizer(n_sensors=8, device='cpu')
sensor_opt.initialize_positions(method='uniform')
sensor_opt.setup_optimizer(learning_rate=0.01)

# Optimization step
gradients, loss_value = sensor_opt.step(loss_tensor)
```

## Configuration Options

### Model Configuration
```python
# MLP configuration
mlp_model = create_reconstructor(
    'mlp', 
    n_sensors=8, 
    output_shape=(64, 64),
    hidden_layers=[128, 256, 512, 256, 128],
    activation='gelu',
    dropout=0.1
)

# CNN configuration
cnn_model = create_reconstructor(
    'cnn',
    n_sensors=8,
    output_shape=(64, 64),
    fc_hidden=256,
    channels=[32, 64, 128, 64, 32]
)
```

### Loss Configuration
```python
loss_weights = {
    'reconstruction': 1.0,    # Primary reconstruction loss
    'diversity': 0.1,         # Sensor diversity (minimum distance)
    'information': 0.01,      # Information content (variance)
    'coverage': 0.05          # Spatial coverage
}
```

## Examples and Results

The implementation includes several demonstration examples:

1. **Basic DSPO**: Standard bi-level optimization with different data types
2. **Sensor comparison**: Performance analysis with varying sensor numbers
3. **Model comparison**: Comparison of different reconstruction models
4. **Visualization**: Comprehensive plotting and analysis tools

Results are automatically saved including:
- Sensor position evolution plots
- Field reconstruction comparisons
- Training convergence curves
- Quantitative metrics (RMSE, correlation, etc.)

## Educational Value

This simplified implementation is designed for:
- **Learning DSPO concepts** without complex dependencies
- **Experimenting with different configurations** easily
- **Understanding bi-level optimization** in practice
- **Developing new sensor placement strategies**

## Differences from Original Implementation

This simplified version:
- Uses synthetic data instead of large flow datasets
- Implements simplified RBF interpolation using SciPy
- Removes complex file path dependencies
- Focuses on core algorithmic concepts
- Provides comprehensive visualization and analysis tools

## References

Based on the concepts from:
"Enhancing deep learning-based field reconstruction with differentiable learning framework" by Xu Liu et al.

## License

This simplified implementation is provided for educational purposes.