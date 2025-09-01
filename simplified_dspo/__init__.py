"""
Simplified DSPO (Differentiable Sensor Placement Optimization) Implementation

This package provides a simplified, educational implementation of DSPO using synthetic data.
It demonstrates the core bi-level optimization concept for sensor placement and field reconstruction.

Main components:
- synthetic_data: Generate various types of synthetic flow field data
- models: Reconstruction models (MLP, CNN, ResMLP, MultiScale)
- sensor_optimization: Differentiable sensor position optimization
- dspo_solver: Main bi-level optimization framework
- utils: Utility functions for RBF interpolation, visualization, etc.
- demo: Demonstration script and examples

Usage:
    from simplified_dspo import DSPOSolver, SyntheticDataGenerator
    
    # Generate synthetic data
    generator = SyntheticDataGenerator()
    field_data, coords = generator.generate_combined_data(n_samples=100)
    
    # Create and train DSPO solver
    solver = DSPOSolver(n_sensors=8, field_shape=(64, 64))
    history = solver.train(field_data)
    
    # Visualize results
    solver.visualize_results(field_data)
"""

__version__ = "1.0.0"
__author__ = "Simplified DSPO Implementation"

# Import main classes and functions
from .synthetic_data import SyntheticDataGenerator
from .dspo_solver import DSPOSolver
from .models import create_reconstructor, MLPReconstructor, CNNReconstructor
from .sensor_optimization import SensorOptimizer, AdaptiveSensorOptimizer, DSPOLoss
from .utils import (
    get_sensor_observations_rbf,
    get_sensor_observations_torch,
    initialize_sensor_positions,
    visualize_sensors_and_field,
    compute_reconstruction_metrics,
    setup_device
)

__all__ = [
    'SyntheticDataGenerator',
    'DSPOSolver',
    'create_reconstructor',
    'MLPReconstructor',
    'CNNReconstructor',
    'SensorOptimizer',
    'AdaptiveSensorOptimizer',
    'DSPOLoss',
    'get_sensor_observations_rbf',
    'get_sensor_observations_torch',
    'initialize_sensor_positions',
    'visualize_sensors_and_field',
    'compute_reconstruction_metrics',
    'setup_device'
]