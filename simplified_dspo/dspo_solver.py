"""
Main DSPO solver implementing bi-level optimization framework.
Combines sensor placement optimization with field reconstruction.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional, Union
import time
import os

try:
    from .models import create_reconstructor
    from .sensor_optimization import SensorOptimizer, AdaptiveSensorOptimizer, DSPOLoss
    from .utils import (get_sensor_observations_torch, visualize_sensors_and_field, 
                       plot_optimization_history, compute_reconstruction_metrics, setup_device)
except ImportError:
    from models import create_reconstructor
    from sensor_optimization import SensorOptimizer, AdaptiveSensorOptimizer, DSPOLoss
    from utils import (get_sensor_observations_torch, visualize_sensors_and_field, 
                      plot_optimization_history, compute_reconstruction_metrics, setup_device)


class DSPOSolver:
    """
    Main DSPO solver implementing bi-level optimization.
    
    Lower level: Train reconstruction model with fixed sensor positions
    Upper level: Optimize sensor positions with fixed reconstruction model
    """
    
    def __init__(self, n_sensors: int, field_shape: Tuple[int, int],
                 model_type: str = 'mlp', spatial_bounds: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
                 device: str = None, **model_kwargs):
        """
        Initialize DSPO solver.
        
        Args:
            n_sensors: Number of sensors
            field_shape: Shape of the field (height, width)
            model_type: Type of reconstruction model ('mlp', 'cnn', 'resmlp', 'multiscale')
            spatial_bounds: Domain bounds (x_min, x_max, y_min, y_max)
            device: Device for computation
            **model_kwargs: Additional arguments for reconstruction model
        """
        self.n_sensors = n_sensors
        self.field_shape = field_shape
        self.spatial_bounds = spatial_bounds
        self.device = device if device else setup_device()
        
        # Create spatial coordinate grid
        height, width = field_shape
        x = np.linspace(spatial_bounds[0], spatial_bounds[1], width)
        y = np.linspace(spatial_bounds[2], spatial_bounds[3], height)
        X, Y = np.meshgrid(x, y)
        self.spatial_coords = torch.tensor(
            np.column_stack([X.ravel(), Y.ravel()]), dtype=torch.float32, device=self.device
        )
        
        # Initialize reconstruction model
        self.reconstruction_model = create_reconstructor(
            model_type, n_sensors, field_shape, **model_kwargs
        ).to(self.device)
        
        # Initialize sensor optimizer
        self.sensor_optimizer = AdaptiveSensorOptimizer(
            n_sensors, spatial_bounds=spatial_bounds, device=self.device
        )
        
        # Training components
        self.model_optimizer = None
        self.model_scheduler = None
        
        # Training history
        self.history = {
            'model_losses': [],
            'sensor_losses': [],
            'reconstruction_metrics': [],
            'sensor_positions': [],
            'best_model_loss': float('inf'),
            'best_sensor_loss': float('inf')
        }
        
        # Best models
        self.best_model_state = None
        self.best_sensor_positions = None
        
        print(f"DSPO Solver initialized:")
        print(f"  Sensors: {n_sensors}")
        print(f"  Field shape: {field_shape}")
        print(f"  Model type: {model_type}")
        print(f"  Device: {self.device}")
    
    def setup_model_optimizer(self, learning_rate: float = 0.001, 
                             optimizer_type: str = 'adam', weight_decay: float = 1e-5):
        """Setup optimizer for reconstruction model."""
        if optimizer_type == 'adam':
            self.model_optimizer = optim.Adam(
                self.reconstruction_model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            self.model_optimizer = optim.SGD(
                self.reconstruction_model.parameters(), lr=learning_rate, 
                momentum=0.9, weight_decay=weight_decay
            )
        elif optimizer_type == 'adamw':
            self.model_optimizer = optim.AdamW(
                self.reconstruction_model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        
        # Setup scheduler
        self.model_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.model_optimizer, mode='min', factor=0.7, patience=15
        )
    
    def initialize_sensors(self, method: str = 'uniform'):
        """Initialize sensor positions."""
        self.sensor_optimizer.initialize_positions(method)
        self.sensor_optimizer.setup_optimizer(learning_rate=0.01)
        self.sensor_optimizer.setup_scheduler('reduce_on_plateau')
    
    def get_sensor_observations(self, field_data: torch.Tensor) -> torch.Tensor:
        """Get sensor observations from field data."""
        sensor_positions = self.sensor_optimizer.get_positions()
        return get_sensor_observations_torch(
            field_data, self.spatial_coords, sensor_positions, sigma=0.05
        )
    
    def train_reconstruction_model(self, field_data: torch.Tensor, n_epochs: int = 100,
                                 batch_size: int = 32, validation_split: float = 0.2,
                                 verbose: bool = False) -> Dict[str, float]:
        """
        Train reconstruction model with fixed sensor positions (Lower level).
        
        Args:
            field_data: Field data tensor of shape (n_samples, height*width)
            n_epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            verbose: Whether to print training progress
            
        Returns:
            Dictionary of training metrics
        """
        self.reconstruction_model.train()
        
        # Split data
        n_samples = field_data.shape[0]
        n_val = int(n_samples * validation_split)
        indices = torch.randperm(n_samples)
        
        train_data = field_data[indices[n_val:]]
        val_data = field_data[indices[:n_val]]
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # Training
            epoch_train_loss = 0.0
            n_batches = 0
            
            for i in range(0, len(train_data), batch_size):
                batch_data = train_data[i:i+batch_size]
                
                # Get sensor observations
                sensor_obs = self.get_sensor_observations(batch_data)
                
                # Forward pass
                self.model_optimizer.zero_grad()
                reconstructed = self.reconstruction_model(sensor_obs)
                
                # Reshape if needed
                if len(reconstructed.shape) == 2:
                    reconstructed = reconstructed.view(-1, *self.field_shape)
                    batch_data = batch_data.view(-1, *self.field_shape)
                
                # Compute loss
                loss = nn.functional.mse_loss(reconstructed, batch_data)
                
                # Backward pass
                loss.backward()
                self.model_optimizer.step()
                
                epoch_train_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = epoch_train_loss / n_batches
            train_losses.append(avg_train_loss)
            
            # Validation
            self.reconstruction_model.eval()
            with torch.no_grad():
                val_sensor_obs = self.get_sensor_observations(val_data)
                val_reconstructed = self.reconstruction_model(val_sensor_obs)
                
                if len(val_reconstructed.shape) == 2:
                    val_reconstructed = val_reconstructed.view(-1, *self.field_shape)
                    val_data_reshaped = val_data.view(-1, *self.field_shape)
                else:
                    val_data_reshaped = val_data
                
                val_loss = nn.functional.mse_loss(val_reconstructed, val_data_reshaped).item()
                val_losses.append(val_loss)
            
            self.reconstruction_model.train()
            
            # Learning rate scheduling
            self.model_scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.best_model_state = self.reconstruction_model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Restore best model
        if self.best_model_state is not None:
            self.reconstruction_model.load_state_dict(self.best_model_state)
        
        return {
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'best_val_loss': best_val_loss,
            'n_epochs': len(train_losses)
        }
    
    def optimize_sensor_positions(self, field_data: torch.Tensor, n_iterations: int = 100,
                                 loss_weights: Dict[str, float] = None, verbose: bool = False) -> Dict[str, float]:
        """
        Optimize sensor positions with fixed reconstruction model (Upper level).
        
        Args:
            field_data: Field data tensor
            n_iterations: Number of optimization iterations
            loss_weights: Weights for different loss components
            verbose: Whether to print optimization progress
            
        Returns:
            Dictionary of optimization metrics
        """
        self.reconstruction_model.eval()  # Fix the model
        
        if loss_weights is None:
            loss_weights = {
                'reconstruction': 1.0,
                'diversity': 0.1,
                'information': 0.01,
                'coverage': 0.05
            }
        
        sensor_losses = []
        best_loss = float('inf')
        
        for iteration in range(n_iterations):
            # Get current sensor observations
            sensor_obs = self.get_sensor_observations(field_data)
            
            # Forward pass through reconstruction model
            with torch.no_grad():
                reconstructed = self.reconstruction_model(sensor_obs)
            
            # Reshape if needed
            if len(reconstructed.shape) == 2:
                reconstructed = reconstructed.view(-1, *self.field_shape)
                field_data_reshaped = field_data.view(-1, *self.field_shape)
            else:
                field_data_reshaped = field_data
            
            # Compute combined loss
            total_loss, loss_components = DSPOLoss.combined_loss(
                field_data_reshaped, reconstructed,
                self.sensor_optimizer.get_positions(),
                sensor_obs, loss_weights
            )
            
            # Optimization step
            gradients, loss_value = self.sensor_optimizer.step(total_loss)
            sensor_losses.append(loss_value)
            
            if loss_value < best_loss:
                best_loss = loss_value
                self.best_sensor_positions = self.sensor_optimizer.get_positions().clone()
            
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: Loss = {loss_value:.6f}")
                for key, value in loss_components.items():
                    print(f"  {key}: {value:.6f}")
            
            # Early stopping check
            if self.sensor_optimizer.should_stop_early():
                if verbose:
                    print(f"Early stopping at iteration {iteration}")
                break
        
        return {
            'final_loss': sensor_losses[-1],
            'best_loss': best_loss,
            'n_iterations': len(sensor_losses)
        }
    
    def train(self, field_data: torch.Tensor, n_outer_iterations: int = 5,
              model_epochs_per_iteration: int = 50, sensor_iterations_per_iteration: int = 100,
              batch_size: int = 32, validation_split: float = 0.2,
              loss_weights: Dict[str, float] = None, verbose: bool = True) -> Dict[str, List]:
        """
        Main bi-level training loop.
        
        Args:
            field_data: Field data tensor of shape (n_samples, height*width)
            n_outer_iterations: Number of bi-level iterations
            model_epochs_per_iteration: Epochs for model training per iteration
            sensor_iterations_per_iteration: Iterations for sensor optimization per iteration
            batch_size: Batch size for model training
            validation_split: Validation split for model training
            loss_weights: Loss weights for sensor optimization
            verbose: Whether to print progress
            
        Returns:
            Training history dictionary
        """
        print(f"Starting DSPO bi-level optimization...")
        print(f"Data shape: {field_data.shape}")
        print(f"Spatial coordinates shape: {self.spatial_coords.shape}")
        
        # Move data to device
        field_data = field_data.to(self.device)
        
        # Setup optimizers
        self.setup_model_optimizer()
        self.initialize_sensors()
        
        start_time = time.time()
        
        for outer_iter in range(n_outer_iterations):
            if verbose:
                print(f"\n=== Bi-level Iteration {outer_iter + 1}/{n_outer_iterations} ===")
            
            # Lower level: Train reconstruction model
            if verbose:
                print("Training reconstruction model...")
            
            model_metrics = self.train_reconstruction_model(
                field_data, 
                n_epochs=model_epochs_per_iteration,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=False
            )
            
            self.history['model_losses'].append(model_metrics['best_val_loss'])
            
            if verbose:
                print(f"Model training - Best val loss: {model_metrics['best_val_loss']:.6f}")
            
            # Upper level: Optimize sensor positions
            if verbose:
                print("Optimizing sensor positions...")
            
            sensor_metrics = self.optimize_sensor_positions(
                field_data,
                n_iterations=sensor_iterations_per_iteration,
                loss_weights=loss_weights,
                verbose=False
            )
            
            self.history['sensor_losses'].append(sensor_metrics['best_loss'])
            self.history['sensor_positions'].append(
                self.sensor_optimizer.get_positions().clone().detach().cpu().numpy()
            )
            
            if verbose:
                print(f"Sensor optimization - Best loss: {sensor_metrics['best_loss']:.6f}")
            
            # Compute reconstruction metrics
            self.reconstruction_model.eval()
            with torch.no_grad():
                sensor_obs = self.get_sensor_observations(field_data)
                reconstructed = self.reconstruction_model(sensor_obs)
                
                if len(reconstructed.shape) == 2:
                    reconstructed = reconstructed.view(-1, *self.field_shape)
                    field_for_metrics = field_data.view(-1, *self.field_shape)
                else:
                    field_for_metrics = field_data
                
                # Use a subset for metrics computation
                subset_size = min(10, len(field_for_metrics))
                metrics = compute_reconstruction_metrics(
                    field_for_metrics[:subset_size].cpu().numpy(),
                    reconstructed[:subset_size].cpu().numpy()
                )
                self.history['reconstruction_metrics'].append(metrics)
                
                if verbose:
                    print(f"Reconstruction metrics - RMSE: {metrics['rmse']:.6f}, Correlation: {metrics['correlation']:.4f}")
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\nTraining completed in {total_time:.2f} seconds")
            print(f"Final model loss: {self.history['model_losses'][-1]:.6f}")
            print(f"Final sensor loss: {self.history['sensor_losses'][-1]:.6f}")
        
        return self.history
    
    def predict(self, field_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict using current model and sensor positions.
        
        Args:
            field_data: Field data tensor
            
        Returns:
            Tuple of (sensor_observations, reconstructed_field)
        """
        self.reconstruction_model.eval()
        
        with torch.no_grad():
            sensor_obs = self.get_sensor_observations(field_data)
            reconstructed = self.reconstruction_model(sensor_obs)
            
            if len(reconstructed.shape) == 2:
                reconstructed = reconstructed.view(-1, *self.field_shape)
        
        return sensor_obs, reconstructed
    
    def visualize_results(self, field_data: torch.Tensor, sample_indices: List[int] = [0],
                         save_path: Optional[str] = None) -> List[plt.Figure]:
        """
        Visualize results including sensor positions and reconstructions.
        
        Args:
            field_data: Field data tensor
            sample_indices: Which samples to visualize
            save_path: Directory to save figures
            
        Returns:
            List of matplotlib figures
        """
        self.reconstruction_model.eval()
        figures = []
        
        # Get predictions
        sensor_obs, reconstructed = self.predict(field_data)
        sensor_positions = self.sensor_optimizer.get_positions().cpu().numpy()
        
        for i, idx in enumerate(sample_indices):
            if idx < len(field_data):
                # Original and reconstructed fields
                original_2d = field_data[idx].view(*self.field_shape).cpu().numpy()
                reconstructed_2d = reconstructed[idx].cpu().numpy()
                
                fig = visualize_sensors_and_field(
                    original_2d, sensor_positions, reconstructed_2d,
                    title=f'DSPO Results - Sample {idx}'
                )
                
                if save_path:
                    os.makedirs(save_path, exist_ok=True)
                    fig.savefig(f"{save_path}/dspo_result_sample_{idx}.png", 
                              dpi=150, bbox_inches='tight')
                
                figures.append(fig)
        
        # Plot optimization history
        if len(self.history['model_losses']) > 1:
            history_fig = plot_optimization_history(
                self.history['sensor_losses'],
                self.history['sensor_positions']
            )
            
            if save_path:
                history_fig.savefig(f"{save_path}/optimization_history.png",
                                  dpi=150, bbox_inches='tight')
            
            figures.append(history_fig)
        
        return figures
    
    def save_model(self, filepath: str):
        """Save the trained model and sensor positions."""
        torch.save({
            'model_state_dict': self.reconstruction_model.state_dict(),
            'sensor_positions': self.sensor_optimizer.get_positions(),
            'history': self.history,
            'config': {
                'n_sensors': self.n_sensors,
                'field_shape': self.field_shape,
                'spatial_bounds': self.spatial_bounds,
                'model_type': type(self.reconstruction_model).__name__
            }
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model and sensor positions."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.reconstruction_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Initialize sensor optimizer if needed
        if not hasattr(self.sensor_optimizer, 'sensor_positions') or self.sensor_optimizer.sensor_positions is None:
            self.sensor_optimizer.initialize_positions()
        
        with torch.no_grad():
            self.sensor_optimizer.sensor_positions.data = checkpoint['sensor_positions']
        
        self.history = checkpoint['history']
        
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Test DSPO solver
    print("Testing DSPO solver...")
    
    # Create sample data
    n_samples = 50
    field_shape = (32, 32)
    n_sensors = 6
    
    # Generate synthetic field data
    field_data = torch.randn(n_samples, field_shape[0] * field_shape[1])
    
    # Initialize solver
    solver = DSPOSolver(n_sensors, field_shape, model_type='mlp')
    
    # Train
    print("Starting training...")
    history = solver.train(field_data, n_outer_iterations=2, 
                          model_epochs_per_iteration=20,
                          sensor_iterations_per_iteration=30)
    
    print("Training completed!")
    print(f"Final model loss: {history['model_losses'][-1]:.6f}")
    print(f"Final sensor loss: {history['sensor_losses'][-1]:.6f}")
    
    # Test prediction
    sensor_obs, reconstructed = solver.predict(field_data[:5])
    print(f"Sensor observations shape: {sensor_obs.shape}")
    print(f"Reconstructed field shape: {reconstructed.shape}")
    
    print("DSPO solver testing completed successfully!")