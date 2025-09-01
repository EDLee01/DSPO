"""
Sensor placement optimization for simplified DSPO implementation.
Handles differentiable sensor position optimization using PyTorch autograd.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Optional, Callable
try:
    from .utils import (apply_sensor_constraints, initialize_sensor_positions, 
                       get_sensor_observations_torch)
except ImportError:
    from utils import (apply_sensor_constraints, initialize_sensor_positions, 
                      get_sensor_observations_torch)


class SensorOptimizer:
    """
    Handles differentiable optimization of sensor positions.
    """
    
    def __init__(self, n_sensors: int, spatial_bounds: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
                 min_distance: float = 0.05, device: str = 'cpu'):
        """
        Initialize sensor optimizer.
        
        Args:
            n_sensors: Number of sensors
            spatial_bounds: Domain bounds (x_min, x_max, y_min, y_max)
            min_distance: Minimum distance between sensors
            device: Device for computation
        """
        self.n_sensors = n_sensors
        self.spatial_bounds = spatial_bounds
        self.min_distance = min_distance
        self.device = device
        
        # Initialize sensor positions as trainable parameters
        self.sensor_positions = None
        self.optimizer = None
        self.history = {'positions': [], 'gradients': [], 'losses': []}
    
    def initialize_positions(self, method: str = 'uniform') -> torch.Tensor:
        """
        Initialize sensor positions.
        
        Args:
            method: Initialization method ('uniform', 'random', 'grid')
            
        Returns:
            Initialized sensor positions
        """
        positions = initialize_sensor_positions(
            self.n_sensors, method, self.spatial_bounds, self.device
        )
        
        # Make positions trainable parameters
        self.sensor_positions = nn.Parameter(positions.clone().detach().requires_grad_(True))
        
        return self.sensor_positions
    
    def setup_optimizer(self, learning_rate: float = 0.01, optimizer_type: str = 'adam',
                       weight_decay: float = 0.0) -> optim.Optimizer:
        """
        Setup optimizer for sensor positions.
        
        Args:
            learning_rate: Learning rate for optimization
            optimizer_type: Type of optimizer ('adam', 'sgd', 'rmsprop')
            weight_decay: Weight decay for regularization
            
        Returns:
            Configured optimizer
        """
        if self.sensor_positions is None:
            raise ValueError("Sensor positions must be initialized first")
        
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam([self.sensor_positions], lr=learning_rate, 
                                      weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD([self.sensor_positions], lr=learning_rate, 
                                     momentum=0.9, weight_decay=weight_decay)
        elif optimizer_type == 'rmsprop':
            self.optimizer = optim.RMSprop([self.sensor_positions], lr=learning_rate,
                                         weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        return self.optimizer
    
    def apply_constraints(self):
        """Apply constraints to sensor positions."""
        if self.sensor_positions is not None:
            with torch.no_grad():
                self.sensor_positions.data = apply_sensor_constraints(
                    self.sensor_positions.data, self.spatial_bounds, self.min_distance
                )
    
    def compute_sensor_gradients(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Compute gradients of loss with respect to sensor positions.
        
        Args:
            loss: Loss tensor to compute gradients for
            
        Returns:
            Gradients with respect to sensor positions
        """
        if self.sensor_positions.grad is not None:
            self.sensor_positions.grad.zero_()
        
        gradients = torch.autograd.grad(loss, self.sensor_positions, 
                                      create_graph=True, retain_graph=True)[0]
        
        return gradients
    
    def step(self, loss: torch.Tensor, apply_constraints: bool = True) -> Tuple[torch.Tensor, float]:
        """
        Perform one optimization step.
        
        Args:
            loss: Loss tensor to optimize
            apply_constraints: Whether to apply constraints after step
            
        Returns:
            Tuple of (gradients, loss_value)
        """
        self.optimizer.zero_grad()
        
        # Compute gradients
        gradients = self.compute_sensor_gradients(loss)
        
        # Set gradients for optimizer
        self.sensor_positions.grad = gradients
        
        # Optimization step
        self.optimizer.step()
        
        # Apply constraints
        if apply_constraints:
            self.apply_constraints()
        
        # Store history
        loss_value = loss.item()
        self.history['positions'].append(self.sensor_positions.data.clone().cpu().numpy())
        self.history['gradients'].append(gradients.clone().detach().cpu().numpy())
        self.history['losses'].append(loss_value)
        
        return gradients, loss_value
    
    def get_positions(self) -> torch.Tensor:
        """Get current sensor positions."""
        return self.sensor_positions
    
    def get_history(self) -> dict:
        """Get optimization history."""
        return self.history
    
    def reset_history(self):
        """Reset optimization history."""
        self.history = {'positions': [], 'gradients': [], 'losses': []}


class DSPOLoss:
    """
    Loss functions for DSPO optimization.
    """
    
    @staticmethod
    def reconstruction_loss(true_field: torch.Tensor, reconstructed_field: torch.Tensor,
                          loss_type: str = 'mse') -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            true_field: Ground truth field
            reconstructed_field: Reconstructed field
            loss_type: Type of loss ('mse', 'mae', 'huber')
            
        Returns:
            Reconstruction loss
        """
        if loss_type == 'mse':
            return torch.mean((true_field - reconstructed_field) ** 2)
        elif loss_type == 'mae':
            return torch.mean(torch.abs(true_field - reconstructed_field))
        elif loss_type == 'huber':
            return torch.nn.functional.huber_loss(reconstructed_field, true_field)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    @staticmethod
    def sensor_diversity_loss(sensor_positions: torch.Tensor, 
                            min_distance: float = 0.05) -> torch.Tensor:
        """
        Compute loss to encourage sensor diversity (minimum distance).
        
        Args:
            sensor_positions: Sensor positions tensor
            min_distance: Minimum desired distance between sensors
            
        Returns:
            Diversity loss
        """
        n_sensors = sensor_positions.shape[0]
        loss = 0.0
        
        for i in range(n_sensors):
            for j in range(i + 1, n_sensors):
                distance = torch.norm(sensor_positions[i] - sensor_positions[j])
                if distance < min_distance:
                    loss += (min_distance - distance) ** 2
        
        return loss
    
    @staticmethod
    def information_content_loss(sensor_observations: torch.Tensor) -> torch.Tensor:
        """
        Compute loss based on information content of sensor observations.
        
        Args:
            sensor_observations: Sensor observations tensor
            
        Returns:
            Information content loss (negative variance to maximize information)
        """
        # Encourage high variance in sensor observations (more information)
        return -torch.var(sensor_observations)
    
    @staticmethod
    def coverage_loss(sensor_positions: torch.Tensor, 
                     spatial_bounds: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)) -> torch.Tensor:
        """
        Compute loss to encourage spatial coverage.
        
        Args:
            sensor_positions: Sensor positions tensor
            spatial_bounds: Domain bounds
            
        Returns:
            Coverage loss
        """
        x_min, x_max, y_min, y_max = spatial_bounds
        
        # Encourage sensors to spread across the domain
        x_spread = torch.max(sensor_positions[:, 0]) - torch.min(sensor_positions[:, 0])
        y_spread = torch.max(sensor_positions[:, 1]) - torch.min(sensor_positions[:, 1])
        
        target_x_spread = 0.8 * (x_max - x_min)
        target_y_spread = 0.8 * (y_max - y_min)
        
        coverage_loss = (target_x_spread - x_spread) ** 2 + (target_y_spread - y_spread) ** 2
        
        return coverage_loss
    
    @staticmethod
    def combined_loss(true_field: torch.Tensor, reconstructed_field: torch.Tensor,
                     sensor_positions: torch.Tensor, sensor_observations: torch.Tensor,
                     weights: dict = None) -> torch.Tensor:
        """
        Compute combined loss for DSPO optimization.
        
        Args:
            true_field: Ground truth field
            reconstructed_field: Reconstructed field
            sensor_positions: Sensor positions
            sensor_observations: Sensor observations
            weights: Dictionary of loss weights
            
        Returns:
            Combined loss
        """
        if weights is None:
            weights = {
                'reconstruction': 1.0,
                'diversity': 0.1,
                'information': 0.01,
                'coverage': 0.05
            }
        
        # Reconstruction loss (primary)
        recon_loss = DSPOLoss.reconstruction_loss(true_field, reconstructed_field)
        
        # Sensor diversity loss
        diversity_loss = DSPOLoss.sensor_diversity_loss(sensor_positions)
        
        # Information content loss
        info_loss = DSPOLoss.information_content_loss(sensor_observations)
        
        # Coverage loss
        coverage_loss = DSPOLoss.coverage_loss(sensor_positions)
        
        # Combine losses
        total_loss = (weights['reconstruction'] * recon_loss +
                     weights['diversity'] * diversity_loss +
                     weights['information'] * info_loss +
                     weights['coverage'] * coverage_loss)
        
        return total_loss, {
            'reconstruction': recon_loss.item() if hasattr(recon_loss, 'item') else float(recon_loss),
            'diversity': diversity_loss.item() if hasattr(diversity_loss, 'item') else float(diversity_loss),
            'information': info_loss.item() if hasattr(info_loss, 'item') else float(info_loss),
            'coverage': coverage_loss.item() if hasattr(coverage_loss, 'item') else float(coverage_loss),
            'total': total_loss.item() if hasattr(total_loss, 'item') else float(total_loss)
        }


class AdaptiveSensorOptimizer(SensorOptimizer):
    """
    Adaptive sensor optimizer with learning rate scheduling and advanced features.
    """
    
    def __init__(self, n_sensors: int, **kwargs):
        super().__init__(n_sensors, **kwargs)
        self.scheduler = None
        self.best_loss = float('inf')
        self.best_positions = None
        self.patience = 20
        self.patience_counter = 0
    
    def setup_scheduler(self, scheduler_type: str = 'reduce_on_plateau', **kwargs):
        """
        Setup learning rate scheduler.
        
        Args:
            scheduler_type: Type of scheduler
            **kwargs: Scheduler-specific arguments
        """
        if self.optimizer is None:
            raise ValueError("Optimizer must be setup first")
        
        if scheduler_type == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.7, patience=10, **kwargs
            )
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=100, **kwargs
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=50, gamma=0.7, **kwargs
            )
    
    def step(self, loss: torch.Tensor, apply_constraints: bool = True) -> Tuple[torch.Tensor, float]:
        """Enhanced step with adaptive features."""
        gradients, loss_value = super().step(loss, apply_constraints)
        
        # Update scheduler
        if self.scheduler is not None:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(loss_value)
            else:
                self.scheduler.step()
        
        # Early stopping check
        if loss_value < self.best_loss:
            self.best_loss = loss_value
            self.best_positions = self.sensor_positions.data.clone()
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return gradients, loss_value
    
    def should_stop_early(self) -> bool:
        """Check if early stopping criteria is met."""
        return self.patience_counter >= self.patience
    
    def restore_best_positions(self):
        """Restore best sensor positions found during optimization."""
        if self.best_positions is not None:
            with torch.no_grad():
                self.sensor_positions.data = self.best_positions.clone()


if __name__ == "__main__":
    # Test sensor optimization
    print("Testing sensor optimization...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create sample data
    n_sensors = 6
    spatial_coords = torch.rand(100, 2, device=device)
    field_data = torch.rand(10, 100, device=device)
    
    # Initialize sensor optimizer
    sensor_opt = SensorOptimizer(n_sensors, device=device)
    sensor_opt.initialize_positions(method='uniform')
    sensor_opt.setup_optimizer(learning_rate=0.01)
    
    print(f"Initial sensor positions:\n{sensor_opt.get_positions()}")
    
    # Test optimization step
    sensor_obs = get_sensor_observations_torch(field_data, spatial_coords, 
                                             sensor_opt.get_positions())
    
    # Create dummy loss
    dummy_loss = torch.mean(sensor_obs ** 2)
    
    gradients, loss_value = sensor_opt.step(dummy_loss)
    print(f"After one step - Loss: {loss_value:.6f}")
    print(f"Updated sensor positions:\n{sensor_opt.get_positions()}")
    
    # Test loss functions
    print("\nTesting loss functions...")
    true_field = torch.rand(5, 64, device=device)
    reconstructed_field = torch.rand(5, 64, device=device)
    
    recon_loss = DSPOLoss.reconstruction_loss(true_field, reconstructed_field)
    print(f"Reconstruction loss: {recon_loss.item():.6f}")
    
    diversity_loss = DSPOLoss.sensor_diversity_loss(sensor_opt.get_positions())
    print(f"Diversity loss: {diversity_loss.item():.6f}")
    
    print("Sensor optimization testing completed!")