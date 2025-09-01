"""
Utility functions for simplified DSPO implementation.
Includes RBF interpolation, visualization, and helper functions.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from typing import Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


def get_sensor_observations_rbf(field_data: np.ndarray, spatial_coords: np.ndarray, 
                               sensor_positions: np.ndarray, rbf_function: str = 'thin_plate_spline') -> np.ndarray:
    """
    Extract sensor observations from field data using RBF interpolation.
    
    Args:
        field_data: Field data array of shape (n_samples, n_spatial_points)
        spatial_coords: Spatial coordinates of shape (n_spatial_points, 2)
        sensor_positions: Sensor positions of shape (n_sensors, 2)
        rbf_function: RBF function type
        
    Returns:
        Sensor observations of shape (n_samples, n_sensors)
    """
    n_samples = field_data.shape[0]
    n_sensors = sensor_positions.shape[0]
    observations = np.zeros((n_samples, n_sensors))
    
    for i in range(n_samples):
        try:
            # Create RBF interpolator for this time step
            rbf = RBFInterpolator(spatial_coords, field_data[i], 
                                kernel=rbf_function, smoothing=1e-6)
            
            # Interpolate at sensor positions
            observations[i] = rbf(sensor_positions)
        except Exception as e:
            # Fallback to nearest neighbor if RBF fails
            observations[i] = get_sensor_observations_nearest(
                field_data[i:i+1], spatial_coords, sensor_positions)[0]
    
    return observations


def get_sensor_observations_nearest(field_data: np.ndarray, spatial_coords: np.ndarray, 
                                   sensor_positions: np.ndarray) -> np.ndarray:
    """
    Extract sensor observations using nearest neighbor interpolation.
    
    Args:
        field_data: Field data array of shape (n_samples, n_spatial_points)
        spatial_coords: Spatial coordinates of shape (n_spatial_points, 2)
        sensor_positions: Sensor positions of shape (n_sensors, 2)
        
    Returns:
        Sensor observations of shape (n_samples, n_sensors)
    """
    n_samples = field_data.shape[0]
    n_sensors = sensor_positions.shape[0]
    observations = np.zeros((n_samples, n_sensors))
    
    for i in range(n_sensors):
        # Find nearest spatial point for each sensor
        distances = np.linalg.norm(spatial_coords - sensor_positions[i], axis=1)
        nearest_idx = np.argmin(distances)
        observations[:, i] = field_data[:, nearest_idx]
    
    return observations


class TorchRBFInterpolator:
    """PyTorch-compatible RBF interpolator for gradient computation."""
    
    def __init__(self, spatial_coords: torch.Tensor, field_values: torch.Tensor, 
                 sigma: float = 0.1, device: str = 'cpu'):
        """
        Initialize PyTorch RBF interpolator.
        
        Args:
            spatial_coords: Spatial coordinates tensor of shape (n_points, 2)
            field_values: Field values tensor of shape (n_points,)
            sigma: RBF kernel width
            device: Device for computation
        """
        self.spatial_coords = spatial_coords.to(device)
        self.field_values = field_values.to(device)
        self.sigma = sigma
        self.device = device
        
        # Precompute RBF matrix and solve for weights
        self._compute_weights()
    
    def _rbf_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Gaussian RBF kernel."""
        dist_sq = torch.cdist(x1, x2, p=2) ** 2
        return torch.exp(-dist_sq / (2 * self.sigma ** 2))
    
    def _compute_weights(self):
        """Precompute RBF weights."""
        K = self._rbf_kernel(self.spatial_coords, self.spatial_coords)
        # Add small regularization for numerical stability
        K += 1e-6 * torch.eye(K.shape[0], device=self.device)
        self.weights = torch.linalg.solve(K, self.field_values)
    
    def __call__(self, query_points: torch.Tensor) -> torch.Tensor:
        """
        Interpolate at query points.
        
        Args:
            query_points: Query points tensor of shape (n_query, 2)
            
        Returns:
            Interpolated values tensor of shape (n_query,)
        """
        K_query = self._rbf_kernel(query_points, self.spatial_coords)
        return K_query @ self.weights


def get_sensor_observations_torch(field_data: torch.Tensor, spatial_coords: torch.Tensor, 
                                 sensor_positions: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """
    Extract sensor observations using PyTorch RBF interpolation for gradient computation.
    
    Args:
        field_data: Field data tensor of shape (n_samples, n_spatial_points)
        spatial_coords: Spatial coordinates tensor of shape (n_spatial_points, 2)
        sensor_positions: Sensor positions tensor of shape (n_sensors, 2)
        sigma: RBF kernel width
        
    Returns:
        Sensor observations tensor of shape (n_samples, n_sensors)
    """
    device = field_data.device
    n_samples = field_data.shape[0]
    n_sensors = sensor_positions.shape[0]
    observations = torch.zeros(n_samples, n_sensors, device=device)
    
    for i in range(n_samples):
        # Create RBF interpolator for this time step
        rbf = TorchRBFInterpolator(spatial_coords, field_data[i], sigma, device)
        observations[i] = rbf(sensor_positions)
    
    return observations


def apply_sensor_constraints(sensor_positions: torch.Tensor, 
                           bounds: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
                           min_distance: float = 0.05) -> torch.Tensor:
    """
    Apply constraints to sensor positions.
    
    Args:
        sensor_positions: Sensor positions tensor of shape (n_sensors, 2)
        bounds: Domain bounds (x_min, x_max, y_min, y_max)
        min_distance: Minimum distance between sensors
        
    Returns:
        Constrained sensor positions
    """
    x_min, x_max, y_min, y_max = bounds
    
    # Apply boundary constraints
    sensor_positions[:, 0] = torch.clamp(sensor_positions[:, 0], x_min, x_max)
    sensor_positions[:, 1] = torch.clamp(sensor_positions[:, 1], y_min, y_max)
    
    # Apply minimum distance constraints (simplified)
    n_sensors = sensor_positions.shape[0]
    for i in range(n_sensors):
        for j in range(i + 1, n_sensors):
            dist = torch.norm(sensor_positions[i] - sensor_positions[j])
            if dist < min_distance:
                # Move sensors apart
                direction = (sensor_positions[j] - sensor_positions[i]) / (dist + 1e-8)
                sensor_positions[j] = sensor_positions[i] + direction * min_distance
                # Re-apply boundary constraints
                sensor_positions[j, 0] = torch.clamp(sensor_positions[j, 0], x_min, x_max)
                sensor_positions[j, 1] = torch.clamp(sensor_positions[j, 1], y_min, y_max)
    
    return sensor_positions


def initialize_sensor_positions(n_sensors: int, method: str = 'uniform', 
                               bounds: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
                               device: str = 'cpu') -> torch.Tensor:
    """
    Initialize sensor positions using different methods.
    
    Args:
        n_sensors: Number of sensors
        method: Initialization method ('uniform', 'random', 'grid')
        bounds: Domain bounds (x_min, x_max, y_min, y_max)
        device: Device for computation
        
    Returns:
        Initial sensor positions tensor of shape (n_sensors, 2)
    """
    x_min, x_max, y_min, y_max = bounds
    
    if method == 'uniform':
        # Uniform distribution
        x = torch.rand(n_sensors, device=device) * (x_max - x_min) + x_min
        y = torch.rand(n_sensors, device=device) * (y_max - y_min) + y_min
        positions = torch.stack([x, y], dim=1)
        
    elif method == 'grid':
        # Grid-based initialization
        n_x = int(np.ceil(np.sqrt(n_sensors)))
        n_y = int(np.ceil(n_sensors / n_x))
        
        x_coords = torch.linspace(x_min + 0.1, x_max - 0.1, n_x, device=device)
        y_coords = torch.linspace(y_min + 0.1, y_max - 0.1, n_y, device=device)
        
        positions = []
        for i in range(n_x):
            for j in range(n_y):
                if len(positions) < n_sensors:
                    positions.append([x_coords[i].item(), y_coords[j].item()])
        
        positions = torch.tensor(positions[:n_sensors], device=device, dtype=torch.float32)
        
    elif method == 'random':
        # Random initialization with constraints
        positions = torch.zeros(n_sensors, 2, device=device)
        for i in range(n_sensors):
            # Try to place sensor with minimum distance constraint
            for _ in range(100):  # Max attempts
                x = torch.rand(1, device=device) * (x_max - x_min) + x_min
                y = torch.rand(1, device=device) * (y_max - y_min) + y_min
                new_pos = torch.tensor([x.item(), y.item()], device=device)
                
                # Check distance from existing sensors
                if i == 0:
                    positions[i] = new_pos
                    break
                else:
                    distances = torch.norm(positions[:i] - new_pos, dim=1)
                    if torch.all(distances > 0.05):  # Minimum distance
                        positions[i] = new_pos
                        break
            else:
                # Fallback to grid position if random placement fails
                positions[i] = torch.tensor([
                    x_min + (i % 4) * (x_max - x_min) / 4 + 0.1,
                    y_min + (i // 4) * (y_max - y_min) / 4 + 0.1
                ], device=device)
    
    return positions


def visualize_sensors_and_field(field_2d: np.ndarray, sensor_positions: np.ndarray, 
                               reconstructed_field: Optional[np.ndarray] = None,
                               title: str = "Sensor Positions and Field", 
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize field data with sensor positions.
    
    Args:
        field_2d: 2D field data
        sensor_positions: Sensor positions
        reconstructed_field: Optional reconstructed field for comparison
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    if reconstructed_field is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes = [axes[0], axes[1], None]
    
    # Original field
    im1 = axes[0].imshow(field_2d, extent=[0, 1, 0, 1], origin='lower', cmap='RdBu_r')
    axes[0].scatter(sensor_positions[:, 0], sensor_positions[:, 1], 
                   c='black', s=50, marker='x', linewidths=2)
    axes[0].set_title('Original Field with Sensors')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])
    
    # Sensor positions only
    axes[1].scatter(sensor_positions[:, 0], sensor_positions[:, 1], 
                   c='red', s=100, marker='o', alpha=0.7)
    for i, pos in enumerate(sensor_positions):
        axes[1].annotate(f'S{i}', (pos[0], pos[1]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_title('Sensor Positions')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].grid(True, alpha=0.3)
    
    # Reconstructed field (if provided)
    if reconstructed_field is not None and axes[2] is not None:
        im3 = axes[2].imshow(reconstructed_field, extent=[0, 1, 0, 1], 
                           origin='lower', cmap='RdBu_r')
        axes[2].scatter(sensor_positions[:, 0], sensor_positions[:, 1], 
                       c='black', s=50, marker='x', linewidths=2)
        axes[2].set_title('Reconstructed Field')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        plt.colorbar(im3, ax=axes[2])
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_optimization_history(losses: list, sensor_positions_history: list, 
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot optimization history including loss and sensor movement.
    
    Args:
        losses: List of loss values during optimization
        sensor_positions_history: List of sensor positions at each iteration
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss curve
    axes[0].plot(losses)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Optimization Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Sensor trajectory
    if len(sensor_positions_history) > 1:
        positions_array = np.array(sensor_positions_history)
        n_sensors = positions_array.shape[1]
        
        # Plot initial and final positions
        initial_pos = positions_array[0]
        final_pos = positions_array[-1]
        
        axes[1].scatter(initial_pos[:, 0], initial_pos[:, 1], 
                       c='blue', s=100, marker='o', alpha=0.7, label='Initial')
        axes[1].scatter(final_pos[:, 0], final_pos[:, 1], 
                       c='red', s=100, marker='s', alpha=0.7, label='Final')
        
        # Plot trajectories
        for i in range(n_sensors):
            trajectory = positions_array[:, i, :]
            axes[1].plot(trajectory[:, 0], trajectory[:, 1], 
                        alpha=0.5, linewidth=1)
    
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_title('Sensor Position Evolution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def compute_reconstruction_metrics(true_field: np.ndarray, reconstructed_field: np.ndarray) -> dict:
    """
    Compute reconstruction quality metrics.
    
    Args:
        true_field: Ground truth field
        reconstructed_field: Reconstructed field
        
    Returns:
        Dictionary of metrics
    """
    # Mean squared error
    mse = np.mean((true_field - reconstructed_field) ** 2)
    
    # Root mean squared error
    rmse = np.sqrt(mse)
    
    # Mean absolute error
    mae = np.mean(np.abs(true_field - reconstructed_field))
    
    # Normalized root mean squared error
    nrmse = rmse / (np.max(true_field) - np.min(true_field))
    
    # Correlation coefficient
    corr = np.corrcoef(true_field.ravel(), reconstructed_field.ravel())[0, 1]
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'nrmse': nrmse,
        'correlation': corr
    }


def setup_device() -> str:
    """Setup computation device (CPU/GPU)."""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU")
    return device


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Create sample data
    spatial_coords = np.random.rand(100, 2)
    field_data = np.random.rand(10, 100)
    sensor_positions = np.random.rand(5, 2)
    
    # Test RBF interpolation
    print("Testing RBF interpolation...")
    observations = get_sensor_observations_rbf(field_data, spatial_coords, sensor_positions)
    print(f"Sensor observations shape: {observations.shape}")
    
    # Test sensor initialization
    print("Testing sensor initialization...")
    device = setup_device()
    sensors = initialize_sensor_positions(8, method='grid', device=device)
    print(f"Initialized sensors shape: {sensors.shape}")
    
    print("Utility functions test completed.")