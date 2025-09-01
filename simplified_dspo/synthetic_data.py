"""
Synthetic data generation for simplified DSPO implementation.
Creates 2D spatial flow field data for sensor placement optimization.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


class SyntheticDataGenerator:
    """Generate synthetic 2D flow field data for DSPO experiments."""
    
    def __init__(self, width: int = 64, height: int = 64):
        """
        Initialize the synthetic data generator.
        
        Args:
            width: Width of the spatial domain
            height: Height of the spatial domain
        """
        self.width = width
        self.height = height
        self.x = np.linspace(0, 1, width)
        self.y = np.linspace(0, 1, height)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Create coordinate arrays for sensor placement
        self.coords = np.column_stack([self.X.ravel(), self.Y.ravel()])
        
    def generate_gaussian_blobs(self, n_samples: int = 100, n_blobs: int = 3, 
                               temporal_variation: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic data with moving Gaussian blobs.
        
        Args:
            n_samples: Number of time snapshots
            n_blobs: Number of Gaussian blobs
            temporal_variation: Whether blobs move over time
            
        Returns:
            Tuple of (field_data, spatial_coordinates)
            field_data shape: (n_samples, height*width)
            spatial_coordinates shape: (height*width, 2)
        """
        data = []
        
        for t in range(n_samples):
            field = np.zeros_like(self.X)
            
            for i in range(n_blobs):
                if temporal_variation:
                    # Move blob centers over time
                    center_x = 0.2 + 0.6 * (i / n_blobs) + 0.1 * np.sin(2 * np.pi * t / n_samples)
                    center_y = 0.3 + 0.4 * np.sin(2 * np.pi * (t + i * n_samples/n_blobs) / n_samples)
                else:
                    # Static blob positions
                    center_x = 0.2 + 0.6 * (i / n_blobs)
                    center_y = 0.5
                
                # Gaussian parameters
                sigma_x = 0.05 + 0.03 * np.sin(2 * np.pi * t / n_samples)
                sigma_y = 0.08 + 0.02 * np.cos(2 * np.pi * t / n_samples)
                amplitude = 1.0 + 0.3 * np.sin(2 * np.pi * (t + i * 20) / n_samples)
                
                # Create Gaussian blob
                blob = amplitude * np.exp(-((self.X - center_x)**2 / (2 * sigma_x**2) + 
                                          (self.Y - center_y)**2 / (2 * sigma_y**2)))
                field += blob
            
            data.append(field.ravel())
        
        return np.array(data), self.coords
    
    def generate_vortex_flow(self, n_samples: int = 100, n_vortices: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic vortex flow data.
        
        Args:
            n_samples: Number of time snapshots
            n_vortices: Number of vortices
            
        Returns:
            Tuple of (field_data, spatial_coordinates)
        """
        data = []
        
        for t in range(n_samples):
            field = np.zeros_like(self.X)
            
            for i in range(n_vortices):
                # Vortex centers moving over time
                center_x = 0.3 + 0.4 * np.cos(2 * np.pi * (t + i * n_samples/n_vortices) / n_samples)
                center_y = 0.3 + 0.4 * np.sin(2 * np.pi * (t + i * n_samples/n_vortices) / n_samples)
                
                # Distance from vortex center
                dx = self.X - center_x
                dy = self.Y - center_y
                r = np.sqrt(dx**2 + dy**2)
                
                # Avoid division by zero
                r = np.maximum(r, 1e-6)
                
                # Vortex strength varying over time
                strength = 0.5 + 0.3 * np.sin(2 * np.pi * t / n_samples)
                
                # Vorticity field (simplified)
                vorticity = strength * np.exp(-r**2 / 0.05) / (r + 0.01)
                field += vorticity
            
            data.append(field.ravel())
        
        return np.array(data), self.coords
    
    def generate_wave_patterns(self, n_samples: int = 100, 
                              wave_numbers: List[Tuple[int, int]] = [(2, 1), (1, 3), (3, 2)]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic wave pattern data.
        
        Args:
            n_samples: Number of time snapshots
            wave_numbers: List of (kx, ky) wave number pairs
            
        Returns:
            Tuple of (field_data, spatial_coordinates)
        """
        data = []
        
        for t in range(n_samples):
            field = np.zeros_like(self.X)
            
            for kx, ky in wave_numbers:
                # Time-varying amplitude and phase
                amplitude = 0.3 + 0.2 * np.sin(2 * np.pi * t / n_samples)
                phase = 2 * np.pi * t / n_samples
                
                # Wave pattern
                wave = amplitude * np.sin(2 * np.pi * kx * self.X + 
                                        2 * np.pi * ky * self.Y + phase)
                field += wave
            
            # Add some noise
            field += 0.05 * np.random.randn(*field.shape)
            
            data.append(field.ravel())
        
        return np.array(data), self.coords
    
    def generate_combined_data(self, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate combined synthetic data with multiple patterns.
        
        Args:
            n_samples: Number of time snapshots
            
        Returns:
            Tuple of (field_data, spatial_coordinates)
        """
        # Generate individual patterns
        gaussian_data, coords = self.generate_gaussian_blobs(n_samples, n_blobs=2)
        vortex_data, _ = self.generate_vortex_flow(n_samples, n_vortices=1)
        wave_data, _ = self.generate_wave_patterns(n_samples)
        
        # Combine with different weights
        combined_data = 0.5 * gaussian_data + 0.3 * vortex_data + 0.2 * wave_data
        
        return combined_data, coords
    
    def visualize_data(self, field_data: np.ndarray, coords: np.ndarray, 
                      sample_indices: List[int] = [0, 25, 50, 75], 
                      title: str = "Synthetic Flow Field Data"):
        """
        Visualize sample snapshots of the generated data.
        
        Args:
            field_data: Generated field data
            coords: Spatial coordinates
            sample_indices: Which time samples to visualize
            title: Plot title
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i, idx in enumerate(sample_indices):
            if idx < len(field_data):
                field_2d = field_data[idx].reshape(self.height, self.width)
                im = axes[i].imshow(field_2d, extent=[0, 1, 0, 1], 
                                  origin='lower', cmap='RdBu_r')
                axes[i].set_title(f'Time step {idx}')
                axes[i].set_xlabel('x')
                axes[i].set_ylabel('y')
                plt.colorbar(im, ax=axes[i])
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig


def create_sample_datasets(save_plots: bool = True):
    """Create sample datasets for demonstration."""
    generator = SyntheticDataGenerator(width=64, height=64)
    
    datasets = {}
    
    # Generate different types of data
    print("Generating Gaussian blob data...")
    datasets['gaussian'] = generator.generate_gaussian_blobs(n_samples=100)
    
    print("Generating vortex flow data...")
    datasets['vortex'] = generator.generate_vortex_flow(n_samples=100)
    
    print("Generating wave pattern data...")
    datasets['waves'] = generator.generate_wave_patterns(n_samples=100)
    
    print("Generating combined data...")
    datasets['combined'] = generator.generate_combined_data(n_samples=100)
    
    if save_plots:
        # Visualize sample data
        for name, (data, coords) in datasets.items():
            fig = generator.visualize_data(data, coords, title=f'{name.title()} Flow Field')
            plt.savefig(f'sample_{name}_data.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    return datasets


if __name__ == "__main__":
    # Create and visualize sample datasets
    datasets = create_sample_datasets()
    
    print("Sample datasets created:")
    for name, (data, coords) in datasets.items():
        print(f"  {name}: {data.shape} field data, {coords.shape} coordinates")