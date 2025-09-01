"""
Reconstruction models for simplified DSPO implementation.
Includes MLP and CNN models for field reconstruction from sensor observations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class MLPReconstructor(nn.Module):
    """
    Multi-Layer Perceptron for field reconstruction from sensor observations.
    """
    
    def __init__(self, n_sensors: int, n_output: int, 
                 hidden_layers: list = [128, 256, 512, 256, 128], 
                 activation: str = 'gelu', dropout: float = 0.0):
        """
        Initialize MLP reconstructor.
        
        Args:
            n_sensors: Number of sensor inputs
            n_output: Number of output field points
            hidden_layers: List of hidden layer sizes
            activation: Activation function ('relu', 'gelu', 'tanh')
            dropout: Dropout probability
        """
        super(MLPReconstructor, self).__init__()
        
        self.n_sensors = n_sensors
        self.n_output = n_output
        
        # Select activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network layers
        layers = []
        in_features = n_sensors
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_size
        
        # Output layer
        layers.append(nn.Linear(in_features, n_output))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        # Count parameters
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"MLP Reconstructor initialized with {self.n_params:,} parameters")
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, sensor_observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            sensor_observations: Sensor observations of shape (batch_size, n_sensors)
            
        Returns:
            Reconstructed field of shape (batch_size, n_output)
        """
        return self.network(sensor_observations)


class CNNReconstructor(nn.Module):
    """
    Convolutional Neural Network for field reconstruction from sensor observations.
    """
    
    def __init__(self, n_sensors: int, output_shape: Tuple[int, int], 
                 fc_hidden: int = 256, channels: list = [32, 64, 128, 64, 32]):
        """
        Initialize CNN reconstructor.
        
        Args:
            n_sensors: Number of sensor inputs
            output_shape: Output field shape (height, width)
            fc_hidden: Hidden size for fully connected layers
            channels: List of channel sizes for conv layers
        """
        super(CNNReconstructor, self).__init__()
        
        self.n_sensors = n_sensors
        self.output_shape = output_shape
        self.height, self.width = output_shape
        
        # Calculate initial spatial size for conv layers
        self.init_height = max(4, self.height // 16)
        self.init_width = max(4, self.width // 16)
        
        # Fully connected layers to generate initial feature map
        self.fc_layers = nn.Sequential(
            nn.Linear(n_sensors, fc_hidden),
            nn.GELU(),
            nn.Linear(fc_hidden, fc_hidden * 2),
            nn.GELU(),
            nn.Linear(fc_hidden * 2, channels[0] * self.init_height * self.init_width)
        )
        
        # Convolutional decoder layers
        conv_layers = []
        for i in range(len(channels) - 1):
            conv_layers.append(
                nn.ConvTranspose2d(channels[i], channels[i+1], 
                                 kernel_size=4, stride=2, padding=1)
            )
            conv_layers.append(nn.GELU())
            conv_layers.append(nn.BatchNorm2d(channels[i+1]))
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Final output layer
        self.output_layer = nn.Conv2d(channels[-1], 1, kernel_size=3, padding=1)
        
        # Count parameters
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"CNN Reconstructor initialized with {self.n_params:,} parameters")
    
    def forward(self, sensor_observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            sensor_observations: Sensor observations of shape (batch_size, n_sensors)
            
        Returns:
            Reconstructed field of shape (batch_size, height, width)
        """
        batch_size = sensor_observations.shape[0]
        
        # Generate initial feature map
        x = self.fc_layers(sensor_observations)
        x = x.view(batch_size, -1, self.init_height, self.init_width)
        
        # Apply convolutional layers
        x = self.conv_layers(x)
        
        # Resize to target output shape
        x = F.interpolate(x, size=self.output_shape, mode='bilinear', align_corners=False)
        
        # Final output
        x = self.output_layer(x)
        
        return x.squeeze(1)  # Remove channel dimension


class ResidualBlock(nn.Module):
    """Residual block for improved reconstruction."""
    
    def __init__(self, in_features: int, out_features: int):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm(out_features)
        self.norm2 = nn.LayerNorm(out_features)
        
        # Skip connection
        if in_features != out_features:
            self.skip = nn.Linear(in_features, out_features)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.activation(out)
        
        out = self.linear2(out)
        out = self.norm2(out)
        
        out += identity
        out = self.activation(out)
        
        return out


class ResMLPReconstructor(nn.Module):
    """
    Residual MLP for field reconstruction with improved performance.
    """
    
    def __init__(self, n_sensors: int, n_output: int, 
                 hidden_size: int = 256, n_blocks: int = 4):
        """
        Initialize Residual MLP reconstructor.
        
        Args:
            n_sensors: Number of sensor inputs
            n_output: Number of output field points
            hidden_size: Hidden layer size
            n_blocks: Number of residual blocks
        """
        super(ResMLPReconstructor, self).__init__()
        
        self.n_sensors = n_sensors
        self.n_output = n_output
        
        # Input projection
        self.input_proj = nn.Linear(n_sensors, hidden_size)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_size, hidden_size) for _ in range(n_blocks)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, n_output)
        )
        
        # Count parameters
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Residual MLP Reconstructor initialized with {self.n_params:,} parameters")
    
    def forward(self, sensor_observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            sensor_observations: Sensor observations of shape (batch_size, n_sensors)
            
        Returns:
            Reconstructed field of shape (batch_size, n_output)
        """
        x = self.input_proj(sensor_observations)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.output_proj(x)
        
        return x


class MultiScaleMLPReconstructor(nn.Module):
    """
    Multi-scale MLP that processes sensor data at different scales.
    """
    
    def __init__(self, n_sensors: int, n_output: int, 
                 scales: list = [64, 128, 256], merge_hidden: int = 512):
        """
        Initialize multi-scale MLP reconstructor.
        
        Args:
            n_sensors: Number of sensor inputs
            n_output: Number of output field points
            scales: List of hidden sizes for different scales
            merge_hidden: Hidden size for merging scales
        """
        super(MultiScaleMLPReconstructor, self).__init__()
        
        self.n_sensors = n_sensors
        self.n_output = n_output
        
        # Create parallel branches for different scales
        self.scale_branches = nn.ModuleList()
        for scale in scales:
            branch = nn.Sequential(
                nn.Linear(n_sensors, scale),
                nn.GELU(),
                nn.Linear(scale, scale),
                nn.GELU(),
                nn.Linear(scale, scale // 2)
            )
            self.scale_branches.append(branch)
        
        # Merge features from all scales
        total_features = sum(scale // 2 for scale in scales)
        self.merge_layers = nn.Sequential(
            nn.Linear(total_features, merge_hidden),
            nn.GELU(),
            nn.Linear(merge_hidden, merge_hidden),
            nn.GELU(),
            nn.Linear(merge_hidden, n_output)
        )
        
        # Count parameters
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Multi-Scale MLP Reconstructor initialized with {self.n_params:,} parameters")
    
    def forward(self, sensor_observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            sensor_observations: Sensor observations of shape (batch_size, n_sensors)
            
        Returns:
            Reconstructed field of shape (batch_size, n_output)
        """
        # Process through each scale branch
        scale_outputs = []
        for branch in self.scale_branches:
            scale_output = branch(sensor_observations)
            scale_outputs.append(scale_output)
        
        # Concatenate outputs from all scales
        merged_features = torch.cat(scale_outputs, dim=1)
        
        # Final processing
        output = self.merge_layers(merged_features)
        
        return output


def create_reconstructor(model_type: str, n_sensors: int, output_shape: Tuple[int, int], 
                        **kwargs) -> nn.Module:
    """
    Factory function to create reconstruction models.
    
    Args:
        model_type: Type of model ('mlp', 'cnn', 'resmlp', 'multiscale')
        n_sensors: Number of sensors
        output_shape: Output field shape (height, width)
        **kwargs: Additional model-specific arguments
        
    Returns:
        Reconstruction model
    """
    n_output = output_shape[0] * output_shape[1]
    
    if model_type == 'mlp':
        return MLPReconstructor(n_sensors, n_output, **kwargs)
    elif model_type == 'cnn':
        return CNNReconstructor(n_sensors, output_shape, **kwargs)
    elif model_type == 'resmlp':
        return ResMLPReconstructor(n_sensors, n_output, **kwargs)
    elif model_type == 'multiscale':
        return MultiScaleMLPReconstructor(n_sensors, n_output, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test reconstruction models
    print("Testing reconstruction models...")
    
    n_sensors = 8
    output_shape = (64, 64)
    batch_size = 4
    
    # Create sample input
    sensor_obs = torch.randn(batch_size, n_sensors)
    
    # Test MLP
    print("\nTesting MLP Reconstructor:")
    mlp = create_reconstructor('mlp', n_sensors, output_shape)
    mlp_output = mlp(sensor_obs)
    print(f"MLP output shape: {mlp_output.shape}")
    
    # Test CNN
    print("\nTesting CNN Reconstructor:")
    cnn = create_reconstructor('cnn', n_sensors, output_shape)
    cnn_output = cnn(sensor_obs)
    print(f"CNN output shape: {cnn_output.shape}")
    
    # Test Residual MLP
    print("\nTesting Residual MLP Reconstructor:")
    resmlp = create_reconstructor('resmlp', n_sensors, output_shape)
    resmlp_output = resmlp(sensor_obs)
    print(f"Residual MLP output shape: {resmlp_output.shape}")
    
    # Test Multi-scale MLP
    print("\nTesting Multi-Scale MLP Reconstructor:")
    multiscale = create_reconstructor('multiscale', n_sensors, output_shape)
    multiscale_output = multiscale(sensor_obs)
    print(f"Multi-Scale MLP output shape: {multiscale_output.shape}")
    
    print("\nModel testing completed successfully!")