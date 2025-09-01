"""
Demonstration script for simplified DSPO implementation.
Shows complete workflow from synthetic data generation to optimization results.
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import argparse

# Add the simplified_dspo module to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from synthetic_data import SyntheticDataGenerator
    from dspo_solver import DSPOSolver
    from utils import setup_device, compute_reconstruction_metrics
except ImportError:
    from .synthetic_data import SyntheticDataGenerator
    from .dspo_solver import DSPOSolver
    from .utils import setup_device, compute_reconstruction_metrics


def run_dspo_experiment(data_type: str = 'combined', n_sensors: int = 8, 
                       field_shape: tuple = (64, 64), model_type: str = 'mlp',
                       n_samples: int = 100, n_outer_iterations: int = 3,
                       save_results: bool = True, results_dir: str = 'dspo_results') -> Dict:
    """
    Run a complete DSPO experiment.
    
    Args:
        data_type: Type of synthetic data ('gaussian', 'vortex', 'waves', 'combined')
        n_sensors: Number of sensors
        field_shape: Shape of the field
        model_type: Type of reconstruction model
        n_samples: Number of data samples
        n_outer_iterations: Number of bi-level iterations
        save_results: Whether to save results
        results_dir: Directory to save results
        
    Returns:
        Experiment results dictionary
    """
    print("="*60)
    print(f"DSPO Experiment: {data_type} data, {n_sensors} sensors, {model_type} model")
    print("="*60)
    
    # Setup device
    device = setup_device()
    
    # Create results directory
    if save_results:
        os.makedirs(results_dir, exist_ok=True)
        exp_dir = f"{results_dir}/{data_type}_{model_type}_{n_sensors}sensors"
        os.makedirs(exp_dir, exist_ok=True)
    else:
        exp_dir = None
    
    # Step 1: Generate synthetic data
    print("\n1. Generating synthetic data...")
    generator = SyntheticDataGenerator(width=field_shape[1], height=field_shape[0])
    
    if data_type == 'gaussian':
        field_data, spatial_coords = generator.generate_gaussian_blobs(n_samples)
    elif data_type == 'vortex':
        field_data, spatial_coords = generator.generate_vortex_flow(n_samples)
    elif data_type == 'waves':
        field_data, spatial_coords = generator.generate_wave_patterns(n_samples)
    elif data_type == 'combined':
        field_data, spatial_coords = generator.generate_combined_data(n_samples)
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    print(f"Generated data shape: {field_data.shape}")
    
    # Visualize sample data
    if save_results:
        fig = generator.visualize_data(field_data, spatial_coords, 
                                     title=f'{data_type.title()} Synthetic Data')
        fig.savefig(f"{exp_dir}/synthetic_data_samples.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # Convert to torch tensor
    field_data_tensor = torch.tensor(field_data, dtype=torch.float32, device=device)
    
    # Step 2: Initialize DSPO solver
    print("\n2. Initializing DSPO solver...")
    solver = DSPOSolver(n_sensors, field_shape, model_type=model_type, device=device)
    
    # Step 3: Train with bi-level optimization
    print("\n3. Running bi-level optimization...")
    history = solver.train(
        field_data_tensor,
        n_outer_iterations=n_outer_iterations,
        model_epochs_per_iteration=50,
        sensor_iterations_per_iteration=100,
        batch_size=16,
        verbose=True
    )
    
    # Step 4: Evaluate results
    print("\n4. Evaluating results...")
    
    # Get final predictions
    sensor_obs, reconstructed = solver.predict(field_data_tensor)
    
    # Compute metrics
    original_fields = field_data_tensor.view(-1, *field_shape).cpu().numpy()
    reconstructed_fields = reconstructed.cpu().numpy()
    
    final_metrics = compute_reconstruction_metrics(
        original_fields[:10], reconstructed_fields[:10]
    )
    
    print(f"Final reconstruction metrics:")
    for key, value in final_metrics.items():
        print(f"  {key}: {value:.6f}")
    
    # Step 5: Visualize results
    print("\n5. Creating visualizations...")
    
    if save_results:
        # Visualize DSPO results
        figures = solver.visualize_results(
            field_data_tensor, sample_indices=[0, 10, 20], save_path=exp_dir
        )
        
        # Plot training curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Model loss
        axes[0, 0].plot(history['model_losses'], 'b-o', label='Model Loss')
        axes[0, 0].set_xlabel('Bi-level Iteration')
        axes[0, 0].set_ylabel('Model Loss')
        axes[0, 0].set_title('Reconstruction Model Training')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Sensor loss
        axes[0, 1].plot(history['sensor_losses'], 'r-s', label='Sensor Loss')
        axes[0, 1].set_xlabel('Bi-level Iteration')
        axes[0, 1].set_ylabel('Sensor Loss')
        axes[0, 1].set_title('Sensor Position Optimization')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Reconstruction metrics over time
        if history['reconstruction_metrics']:
            rmse_values = [m['rmse'] for m in history['reconstruction_metrics']]
            corr_values = [m['correlation'] for m in history['reconstruction_metrics']]
            
            axes[1, 0].plot(rmse_values, 'g-^', label='RMSE')
            axes[1, 0].set_xlabel('Bi-level Iteration')
            axes[1, 0].set_ylabel('RMSE')
            axes[1, 0].set_title('Reconstruction Quality (RMSE)')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
            
            axes[1, 1].plot(corr_values, 'm-d', label='Correlation')
            axes[1, 1].set_xlabel('Bi-level Iteration')
            axes[1, 1].set_ylabel('Correlation')
            axes[1, 1].set_title('Reconstruction Quality (Correlation)')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        
        plt.suptitle(f'DSPO Training Progress - {data_type} data')
        plt.tight_layout()
        fig.savefig(f"{exp_dir}/training_progress.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Create sensor position evolution plot
        if len(history['sensor_positions']) > 1:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            
            # Plot sensor trajectories
            positions_array = np.array(history['sensor_positions'])
            
            for i in range(n_sensors):
                trajectory = positions_array[:, i, :]
                ax.plot(trajectory[:, 0], trajectory[:, 1], '-', alpha=0.7, 
                       linewidth=2, label=f'Sensor {i+1}')
                
                # Mark initial and final positions
                ax.scatter(trajectory[0, 0], trajectory[0, 1], 
                          s=100, marker='o', alpha=0.8)
                ax.scatter(trajectory[-1, 0], trajectory[-1, 1], 
                          s=100, marker='s', alpha=0.8)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Sensor Position Evolution\n(Circles: Initial, Squares: Final)')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            fig.savefig(f"{exp_dir}/sensor_evolution.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        # Save model
        solver.save_model(f"{exp_dir}/dspo_model.pth")
        
        print(f"Results saved to {exp_dir}/")
    
    # Prepare results summary
    results = {
        'data_type': data_type,
        'n_sensors': n_sensors,
        'model_type': model_type,
        'field_shape': field_shape,
        'n_samples': n_samples,
        'final_metrics': final_metrics,
        'history': history,
        'sensor_positions': solver.sensor_optimizer.get_positions().cpu().numpy(),
        'exp_dir': exp_dir
    }
    
    return results


def compare_sensor_numbers(data_type: str = 'combined', model_type: str = 'mlp',
                          sensor_counts: List[int] = [4, 8, 12, 16],
                          results_dir: str = 'sensor_comparison') -> Dict:
    """
    Compare DSPO performance with different numbers of sensors.
    
    Args:
        data_type: Type of synthetic data
        model_type: Type of reconstruction model
        sensor_counts: List of sensor numbers to test
        results_dir: Directory to save results
        
    Returns:
        Comparison results
    """
    print("="*60)
    print(f"Sensor Number Comparison: {data_type} data, {model_type} model")
    print("="*60)
    
    os.makedirs(results_dir, exist_ok=True)
    
    results = {}
    
    for n_sensors in sensor_counts:
        print(f"\nTesting with {n_sensors} sensors...")
        
        exp_results = run_dspo_experiment(
            data_type=data_type,
            n_sensors=n_sensors,
            model_type=model_type,
            n_outer_iterations=3,
            save_results=True,
            results_dir=results_dir
        )
        
        results[n_sensors] = exp_results
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    sensor_nums = list(results.keys())
    final_rmse = [results[n]['final_metrics']['rmse'] for n in sensor_nums]
    final_corr = [results[n]['final_metrics']['correlation'] for n in sensor_nums]
    final_model_loss = [results[n]['history']['model_losses'][-1] for n in sensor_nums]
    final_sensor_loss = [results[n]['history']['sensor_losses'][-1] for n in sensor_nums]
    
    # RMSE vs number of sensors
    axes[0, 0].plot(sensor_nums, final_rmse, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Number of Sensors')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].set_title('Reconstruction RMSE vs Sensor Count')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Correlation vs number of sensors
    axes[0, 1].plot(sensor_nums, final_corr, 'ro-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Number of Sensors')
    axes[0, 1].set_ylabel('Correlation')
    axes[0, 1].set_title('Reconstruction Correlation vs Sensor Count')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Model loss vs number of sensors
    axes[1, 0].plot(sensor_nums, final_model_loss, 'go-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Number of Sensors')
    axes[1, 0].set_ylabel('Model Loss')
    axes[1, 0].set_title('Final Model Loss vs Sensor Count')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Sensor loss vs number of sensors
    axes[1, 1].plot(sensor_nums, final_sensor_loss, 'mo-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Number of Sensors')
    axes[1, 1].set_ylabel('Sensor Loss')
    axes[1, 1].set_title('Final Sensor Loss vs Sensor Count')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Sensor Count Comparison - {data_type} data, {model_type} model')
    plt.tight_layout()
    fig.savefig(f"{results_dir}/sensor_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nComparison results saved to {results_dir}/")
    
    return results


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description='DSPO Demonstration')
    parser.add_argument('--data_type', type=str, default='combined',
                       choices=['gaussian', 'vortex', 'waves', 'combined'],
                       help='Type of synthetic data')
    parser.add_argument('--model_type', type=str, default='mlp',
                       choices=['mlp', 'cnn', 'resmlp', 'multiscale'],
                       help='Type of reconstruction model')
    parser.add_argument('--n_sensors', type=int, default=8,
                       help='Number of sensors')
    parser.add_argument('--n_samples', type=int, default=100,
                       help='Number of data samples')
    parser.add_argument('--n_iterations', type=int, default=3,
                       help='Number of bi-level iterations')
    parser.add_argument('--comparison', action='store_true',
                       help='Run sensor number comparison')
    parser.add_argument('--results_dir', type=str, default='dspo_results',
                       help='Results directory')
    
    args = parser.parse_args()
    
    if args.comparison:
        # Run sensor number comparison
        results = compare_sensor_numbers(
            data_type=args.data_type,
            model_type=args.model_type,
            results_dir=args.results_dir + '_comparison'
        )
    else:
        # Run single experiment
        results = run_dspo_experiment(
            data_type=args.data_type,
            n_sensors=args.n_sensors,
            model_type=args.model_type,
            n_samples=args.n_samples,
            n_outer_iterations=args.n_iterations,
            results_dir=args.results_dir
        )
    
    print("\n" + "="*60)
    print("DSPO Demonstration completed successfully!")
    if not args.comparison:
        print(f"Final RMSE: {results['final_metrics']['rmse']:.6f}")
        print(f"Final Correlation: {results['final_metrics']['correlation']:.4f}")
        print(f"Results saved to: {results['exp_dir']}")
    print("="*60)


if __name__ == "__main__":
    # Run demonstration with different configurations
    print("Running DSPO Simplified Implementation Demo")
    print("This will demonstrate the core DSPO functionality with synthetic data")
    
    # Example 1: Quick demo with combined data
    print("\nExample 1: Combined synthetic data with MLP model")
    results1 = run_dspo_experiment(
        data_type='combined',
        n_sensors=8,
        model_type='mlp',
        n_samples=80,
        n_outer_iterations=2,
        save_results=True,
        results_dir='demo_results'
    )
    
    # Example 2: Gaussian data with CNN model
    print("\nExample 2: Gaussian blob data with CNN model")
    results2 = run_dspo_experiment(
        data_type='gaussian',
        n_sensors=6,
        model_type='cnn',
        n_samples=60,
        n_outer_iterations=2,
        save_results=True,
        results_dir='demo_results'
    )
    
    print("\nDemo completed! Check the 'demo_results' directory for visualizations.")