"""
PrDT4Rec: Experiments and Evaluation

This file contains:
1. Synthetic data generation utilities
2. Experiment configurations
3. Training and evaluation scripts
4. Visualization utilities
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
import json
import os
from datetime import datetime
import virtualTB

from model import (
    Trajectory,
    TrajectoryPreferenceModel,
    PrDT4Rec,
    PrDT4RecTrainer
)


# =============================================================================
# Data Generation Utilities
# =============================================================================

def create_synthetic_trajectories(
    num_trajectories: int = 1000,
    state_dim: int = 64,
    action_dim: int = 100,
    max_length: int = 20,
    min_length: int = 5,
    seed: Optional[int] = None
) -> List[Trajectory]:
    """
    Create synthetic trajectories for testing.
    
    The positive_signals count is correlated with the quality of actions
    (higher action indices = better for this synthetic data).
    
    Args:
        num_trajectories: Number of trajectories to generate
        state_dim: Dimension of state features
        action_dim: Number of possible actions (items)
        max_length: Maximum trajectory length
        min_length: Minimum trajectory length
        seed: Random seed for reproducibility
        
    Returns:
        List of Trajectory objects
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    trajectories = []
    
    for _ in range(num_trajectories):
        length = np.random.randint(min_length, max_length + 1)
        
        # Random states
        states = torch.randn(length, state_dim)
        
        # Random actions with some structure
        actions = torch.randint(0, action_dim, (length,))
        
        # Timesteps
        timesteps = torch.arange(length)
        
        # Positive signals correlated with mean action value (synthetic)
        positive_signals = int(actions.float().mean().item() / action_dim * 10)
        positive_signals = max(0, min(10, positive_signals + np.random.randint(-2, 3)))
        
        trajectories.append(Trajectory(
            states=states,
            actions=actions,
            timesteps=timesteps,
            positive_signals=positive_signals
        ))
    
    return trajectories


def create_user_behavior_trajectories(
    num_users: int = 100,
    trajectories_per_user: int = 10,
    state_dim: int = 64,
    action_dim: int = 100,
    max_length: int = 20,
    min_length: int = 5,
    user_preference_strength: float = 0.5,
    seed: Optional[int] = None
) -> List[Trajectory]:
    """
    Create trajectories that simulate realistic user behavior patterns.
    
    Each user has latent preferences, and their positive signals are correlated
    with how well the recommended actions match their preferences.
    
    Args:
        num_users: Number of unique users
        trajectories_per_user: Number of trajectories per user
        state_dim: Dimension of state features
        action_dim: Number of possible actions (items)
        max_length: Maximum trajectory length
        min_length: Minimum trajectory length
        user_preference_strength: How strongly user preferences affect signals
        seed: Random seed
        
    Returns:
        List of Trajectory objects
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    trajectories = []
    
    for user_id in range(num_users):
        # User's latent preference vector (which items they prefer)
        user_preference = torch.randn(action_dim)
        user_preference = F.softmax(user_preference * user_preference_strength, dim=0)
        
        # User's latent state embedding
        user_state_bias = torch.randn(state_dim) * 0.5
        
        for _ in range(trajectories_per_user):
            length = np.random.randint(min_length, max_length + 1)
            
            # States: user's context changes over session
            states = torch.randn(length, state_dim) + user_state_bias
            
            # Actions: recommendations (some match user preferences, some don't)
            actions = torch.randint(0, action_dim, (length,))
            
            # Timesteps
            timesteps = torch.arange(length)
            
            # Positive signals: based on how well actions match user preferences
            action_prefs = user_preference[actions]
            positive_signals = int((action_prefs.sum().item() / length) * 15)
            positive_signals = max(0, min(10, positive_signals + np.random.randint(-1, 2)))
            
            trajectories.append(Trajectory(
                states=states,
                actions=actions,
                timesteps=timesteps,
                positive_signals=positive_signals
            ))
    
    return trajectories


def split_trajectories(
    trajectories: List[Trajectory],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[List[Trajectory], List[Trajectory], List[Trajectory]]:
    """
    Split trajectories into train/val/test sets.
    
    Args:
        trajectories: List of all trajectories
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        seed: Random seed
        
    Returns:
        (train_trajectories, val_trajectories, test_trajectories)
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = len(trajectories)
    indices = np.random.permutation(n)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    train = [trajectories[i] for i in train_idx]
    val = [trajectories[i] for i in val_idx]
    test = [trajectories[i] for i in test_idx]
    
    return train, val, test


# =============================================================================
# Evaluation Metrics
# =============================================================================

def evaluate_policy(
    trainer: PrDT4RecTrainer,
    test_trajectories: List[Trajectory],
    max_seq_len: int = 100
) -> Dict[str, float]:
    """
    Evaluate the trained policy on test trajectories.
    
    Metrics:
    - Action accuracy: How often the policy predicts the correct action
    - Top-k accuracy: How often the correct action is in top-k predictions
    
    Args:
        trainer: Trained PrDT4RecTrainer
        test_trajectories: Test trajectories
        max_seq_len: Maximum sequence length
        
    Returns:
        Dictionary of evaluation metrics
    """
    trainer.policy_model.eval()
    device = trainer.device
    
    total_correct = 0
    total_top5_correct = 0
    total_top10_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for traj in test_trajectories:
            seq_len = len(traj.actions)
            
            states = traj.states.unsqueeze(0).to(device)
            actions = traj.actions.unsqueeze(0).to(device)
            timesteps = traj.timesteps.unsqueeze(0).to(device)
            
            # Pad if necessary
            if seq_len < max_seq_len:
                pad_len = max_seq_len - seq_len
                states = F.pad(states, (0, 0, 0, pad_len))
                actions = F.pad(actions, (0, pad_len))
                timesteps = F.pad(timesteps, (0, pad_len))
                mask = torch.cat([torch.ones(seq_len), torch.zeros(pad_len)]).unsqueeze(0).to(device)
            else:
                states = states[:, :max_seq_len]
                actions = actions[:, :max_seq_len]
                timesteps = timesteps[:, :max_seq_len]
                mask = torch.ones(1, max_seq_len).to(device)
                seq_len = max_seq_len
            
            # Get predictions
            logits = trainer.policy_model(states, actions, timesteps, mask)
            predictions = logits.argmax(dim=-1)
            
            # Top-k predictions
            top5 = logits.topk(5, dim=-1).indices
            top10 = logits.topk(10, dim=-1).indices
            
            # Compute metrics for valid positions
            for t in range(seq_len):
                total_tokens += 1
                target = actions[0, t].item()
                pred = predictions[0, t].item()
                
                if pred == target:
                    total_correct += 1
                if target in top5[0, t].tolist():
                    total_top5_correct += 1
                if target in top10[0, t].tolist():
                    total_top10_correct += 1
    
    metrics = {
        'accuracy': total_correct / total_tokens if total_tokens > 0 else 0,
        'top5_accuracy': total_top5_correct / total_tokens if total_tokens > 0 else 0,
        'top10_accuracy': total_top10_correct / total_tokens if total_tokens > 0 else 0,
        'total_tokens': total_tokens
    }
    
    return metrics


def evaluate_preference_model(
    trainer: PrDT4RecTrainer,
    test_trajectories: List[Trajectory],
    max_seq_len: int = 100
) -> Dict[str, float]:
    """
    Evaluate the preference model's ability to rank trajectories.
    
    Metrics:
    - Ranking accuracy: How often higher-signal trajectories get higher scores
    - Correlation: Spearman correlation between scores and positive signals
    
    Args:
        trainer: Trained PrDT4RecTrainer
        test_trajectories: Test trajectories
        max_seq_len: Maximum sequence length
        
    Returns:
        Dictionary of evaluation metrics
    """
    trainer.preference_model.eval()
    device = trainer.device
    
    scores = []
    signals = []
    
    with torch.no_grad():
        for traj in test_trajectories:
            seq_len = len(traj.actions)
            
            states = traj.states.unsqueeze(0).to(device)
            actions = traj.actions.unsqueeze(0).to(device)
            timesteps = traj.timesteps.unsqueeze(0).to(device)
            mask = torch.ones(1, seq_len).to(device)
            
            # Pad if necessary
            if seq_len < max_seq_len:
                pad_len = max_seq_len - seq_len
                states = F.pad(states, (0, 0, 0, pad_len))
                actions = F.pad(actions, (0, pad_len))
                timesteps = F.pad(timesteps, (0, pad_len))
                mask = F.pad(mask, (0, pad_len))
            
            score = trainer.preference_model(states, actions, timesteps, mask)
            scores.append(score.item())
            signals.append(traj.positive_signals)
    
    scores = np.array(scores)
    signals = np.array(signals)
    
    # Pairwise ranking accuracy
    correct_pairs = 0
    total_pairs = 0
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            if signals[i] != signals[j]:
                total_pairs += 1
                if (signals[i] > signals[j] and scores[i] > scores[j]) or \
                   (signals[i] < signals[j] and scores[i] < scores[j]):
                    correct_pairs += 1
    
    # Spearman correlation
    from scipy.stats import spearmanr
    try:
        correlation, p_value = spearmanr(scores, signals)
    except:
        correlation, p_value = 0, 1
    
    metrics = {
        'pairwise_ranking_accuracy': correct_pairs / total_pairs if total_pairs > 0 else 0,
        'spearman_correlation': correlation,
        'p_value': p_value,
        'total_pairs': total_pairs
    }
    
    return metrics


# =============================================================================
# Experiment Configurations
# =============================================================================

class ExperimentConfig:
    """Configuration for experiments."""
    
    def __init__(
        self,
        name: str = "default",
        # Data parameters
        num_trajectories: int = 1000,
        state_dim: int = 64,
        action_dim: int = 100,
        max_seq_len: int = 20,
        min_seq_len: int = 5,
        # Model parameters
        hidden_dim: int = 128,
        n_heads: int = 4,
        preference_n_layers: int = 2,
        policy_n_layers: int = 3,
        dropout: float = 0.1,
        # Training parameters
        preference_epochs: int = 50,
        policy_epochs: int = 100,
        batch_size: int = 32,
        preference_lr: float = 1e-4,
        policy_lr: float = 1e-4,
        regularization_lambda: float = 0.01,
        # Other
        seed: int = 42,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.name = name
        self.num_trajectories = num_trajectories
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.preference_n_layers = preference_n_layers
        self.policy_n_layers = policy_n_layers
        self.dropout = dropout
        self.preference_epochs = preference_epochs
        self.policy_epochs = policy_epochs
        self.batch_size = batch_size
        self.preference_lr = preference_lr
        self.policy_lr = policy_lr
        self.regularization_lambda = regularization_lambda
        self.seed = seed
        self.device = device
    
    def to_dict(self) -> Dict:
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ExperimentConfig':
        return cls(**d)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_experiment(config: ExperimentConfig, verbose: bool = True) -> Dict:
    """
    Run a complete PrDT4Rec experiment.
    
    Args:
        config: Experiment configuration
        verbose: Whether to print progress
        
    Returns:
        Dictionary containing training history and evaluation metrics
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running Experiment: {config.name}")
        print(f"{'='*60}")
    
    # Set seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # Create data
    if verbose:
        print("\nCreating synthetic data...")
    
    trajectories = create_synthetic_trajectories(
        num_trajectories=config.num_trajectories,
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        max_length=config.max_seq_len,
        min_length=config.min_seq_len,
        seed=config.seed
    )
    
    # Split data
    train_trajs, val_trajs, test_trajs = split_trajectories(
        trajectories,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=config.seed
    )
    
    if verbose:
        print(f"Data split: {len(train_trajs)} train, {len(val_trajs)} val, {len(test_trajs)} test")
        print(f"Positive signals distribution:")
        signals = [t.positive_signals for t in trajectories]
        print(f"  min={min(signals)}, max={max(signals)}, mean={np.mean(signals):.2f}")
    
    # Initialize models
    preference_model = TrajectoryPreferenceModel(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
        n_heads=config.n_heads,
        n_layers=config.preference_n_layers,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout
    )
    
    policy_model = PrDT4Rec(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
        n_heads=config.n_heads,
        n_layers=config.policy_n_layers,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout
    )
    
    # Create trainer
    trainer = PrDT4RecTrainer(
        preference_model=preference_model,
        policy_model=policy_model,
        device=config.device
    )
    
    # Train
    history = trainer.train(
        trajectories=train_trajs,
        preference_epochs=config.preference_epochs,
        policy_epochs=config.policy_epochs,
        batch_size=config.batch_size,
        preference_lr=config.preference_lr,
        policy_lr=config.policy_lr,
        regularization_lambda=config.regularization_lambda,
        max_seq_len=config.max_seq_len
    )
    
    # Evaluate
    if verbose:
        print("\n" + "="*50)
        print("Evaluation")
        print("="*50)
    
    # Preference model evaluation
    pref_metrics = evaluate_preference_model(trainer, test_trajs, config.max_seq_len)
    if verbose:
        print(f"\nPreference Model Metrics:")
        print(f"  Pairwise Ranking Accuracy: {pref_metrics['pairwise_ranking_accuracy']:.4f}")
        print(f"  Spearman Correlation: {pref_metrics['spearman_correlation']:.4f}")
    
    # Policy evaluation
    policy_metrics = evaluate_policy(trainer, test_trajs, config.max_seq_len)
    if verbose:
        print(f"\nPolicy Model Metrics:")
        print(f"  Accuracy: {policy_metrics['accuracy']:.4f}")
        print(f"  Top-5 Accuracy: {policy_metrics['top5_accuracy']:.4f}")
        print(f"  Top-10 Accuracy: {policy_metrics['top10_accuracy']:.4f}")
    
    # Test trajectory generation
    if verbose:
        print("\nTesting trajectory generation...")
    
    initial_state = torch.randn(config.state_dim).to(config.device)
    generated_states, generated_actions = trainer.policy_model.generate_trajectory(
        initial_state=initial_state,
        max_length=10
    )
    
    if verbose:
        print(f"Generated trajectory (actions): {generated_actions.tolist()}")
    
    results = {
        'config': config.to_dict(),
        'history': history,
        'preference_metrics': pref_metrics,
        'policy_metrics': policy_metrics,
        'generated_actions': generated_actions.tolist()
    }
    
    return results, trainer


def run_ablation_study(
    base_config: ExperimentConfig,
    param_name: str,
    param_values: List,
    verbose: bool = True
) -> List[Dict]:
    """
    Run an ablation study varying one parameter.
    
    Args:
        base_config: Base experiment configuration
        param_name: Name of parameter to vary
        param_values: List of values to try
        verbose: Whether to print progress
        
    Returns:
        List of experiment results
    """
    results = []
    
    for value in param_values:
        # Create config with modified parameter
        config_dict = base_config.to_dict()
        config_dict[param_name] = value
        config_dict['name'] = f"{base_config.name}_{param_name}={value}"
        
        config = ExperimentConfig.from_dict(config_dict)
        
        result, _ = run_experiment(config, verbose=verbose)
        results.append(result)
    
    return results


# =============================================================================
# Visualization Utilities
# =============================================================================

def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """
    Plot training history curves.
    
    Args:
        history: Training history from trainer.train()
        save_path: Optional path to save the figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Preference model loss
    ax = axes[0, 0]
    ax.plot(history['preference']['total_loss'], label='Total Loss')
    ax.plot(history['preference']['bt_loss'], label='BT Loss')
    ax.plot(history['preference']['reg_loss'], label='Reg Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Preference Model Training Loss')
    ax.legend()
    ax.grid(True)
    
    # Preference model accuracy
    ax = axes[0, 1]
    ax.plot(history['preference']['accuracy'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Preference Model Pairwise Accuracy')
    ax.grid(True)
    
    # Policy loss
    ax = axes[1, 0]
    ax.plot(history['policy']['policy_loss'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Policy Model Training Loss')
    ax.grid(True)
    
    # Policy accuracy
    ax = axes[1, 1]
    ax.plot(history['policy']['accuracy'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Policy Model Action Prediction Accuracy')
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def print_experiment_summary(results: Dict):
    """Print a summary of experiment results."""
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    print(f"\nExperiment: {results['config']['name']}")
    print(f"Device: {results['config']['device']}")
    
    print("\n--- Configuration ---")
    print(f"Trajectories: {results['config']['num_trajectories']}")
    print(f"State dim: {results['config']['state_dim']}")
    print(f"Action dim: {results['config']['action_dim']}")
    print(f"Max seq len: {results['config']['max_seq_len']}")
    print(f"Hidden dim: {results['config']['hidden_dim']}")
    
    print("\n--- Preference Model Results ---")
    pm = results['preference_metrics']
    print(f"Pairwise Ranking Accuracy: {pm['pairwise_ranking_accuracy']:.4f}")
    print(f"Spearman Correlation: {pm['spearman_correlation']:.4f}")
    
    print("\n--- Policy Model Results ---")
    pol = results['policy_metrics']
    print(f"Action Accuracy: {pol['accuracy']:.4f}")
    print(f"Top-5 Accuracy: {pol['top5_accuracy']:.4f}")
    print(f"Top-10 Accuracy: {pol['top10_accuracy']:.4f}")
    
    print("\n" + "="*60)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Default experiment configuration
    config = ExperimentConfig(
        name="prdt4rec_demo",
        num_trajectories=500,
        state_dim=64,
        action_dim=100,
        max_seq_len=20,
        min_seq_len=5,
        hidden_dim=128,
        n_heads=4,
        preference_n_layers=2,
        policy_n_layers=3,
        dropout=0.1,
        preference_epochs=5,   # Use more epochs in practice (e.g., 50)
        policy_epochs=5,       # Use more epochs in practice (e.g., 100)
        batch_size=32,
        preference_lr=1e-4,
        policy_lr=1e-4,
        regularization_lambda=0.01,
        seed=42
    )
    
    # Run experiment
    results, trainer = run_experiment(config, verbose=True)
    
    # Print summary
    print_experiment_summary(results)
    
    # Optional: Plot training history (requires matplotlib)
    try:
        plot_training_history(results['history'], save_path='training_history.png')
    except Exception as e:
        print(f"Could not plot training history: {e}")
    
    # Optional: Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"results_{config.name}_{timestamp}.json"
    
    # Convert non-serializable items
    results_serializable = {
        'config': results['config'],
        'preference_metrics': results['preference_metrics'],
        'policy_metrics': results['policy_metrics'],
        'generated_actions': results['generated_actions'],
        'final_preference_loss': results['history']['preference']['total_loss'][-1],
        'final_preference_accuracy': results['history']['preference']['accuracy'][-1],
        'final_policy_loss': results['history']['policy']['policy_loss'][-1],
        'final_policy_accuracy': results['history']['policy']['accuracy'][-1],
    }
    
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    print("\nPrDT4Rec experiment complete!")