"""
PrDT4Rec: Preference-Guided Decision Transformer for Recommendation

Model Components:
1. TrajectoryEncoder: Encodes trajectories into fixed-size representations
2. TrajectoryPreferenceModel: Learns trajectory-level preferences using Bradley-Terry loss
3. PrDT4Rec: The main policy model using preference-weighted imitation learning
4. PrDT4RecTrainer: Handles the two-stage training process
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm

# Assuming CDT4Rec is already implemented and can be imported
# from cdt4rec import CDT4Rec


@dataclass
class Trajectory:
    """
    Represents a recommendation session trajectory.
    
    Attributes:
        states: Tensor of shape (T, state_dim) - user states at each timestep
        actions: Tensor of shape (T,) - recommended items at each timestep
        timesteps: Tensor of shape (T,) - timestep indices
        positive_signals: Number of positive user signals (e.g., clicks) in trajectory
    """
    states: torch.Tensor
    actions: torch.Tensor
    timesteps: torch.Tensor
    positive_signals: int = 0


class TrajectoryEncoder(nn.Module):
    """
    Encodes a trajectory into a fixed-size representation using a Transformer.
    This is the backbone for the preference model f_θ(τ).
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 100,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Embeddings for states and actions
        self.state_encoder = nn.Linear(state_dim, hidden_dim)
        self.action_encoder = nn.Embedding(action_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Embedding(max_seq_len, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Layer norm
        self.ln = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode a batch of trajectories.
        
        Args:
            states: (batch, seq_len, state_dim)
            actions: (batch, seq_len)
            timesteps: (batch, seq_len)
            attention_mask: (batch, seq_len) - 1 for valid, 0 for padding
            
        Returns:
            trajectory_embedding: (batch, hidden_dim)
        """
        batch_size, seq_len = states.shape[:2]
        
        # Encode states and actions
        state_emb = self.state_encoder(states)  # (batch, seq_len, hidden_dim)
        action_emb = self.action_encoder(actions)  # (batch, seq_len, hidden_dim)
        
        # Add positional encoding
        pos_emb = self.pos_encoding(timesteps)  # (batch, seq_len, hidden_dim)
        
        # Combine embeddings: interleave state and action
        # For simplicity, we sum them (alternative: concatenate and project)
        combined = state_emb + action_emb + pos_emb
        combined = self.ln(combined)
        
        # Create attention mask for transformer (True = ignore)
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None
        
        # Apply transformer
        encoded = self.transformer(combined, src_key_padding_mask=src_key_padding_mask)
        
        # Aggregate: mean pooling over sequence (excluding padding)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            trajectory_emb = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            trajectory_emb = encoded.mean(dim=1)
        
        return trajectory_emb


class TrajectoryPreferenceModel(nn.Module):
    """
    Trajectory-level preference model f_θ(τ) that maps a complete recommendation
    session to a scalar score.
    
    Trained with Bradley-Terry pairwise comparison loss:
    L_pref(θ) = E[-log σ(f_θ(τ_i) - f_θ(τ_j))] + λ E[||f_θ(τ)||²]
    
    where τ_i ≻ τ_j means τ_i has more positive user signals than τ_j.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 100,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.encoder = TrajectoryEncoder(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        # Score head: maps trajectory embedding to scalar
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute preference score for trajectories.
        
        Args:
            states: (batch, seq_len, state_dim)
            actions: (batch, seq_len)
            timesteps: (batch, seq_len)
            attention_mask: (batch, seq_len)
            
        Returns:
            scores: (batch,) - scalar preference scores
        """
        trajectory_emb = self.encoder(states, actions, timesteps, attention_mask)
        scores = self.score_head(trajectory_emb).squeeze(-1)
        return scores
    
    def compute_pairwise_loss(
        self,
        preferred_states: torch.Tensor,
        preferred_actions: torch.Tensor,
        preferred_timesteps: torch.Tensor,
        preferred_mask: torch.Tensor,
        rejected_states: torch.Tensor,
        rejected_actions: torch.Tensor,
        rejected_timesteps: torch.Tensor,
        rejected_mask: torch.Tensor,
        regularization_lambda: float = 0.01
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Bradley-Terry pairwise comparison loss.
        
        L_pref(θ) = E[-log σ(f_θ(τ_i) - f_θ(τ_j))] + λ E[||f_θ(τ)||²]
        
        Args:
            preferred_*: Batch of preferred trajectories (τ_i ≻ τ_j)
            rejected_*: Batch of rejected trajectories
            regularization_lambda: λ for L2 regularization
            
        Returns:
            loss: Scalar loss value
            metrics: Dictionary of training metrics
        """
        # Compute scores for preferred and rejected trajectories
        preferred_scores = self.forward(
            preferred_states, preferred_actions, preferred_timesteps, preferred_mask
        )
        rejected_scores = self.forward(
            rejected_states, rejected_actions, rejected_timesteps, rejected_mask
        )
        
        # Bradley-Terry loss: -log σ(f(τ_i) - f(τ_j))
        score_diff = preferred_scores - rejected_scores
        bt_loss = -F.logsigmoid(score_diff).mean()
        
        # L2 regularization on scores
        reg_loss = regularization_lambda * (
            preferred_scores.pow(2).mean() + rejected_scores.pow(2).mean()
        )
        
        total_loss = bt_loss + reg_loss
        
        # Compute accuracy (how often preferred > rejected)
        accuracy = (score_diff > 0).float().mean().item()
        
        metrics = {
            'bt_loss': bt_loss.item(),
            'reg_loss': reg_loss.item(),
            'total_loss': total_loss.item(),
            'accuracy': accuracy,
            'avg_score_diff': score_diff.mean().item()
        }
        
        return total_loss, metrics


class PreferencePairDataset(Dataset):
    """
    Dataset for training the preference model with pairwise comparisons.
    
    Pairwise labels are defined as τ_i ≻ τ_j when τ_i contains more positive
    user signals (e.g., clicks) than τ_j.
    """
    
    def __init__(
        self,
        trajectories: List[Trajectory],
        max_seq_len: int = 100,
        num_pairs_per_trajectory: int = 5
    ):
        """
        Args:
            trajectories: List of Trajectory objects
            max_seq_len: Maximum sequence length for padding
            num_pairs_per_trajectory: Number of comparison pairs to generate per trajectory
        """
        self.trajectories = trajectories
        self.max_seq_len = max_seq_len
        self.pairs = self._generate_pairs(num_pairs_per_trajectory)
        
    def _generate_pairs(self, num_pairs: int) -> List[Tuple[int, int]]:
        """Generate pairwise comparison indices based on positive signals."""
        pairs = []
        n = len(self.trajectories)
        
        for i in range(n):
            traj_i = self.trajectories[i]
            # Find trajectories with fewer positive signals
            for _ in range(num_pairs):
                # Randomly sample a trajectory with fewer positive signals
                candidates = [
                    j for j in range(n)
                    if self.trajectories[j].positive_signals < traj_i.positive_signals
                ]
                if candidates:
                    j = np.random.choice(candidates)
                    pairs.append((i, j))  # i is preferred over j
                    
        return pairs
    
    def _pad_trajectory(self, traj: Trajectory) -> Tuple[torch.Tensor, ...]:
        """Pad trajectory to max_seq_len."""
        seq_len = len(traj.actions)
        pad_len = self.max_seq_len - seq_len
        
        if pad_len > 0:
            states = F.pad(traj.states, (0, 0, 0, pad_len))
            actions = F.pad(traj.actions, (0, pad_len))
            timesteps = F.pad(traj.timesteps, (0, pad_len))
            mask = torch.cat([torch.ones(seq_len), torch.zeros(pad_len)])
        else:
            states = traj.states[:self.max_seq_len]
            actions = traj.actions[:self.max_seq_len]
            timesteps = traj.timesteps[:self.max_seq_len]
            mask = torch.ones(self.max_seq_len)
            
        return states, actions, timesteps, mask
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i, j = self.pairs[idx]
        
        preferred = self._pad_trajectory(self.trajectories[i])
        rejected = self._pad_trajectory(self.trajectories[j])
        
        return {
            'preferred_states': preferred[0],
            'preferred_actions': preferred[1],
            'preferred_timesteps': preferred[2],
            'preferred_mask': preferred[3],
            'rejected_states': rejected[0],
            'rejected_actions': rejected[1],
            'rejected_timesteps': rejected[2],
            'rejected_mask': rejected[3],
        }


class TrajectoryDataset(Dataset):
    """Dataset for policy training with preference weights."""
    
    def __init__(
        self,
        trajectories: List[Trajectory],
        weights: torch.Tensor,
        max_seq_len: int = 100
    ):
        self.trajectories = trajectories
        self.weights = weights
        self.max_seq_len = max_seq_len
        
    def _pad_trajectory(self, traj: Trajectory) -> Tuple[torch.Tensor, ...]:
        """Pad trajectory to max_seq_len."""
        seq_len = len(traj.actions)
        pad_len = self.max_seq_len - seq_len
        
        if pad_len > 0:
            states = F.pad(traj.states, (0, 0, 0, pad_len))
            actions = F.pad(traj.actions, (0, pad_len))
            timesteps = F.pad(traj.timesteps, (0, pad_len))
            mask = torch.cat([torch.ones(seq_len), torch.zeros(pad_len)])
        else:
            states = traj.states[:self.max_seq_len]
            actions = traj.actions[:self.max_seq_len]
            timesteps = traj.timesteps[:self.max_seq_len]
            mask = torch.ones(self.max_seq_len)
            
        return states, actions, timesteps, mask
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj = self.trajectories[idx]
        states, actions, timesteps, mask = self._pad_trajectory(traj)
        
        return {
            'states': states,
            'actions': actions,
            'timesteps': timesteps,
            'mask': mask,
            'weight': self.weights[idx]
        }


class PrDT4Rec(nn.Module):
    """
    Preference-Guided Decision Transformer for Recommendation (PrDT4Rec).
    
    This model uses CDT4Rec as the policy backbone but replaces return-to-go
    targets with preference guidance. The policy is trained using preference-weighted
    imitation learning.
    
    Key differences from CDT4Rec:
    1. No return-to-go conditioning - input is only (s_t, a_{t-1}) pairs
    2. Training uses soft preference weights computed from learned preference model
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        max_seq_len: int = 100,
        dropout: float = 0.1
    ):
        """
        Args:
            state_dim: Dimension of state features
            action_dim: Number of possible actions (items)
            hidden_dim: Hidden dimension for transformer
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # State and action embeddings
        self.state_encoder = nn.Linear(state_dim, hidden_dim)
        self.action_encoder = nn.Embedding(action_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Embedding(max_seq_len, hidden_dim)
        
        # Input embedding layer norm
        self.embed_ln = nn.LayerNorm(hidden_dim)
        
        # Transformer decoder (causal)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Generate causal mask
        self.register_buffer(
            'causal_mask',
            self._generate_causal_mask(max_seq_len * 2)  # *2 for interleaved state-action
        )
        
    def _generate_causal_mask(self, size: int) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask
    
    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for action prediction.
        
        The input sequence contains only (s_t, a_{t-1}) pairs - no return information.
        
        Args:
            states: (batch, seq_len, state_dim)
            actions: (batch, seq_len) - actions at previous timesteps
            timesteps: (batch, seq_len)
            attention_mask: (batch, seq_len)
            
        Returns:
            action_logits: (batch, seq_len, action_dim)
        """
        batch_size, seq_len = states.shape[:2]
        
        # Encode states and actions
        state_emb = self.state_encoder(states)  # (batch, seq_len, hidden_dim)
        action_emb = self.action_encoder(actions)  # (batch, seq_len, hidden_dim)
        
        # Add positional encoding
        pos_emb = self.pos_encoding(timesteps)
        
        # Interleave states and actions: [s_1, a_0, s_2, a_1, ...]
        # For simplicity, we'll use a different approach:
        # Condition on state and previous action together
        combined = state_emb + pos_emb
        
        # Shift actions by 1 to get a_{t-1} for predicting a_t
        shifted_actions = torch.zeros_like(action_emb)
        shifted_actions[:, 1:] = action_emb[:, :-1]
        combined = combined + shifted_actions
        
        combined = self.embed_ln(combined)
        
        # Apply causal transformer
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        
        if attention_mask is not None:
            # Combine causal mask with padding mask
            padding_mask = ~attention_mask.bool()
        else:
            padding_mask = None
        
        # Use transformer decoder with self-attention only (no cross-attention)
        # We pass combined as both memory and target
        output = self.transformer(
            combined,
            combined,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=padding_mask
        )
        
        # Predict actions
        action_logits = self.action_head(output)
        
        return action_logits
    
    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor,
        weights: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute preference-weighted policy loss.
        
        L_policy(ψ) = Σ_τ w(τ) Σ_t log π_ψ(a_t | s_≤t, a_<t)
        
        Args:
            states: (batch, seq_len, state_dim)
            actions: (batch, seq_len) - target actions
            timesteps: (batch, seq_len)
            attention_mask: (batch, seq_len)
            weights: (batch,) - preference weights w(τ)
            
        Returns:
            loss: Scalar loss
            metrics: Training metrics
        """
        # Get action logits
        action_logits = self.forward(states, actions, timesteps, attention_mask)
        
        # Compute cross-entropy loss for each timestep
        # Reshape for cross_entropy: (batch * seq_len, action_dim) vs (batch * seq_len,)
        batch_size, seq_len = actions.shape
        
        logits_flat = action_logits.view(-1, self.action_dim)
        actions_flat = actions.view(-1)
        mask_flat = attention_mask.view(-1)
        
        # Per-token cross-entropy
        ce_loss = F.cross_entropy(logits_flat, actions_flat, reduction='none')
        ce_loss = ce_loss.view(batch_size, seq_len)
        
        # Apply attention mask
        ce_loss = ce_loss * attention_mask
        
        # Sum over sequence for each trajectory
        trajectory_loss = ce_loss.sum(dim=1)  # (batch,)
        
        # Apply preference weights (negative because we maximize log-likelihood)
        # Note: weights should sum to 1 (softmax normalized)
        weighted_loss = (weights * trajectory_loss).sum()
        
        # Normalize by total tokens
        total_tokens = attention_mask.sum()
        normalized_loss = weighted_loss / total_tokens.clamp(min=1)
        
        # Compute metrics
        with torch.no_grad():
            predictions = action_logits.argmax(dim=-1)
            correct = ((predictions == actions) * attention_mask).sum()
            accuracy = correct / total_tokens.clamp(min=1)
        
        metrics = {
            'policy_loss': normalized_loss.item(),
            'accuracy': accuracy.item(),
            'avg_weight': weights.mean().item()
        }
        
        return normalized_loss, metrics
    
    def predict(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict next action given current state and history.
        
        a_t = argmax_a π_ψ(a | s_≤t, a_<t)
        
        Args:
            states: (batch, seq_len, state_dim) - including current state
            actions: (batch, seq_len) - previous actions (last one is placeholder)
            timesteps: (batch, seq_len)
            
        Returns:
            predicted_actions: (batch,) - predicted action for current timestep
        """
        self.eval()
        with torch.no_grad():
            action_logits = self.forward(states, actions, timesteps)
            # Get prediction for the last timestep
            predicted_actions = action_logits[:, -1, :].argmax(dim=-1)
        return predicted_actions
    
    def generate_trajectory(
        self,
        initial_state: torch.Tensor,
        max_length: int,
        get_next_state_fn: Optional[callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a recommendation trajectory autoregressively.
        
        Args:
            initial_state: (state_dim,) - initial user state
            max_length: Maximum trajectory length T
            get_next_state_fn: Function to get next state given current state and action
                               If None, assumes state doesn't change
                               
        Returns:
            states: (max_length, state_dim)
            actions: (max_length,)
        """
        self.eval()
        device = initial_state.device
        
        states_list = [initial_state.unsqueeze(0)]  # List of (1, state_dim)
        actions_list = [torch.zeros(1, dtype=torch.long, device=device)]  # Placeholder
        
        with torch.no_grad():
            for t in range(max_length):
                # Prepare inputs
                states = torch.stack([s.squeeze(0) for s in states_list], dim=0).unsqueeze(0)
                actions = torch.stack(actions_list, dim=0).unsqueeze(0)
                timesteps = torch.arange(len(states_list), device=device).unsqueeze(0)
                
                # Predict action
                action_logits = self.forward(states, actions, timesteps)
                action = action_logits[0, -1, :].argmax()
                
                # Update actions list (replace placeholder with actual action)
                actions_list[-1] = action.unsqueeze(0)
                
                if t < max_length - 1:
                    # Get next state
                    if get_next_state_fn is not None:
                        next_state = get_next_state_fn(states_list[-1], action)
                    else:
                        next_state = states_list[-1]  # State unchanged
                    
                    states_list.append(next_state)
                    actions_list.append(torch.zeros(1, dtype=torch.long, device=device))
        
        final_states = torch.cat(states_list, dim=0)
        final_actions = torch.cat(actions_list, dim=0)
        
        return final_states, final_actions


class PrDT4RecTrainer:
    """
    Trainer for PrDT4Rec that handles the two-stage training process:
    1. Train preference model on pairwise comparisons
    2. Train policy with preference-weighted imitation
    """
    
    def __init__(
        self,
        preference_model: TrajectoryPreferenceModel,
        policy_model: PrDT4Rec,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.preference_model = preference_model.to(device)
        self.policy_model = policy_model.to(device)
        self.device = device
        
        # Preference scores cache
        self._preference_scores = None
        self._preference_weights = None
        
    def train_preference_model(
        self,
        trajectories: List[Trajectory],
        num_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        regularization_lambda: float = 0.01,
        num_pairs_per_trajectory: int = 5,
        max_seq_len: int = 100
    ) -> Dict[str, List[float]]:
        """
        Stage 1: Train the preference model using Bradley-Terry loss.
        
        Args:
            trajectories: List of offline trajectories
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            regularization_lambda: L2 regularization coefficient λ
            num_pairs_per_trajectory: Number of comparison pairs per trajectory
            max_seq_len: Maximum sequence length
            
        Returns:
            history: Training history
        """
        print("Stage 1: Training Preference Model")
        print("=" * 50)
        
        # Create dataset
        dataset = PreferencePairDataset(
            trajectories,
            max_seq_len=max_seq_len,
            num_pairs_per_trajectory=num_pairs_per_trajectory
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.preference_model.parameters(),
            lr=learning_rate
        )
        
        history = {
            'bt_loss': [],
            'reg_loss': [],
            'total_loss': [],
            'accuracy': []
        }
        
        self.preference_model.train()
        
        for epoch in range(num_epochs):
            epoch_metrics = {k: [] for k in history.keys()}
            
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                optimizer.zero_grad()
                
                loss, metrics = self.preference_model.compute_pairwise_loss(
                    preferred_states=batch['preferred_states'],
                    preferred_actions=batch['preferred_actions'],
                    preferred_timesteps=batch['preferred_timesteps'],
                    preferred_mask=batch['preferred_mask'],
                    rejected_states=batch['rejected_states'],
                    rejected_actions=batch['rejected_actions'],
                    rejected_timesteps=batch['rejected_timesteps'],
                    rejected_mask=batch['rejected_mask'],
                    regularization_lambda=regularization_lambda
                )
                
                loss.backward()
                optimizer.step()
                
                for k, v in metrics.items():
                    if k in epoch_metrics:
                        epoch_metrics[k].append(v)
            
            # Average metrics
            for k in history.keys():
                avg_val = np.mean(epoch_metrics[k])
                history[k].append(avg_val)
            
            print(f"Epoch {epoch+1}: Loss={history['total_loss'][-1]:.4f}, "
                  f"Accuracy={history['accuracy'][-1]:.4f}")
        
        # Freeze preference model after training
        for param in self.preference_model.parameters():
            param.requires_grad = False
        
        print("\nPreference model training complete. Model frozen.")
        
        return history
    
    def compute_preference_weights(
        self,
        trajectories: List[Trajectory],
        max_seq_len: int = 100
    ) -> torch.Tensor:
        """
        Compute preference weights for all trajectories.
        
        w(τ) = exp(f_θ(τ) - f̄) / Σ_τ' exp(f_θ(τ') - f̄)
        
        where f̄ is the mean preference score.
        
        Args:
            trajectories: List of offline trajectories
            max_seq_len: Maximum sequence length
            
        Returns:
            weights: (num_trajectories,) normalized weights
        """
        print("\nComputing preference weights...")
        
        self.preference_model.eval()
        
        scores = []
        
        with torch.no_grad():
            for traj in tqdm(trajectories, desc="Scoring trajectories"):
                # Prepare trajectory
                seq_len = len(traj.actions)
                
                states = traj.states.unsqueeze(0).to(self.device)
                actions = traj.actions.unsqueeze(0).to(self.device)
                timesteps = traj.timesteps.unsqueeze(0).to(self.device)
                mask = torch.ones(1, seq_len).to(self.device)
                
                # Pad if necessary
                if seq_len < max_seq_len:
                    pad_len = max_seq_len - seq_len
                    states = F.pad(states, (0, 0, 0, pad_len))
                    actions = F.pad(actions, (0, pad_len))
                    timesteps = F.pad(timesteps, (0, pad_len))
                    mask = F.pad(mask, (0, pad_len))
                else:
                    states = states[:, :max_seq_len]
                    actions = actions[:, :max_seq_len]
                    timesteps = timesteps[:, :max_seq_len]
                    mask = mask[:, :max_seq_len]
                
                score = self.preference_model(states, actions, timesteps, mask)
                scores.append(score.item())
        
        scores = torch.tensor(scores)
        
        # Compute weights with mean subtraction for numerical stability
        # w(τ) = exp(f_θ(τ) - f̄) / Σ exp(f_θ(τ') - f̄)
        f_bar = scores.mean()
        centered_scores = scores - f_bar
        weights = F.softmax(centered_scores, dim=0)
        
        self._preference_scores = scores
        self._preference_weights = weights
        
        print(f"Weights computed. Mean: {weights.mean():.6f}, "
              f"Std: {weights.std():.6f}, "
              f"Max: {weights.max():.6f}, Min: {weights.min():.6f}")
        
        return weights
    
    def train_policy(
        self,
        trajectories: List[Trajectory],
        num_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        max_seq_len: int = 100,
        recompute_weights_every: int = 1  # Recompute weights every N epochs
    ) -> Dict[str, List[float]]:
        """
        Stage 2: Train the policy model with preference-weighted imitation.
        
        L_policy(ψ) = Σ_τ w(τ) Σ_t log π_ψ(a_t | s_≤t, a_<t)
        
        Args:
            trajectories: List of offline trajectories
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            max_seq_len: Maximum sequence length
            recompute_weights_every: Recompute preference weights every N epochs
            
        Returns:
            history: Training history
        """
        print("\nStage 2: Training Policy Model")
        print("=" * 50)
        
        # Initial weight computation
        weights = self.compute_preference_weights(trajectories, max_seq_len)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=learning_rate
        )
        
        history = {
            'policy_loss': [],
            'accuracy': []
        }
        
        self.policy_model.train()
        
        for epoch in range(num_epochs):
            # Recompute weights periodically (they're held fixed within each epoch)
            if epoch > 0 and epoch % recompute_weights_every == 0:
                weights = self.compute_preference_weights(trajectories, max_seq_len)
            
            # Create dataset with current weights
            dataset = TrajectoryDataset(trajectories, weights, max_seq_len)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            epoch_metrics = {k: [] for k in history.keys()}
            
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                optimizer.zero_grad()
                
                loss, metrics = self.policy_model.compute_loss(
                    states=batch['states'],
                    actions=batch['actions'],
                    timesteps=torch.arange(max_seq_len, device=self.device).unsqueeze(0).expand(batch['states'].shape[0], -1),
                    attention_mask=batch['mask'],
                    weights=batch['weight']
                )
                
                loss.backward()
                optimizer.step()
                
                for k, v in metrics.items():
                    if k in epoch_metrics:
                        epoch_metrics[k].append(v)
            
            # Average metrics
            for k in history.keys():
                avg_val = np.mean(epoch_metrics[k])
                history[k].append(avg_val)
            
            print(f"Epoch {epoch+1}: Loss={history['policy_loss'][-1]:.4f}, "
                  f"Accuracy={history['accuracy'][-1]:.4f}")
        
        print("\nPolicy training complete.")
        
        return history
    
    def train(
        self,
        trajectories: List[Trajectory],
        preference_epochs: int = 50,
        policy_epochs: int = 100,
        batch_size: int = 32,
        preference_lr: float = 1e-4,
        policy_lr: float = 1e-4,
        regularization_lambda: float = 0.01,
        max_seq_len: int = 100
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Full training pipeline for PrDT4Rec.
        
        Args:
            trajectories: List of offline trajectories
            preference_epochs: Epochs for preference model
            policy_epochs: Epochs for policy model
            batch_size: Batch size
            preference_lr: Learning rate for preference model
            policy_lr: Learning rate for policy model
            regularization_lambda: L2 regularization for preference model
            max_seq_len: Maximum sequence length
            
        Returns:
            history: Dictionary containing training histories
        """
        # Stage 1: Train preference model
        preference_history = self.train_preference_model(
            trajectories=trajectories,
            num_epochs=preference_epochs,
            batch_size=batch_size,
            learning_rate=preference_lr,
            regularization_lambda=regularization_lambda,
            max_seq_len=max_seq_len
        )
        
        # Stage 2: Train policy model
        policy_history = self.train_policy(
            trajectories=trajectories,
            num_epochs=policy_epochs,
            batch_size=batch_size,
            learning_rate=policy_lr,
            max_seq_len=max_seq_len
        )
        
        return {
            'preference': preference_history,
            'policy': policy_history
        }
    
    def save_models(self, path_prefix: str):
        """Save both models to disk."""
        torch.save(self.preference_model.state_dict(), f"{path_prefix}_preference.pt")
        torch.save(self.policy_model.state_dict(), f"{path_prefix}_policy.pt")
        print(f"Models saved to {path_prefix}_*.pt")
    
    def load_models(self, path_prefix: str):
        """Load both models from disk."""
        self.preference_model.load_state_dict(
            torch.load(f"{path_prefix}_preference.pt", map_location=self.device)
        )
        self.policy_model.load_state_dict(
            torch.load(f"{path_prefix}_policy.pt", map_location=self.device)
        )
        print(f"Models loaded from {path_prefix}_*.pt")