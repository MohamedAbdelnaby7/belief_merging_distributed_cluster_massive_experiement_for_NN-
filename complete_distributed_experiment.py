#!/usr/bin/env python3
"""
Complete Standalone Distributed Belief Merging Experiment
Modified to support multiple grid sizes and agent numbers
Includes all original classes + distributed execution framework
No external dependencies on your original file
FIXED: Consistent seed generation for proper checkpointing
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import json
import time
import pickle
import hashlib
from datetime import datetime
import os
import sys
from pathlib import Path
import logging
from scipy.stats import entropy, pearsonr, spearmanr
from scipy.optimize import minimize
import pandas as pd
import seaborn as sns
from typing import Dict, List, Tuple, Any, Union, Optional
import itertools
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import signal
import fcntl
import tempfile
import shutil
from dataclasses import dataclass, asdict, field
import argparse
import socket
import subprocess

warnings.filterwarnings('ignore')


# ===================================================================
# SEED FIX: Consistent seed generation for proper checkpointing
# ===================================================================

def generate_consistent_seed(grid_size, n_agents, pattern, trial_id):
    """
    Generate a consistent seed that will be the same across Python sessions
    Replaces the problematic hash() function with hashlib for consistency
    
    Args:
        grid_size: (rows, cols) tuple
        n_agents: number of agents
        pattern: movement pattern string
        trial_id: trial identifier
    
    Returns:
        Consistent integer seed
    """
    seed_string = f"{grid_size[0]}x{grid_size[1]}_{n_agents}agents_{pattern}_trial{trial_id}"
    hash_object = hashlib.sha256(seed_string.encode())
    hash_hex = hash_object.hexdigest()
    seed = int(hash_hex[:8], 16) % (2**31 - 1)  # Keep it within int32 range
    return seed


# ===================================================================
# ORIGINAL BELIEF MERGING CLASSES
# ===================================================================

class UnifiedBeliefMergingFramework:
    """
    Framework for different belief merging approaches
    """
    def __init__(self, grid_size=(20, 20), n_agents=4):
        self.grid_size = grid_size
        self.total_states = grid_size[0] * grid_size[1]
        self.n_agents = n_agents
        
    def merge_beliefs_average(self, beliefs, agent_weights=None):
        """Simple averaging of beliefs"""
        if agent_weights is None:
            agent_weights = np.ones(len(beliefs)) / len(beliefs)
        
        merged = np.zeros_like(beliefs[0])
        for i, belief in enumerate(beliefs):
            merged += agent_weights[i] * belief
        
        return merged / np.sum(merged)
    
    def merge_beliefs_kl(self, beliefs, agent_weights=None):
        """KL divergence-based merging"""
        if agent_weights is None:
            agent_weights = np.ones(len(beliefs))
        
        def kl_divergence(p, q):
            p = np.clip(p, 1e-10, 1)
            q = np.clip(q, 1e-10, 1)
            return np.sum(p * np.log(p / q))
        
        def objective(merged_flat):
            merged = merged_flat.reshape(beliefs[0].shape)
            merged = merged / np.sum(merged)
            
            total_divergence = 0
            for i, belief in enumerate(beliefs):
                total_divergence += agent_weights[i] * kl_divergence(belief, merged)
            
            return total_divergence
        
        # Initial guess: weighted average
        initial_guess = self.merge_beliefs_average(beliefs, agent_weights)
        
        # Constraints and bounds
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = [(0, 1) for _ in range(len(initial_guess.flatten()))]
        
        result = minimize(
            objective,
            initial_guess.flatten(),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
        
        if result.success:
            merged = result.x.reshape(initial_guess.shape)
            return merged / np.sum(merged)
        else:
            return initial_guess
    
    def jensen_shannon_divergence(self, p, q):
        """Calculate Jensen-Shannon divergence between two distributions"""
        p = np.clip(p, 1e-10, 1)
        q = np.clip(q, 1e-10, 1)
        m = 0.5 * (p + q)
        return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))


class TargetMovementPolicy:
    """
    Target movement policy - simple patterns without MPC/MDP
    """
    def __init__(self, grid_size, movement_pattern='random'):
        self.grid_size = grid_size
        self.rows, self.cols = grid_size
        self.movement_pattern = movement_pattern
        self.step_count = 0
        
    def get_next_position(self, current_pos, step=None):
        """Get next position based on movement pattern"""
        if step is not None:
            self.step_count = step
            
        r, c = divmod(current_pos, self.cols)
        
        if self.movement_pattern == 'random':
            # Random walk with 0.8 prob of moving, 0.2 of staying
            if np.random.random() < 0.2:
                return current_pos
                
            moves = []
            if r > 0: moves.append(current_pos - self.cols)
            if r < self.rows-1: moves.append(current_pos + self.cols)
            if c > 0: moves.append(current_pos - 1)
            if c < self.cols-1: moves.append(current_pos + 1)
            
            return np.random.choice(moves) if moves else current_pos
            
        elif self.movement_pattern == 'evasive':
            # Try to move away from center
            center_r, center_c = self.rows // 2, self.cols // 2
            
            # Calculate direction away from center
            dr = 1 if r > center_r else -1 if r < center_r else 0
            dc = 1 if c > center_c else -1 if c < center_c else 0
            
            # Preferred moves
            preferred = []
            if 0 <= r + dr < self.rows:
                preferred.append(current_pos + dr * self.cols)
            if 0 <= c + dc < self.cols:
                preferred.append(current_pos + dc)
                
            if preferred and np.random.random() < 0.7:
                return np.random.choice(preferred)
            else:
                # Random move
                moves = []
                if r > 0: moves.append(current_pos - self.cols)
                if r < self.rows-1: moves.append(current_pos + self.cols)
                if c > 0: moves.append(current_pos - 1)
                if c < self.cols-1: moves.append(current_pos + 1)
                return np.random.choice(moves) if moves else current_pos
                
        elif self.movement_pattern == 'patrol':
            # Circular patrol pattern
            corners = [
                0,  # top-left
                self.cols - 1,  # top-right
                (self.rows - 1) * self.cols + self.cols - 1,  # bottom-right
                (self.rows - 1) * self.cols  # bottom-left
            ]
            
            # Find closest corner
            min_dist = float('inf')
            target_corner = corners[0]
            for corner in corners:
                corner_r, corner_c = divmod(corner, self.cols)
                dist = abs(r - corner_r) + abs(c - corner_c)
                if dist < min_dist and dist > 0:  # Don't stay at current corner
                    min_dist = dist
                    target_corner = corner
                    
            # Move towards target corner
            target_r, target_c = divmod(target_corner, self.cols)
            dr = 1 if target_r > r else -1 if target_r < r else 0
            dc = 1 if target_c > c else -1 if target_c < c else 0
            
            new_r = r + dr
            new_c = c + dc
            
            if 0 <= new_r < self.rows and 0 <= new_c < self.cols:
                return new_r * self.cols + new_c
                
        return current_pos


class MultiAgentMPC:
    """
    Correct MPC implementation for multi-agent search
    """
    def __init__(self, grid_size, n_agents, horizon=2, alpha=0.1, beta=0.2):
        self.grid_size = grid_size
        self.rows, self.cols = grid_size
        self.n_agents = n_agents
        self.horizon = horizon
        self.alpha = alpha  # False positive rate
        self.beta = beta    # False negative rate
        self.n_states = grid_size[0] * grid_size[1]
        
    def get_joint_action(self, beliefs: Union[np.ndarray, List[np.ndarray]], 
                        agent_positions: List[int], 
                        fast_mode: bool = False) -> List[int]:
        """
        Get optimal joint action for all agents
        TRUE MPC implementation (computationally intensive)
        """
        if fast_mode:
            return self._get_greedy_joint_action(beliefs, agent_positions)
        
        best_joint_action = agent_positions.copy()
        best_value = -float('inf')
        
        # Generate candidate actions for each agent
        candidate_actions = []
        for pos in agent_positions:
            neighbors = self._get_neighbors(pos)
            candidate_actions.append(neighbors)
        
        # Limit search if too many combinations
        max_combinations = 625  # 5^4 for 4 agents
        all_combinations = list(itertools.product(*candidate_actions))
        
        if len(all_combinations) > max_combinations:
            # Sample a subset for large action spaces
            sampled_combinations = [all_combinations[i] for i in 
                                   np.random.choice(len(all_combinations), max_combinations, replace=False)]
        else:
            sampled_combinations = all_combinations
        
        # Evaluate sampled joint actions (TRUE MPC - computationally intensive)
        for joint_action in sampled_combinations:
            # Evaluate this joint action over the horizon
            if isinstance(beliefs, np.ndarray):  # Shared belief
                total_value = self._evaluate_shared_belief(
                    beliefs.copy(), list(joint_action), self.horizon
                )
            else:  # Independent beliefs
                total_value = self._evaluate_independent_beliefs(
                    [b.copy() for b in beliefs], list(joint_action), self.horizon
                )
            
            if total_value > best_value:
                best_value = total_value
                best_joint_action = list(joint_action)
        
        return best_joint_action
    
    def _get_greedy_joint_action(self, beliefs: Union[np.ndarray, List[np.ndarray]], 
                                 agent_positions: List[int]) -> List[int]:
        """
        Fast greedy approximation: each agent moves to highest belief neighbor
        """
        joint_action = []
        
        if isinstance(beliefs, np.ndarray):  # Shared belief
            for pos in agent_positions:
                neighbors = self._get_neighbors(pos)
                best_pos = max(neighbors, key=lambda n: beliefs[n])
                joint_action.append(best_pos)
        else:  # Independent beliefs
            for i, pos in enumerate(agent_positions):
                neighbors = self._get_neighbors(pos)
                best_pos = max(neighbors, key=lambda n: beliefs[i][n])
                joint_action.append(best_pos)
        
        return joint_action
    
    def _evaluate_shared_belief(self, belief: np.ndarray, 
                               joint_action: List[int], 
                               horizon: int) -> float:
        """
        Evaluate joint action with shared belief over horizon
        """
        total_value = 0.0
        current_belief = belief.copy()
        
        for h in range(horizon):
            # Simulate joint observations
            simulated_obs = self._simulate_joint_observation(joint_action, current_belief)
            
            # Update shared belief with all observations
            current_belief = self._update_belief_joint(
                current_belief, joint_action, simulated_obs
            )
            
            # Calculate objective (negative entropy for information gain)
            total_value += -entropy(current_belief)
            
            # For future horizons, should consider future movements
            # Simplified: agents stay in place
        
        return total_value
    
    def _evaluate_independent_beliefs(self, beliefs: List[np.ndarray], 
                                    joint_action: List[int], 
                                    horizon: int) -> float:
        """
        Evaluate joint action with independent beliefs over horizon
        """
        total_value = 0.0
        current_beliefs = [b.copy() for b in beliefs]
        
        for h in range(horizon):
            # Each agent has its own belief and makes its own observation
            for i, (action, belief) in enumerate(zip(joint_action, current_beliefs)):
                # Simulate observation for this agent
                obs = self._simulate_single_observation(action, belief)
                
                # Update this agent's belief
                current_beliefs[i] = self._update_belief_single(
                    belief, action, obs
                )
            
            # Calculate total objective across all agents
            for belief in current_beliefs:
                total_value += -entropy(belief)
        
        return total_value
    
    def _simulate_joint_observation(self, joint_positions: List[int], 
                               belief: np.ndarray) -> List[int]:
        """
        Simulate observations for all agents at their positions
        """
        # Normalize belief to ensure it sums to 1
        belief_normalized = belief / np.sum(belief)
        
        # Sample target position from belief (for simulation)
        target_pos = np.random.choice(self.n_states, p=belief_normalized)
        
        observations = []
        for pos in joint_positions:
            if pos == target_pos:
                # True positive with probability (1-beta)
                obs = 1 if np.random.random() < (1 - self.beta) else 0
            else:
                # False positive with probability alpha
                obs = 1 if np.random.random() < self.alpha else 0
            observations.append(obs)
        
        return observations
    
    def _simulate_single_observation(self, position: int, belief: np.ndarray) -> int:
        """
        Simulate observation for single agent
        """
        # Sample target position from belief
        target_pos = np.random.choice(self.n_states, p=belief)
        
        if position == target_pos:
            return 1 if np.random.random() < (1 - self.beta) else 0
        else:
            return 1 if np.random.random() < self.alpha else 0
    
    def _update_belief_joint(self, belief: np.ndarray, 
                           positions: List[int], 
                           observations: List[int]) -> np.ndarray:
        """
        Update belief with joint observations from all agents
        """
        # Likelihood for each state
        likelihood = np.ones(self.n_states)
        
        for pos, obs in zip(positions, observations):
            if obs == 1:  # Detection
                likelihood[pos] *= (1 - self.beta)  # True positive at position
                # False positive elsewhere
                mask = np.ones(self.n_states, dtype=bool)
                mask[pos] = False
                likelihood[mask] *= self.alpha
            else:  # No detection
                likelihood[pos] *= self.beta  # False negative at position
                # True negative elsewhere
                mask = np.ones(self.n_states, dtype=bool)
                mask[pos] = False
                likelihood[mask] *= (1 - self.alpha)
        
        # Bayesian update
        posterior = belief * likelihood
        return posterior / (np.sum(posterior) + 1e-10)
    
    def _update_belief_single(self, belief: np.ndarray, 
                            position: int, 
                            observation: int) -> np.ndarray:
        """
        Update single agent's belief
        """
        likelihood = np.ones(self.n_states)
        
        if observation == 1:  # Detection
            likelihood[position] = 1 - self.beta
            likelihood[np.arange(self.n_states) != position] = self.alpha
        else:  # No detection
            likelihood[position] = self.beta
            likelihood[np.arange(self.n_states) != position] = 1 - self.alpha
        
        # Bayesian update
        posterior = belief * likelihood
        return posterior / (np.sum(posterior) + 1e-10)
    
    def _get_neighbors(self, position: int) -> List[int]:
        """Get valid neighboring positions including current position"""
        r, c = divmod(position, self.cols)
        neighbors = [position]  # Include current position
        
        if r > 0:
            neighbors.append(position - self.cols)
        if r < self.rows - 1:
            neighbors.append(position + self.cols)
        if c > 0:
            neighbors.append(position - 1)
        if c < self.cols - 1:
            neighbors.append(position + 1)
            
        return neighbors


class ControlledMergingExperiment:
    """
    Main experiment class with proper MPC implementation
    """
    
    def __init__(self, grid_size=(20, 20), n_agents=4, alpha=0.1, beta=0.2, horizon=2):
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.alpha = alpha  # False positive rate
        self.beta = beta   # False negative rate
        self.merger = UnifiedBeliefMergingFramework(grid_size, n_agents)
        self.mpc = MultiAgentMPC(grid_size, n_agents, horizon, alpha, beta)
        
    def _run_single_experiment(self, trial_config, merge_interval, max_steps, fast_mode=False):
        """Run a single experiment with specified merge interval"""
        # Special case for full communication
        if merge_interval == 0:
            return self._run_centralized_full_communication(trial_config, max_steps, fast_mode)
        
        # For other strategies
        start_time = time.time()
        
        # Initialize agents with independent beliefs
        agent_beliefs = [
            np.ones(self.grid_size[0] * self.grid_size[1]) / (self.grid_size[0] * self.grid_size[1])
            for _ in range(self.n_agents)
        ]
        
        agent_positions = trial_config['initial_positions'].copy()
        agent_trajectories = [[pos] for pos in agent_positions]
        
        # Tracking variables
        target_found = False
        first_discovery_step = max_steps
        discovery_count = 0
        entropy_history = []
        divergence_history = []
        merge_events = []
        
        # Run simulation
        for step in range(max_steps):
            # Target position for this step
            target_pos = trial_config['target_trajectory'][step]
            
            # Calculate and store metrics BEFORE actions
            entropies = [entropy(b) for b in agent_beliefs]
            entropy_history.append({
                'mean': np.mean(entropies),
                'std': np.std(entropies),
                'max': np.max(entropies),
                'min': np.min(entropies)
            })
            
            # Calculate divergence between agents
            divergences = []
            for i in range(len(agent_beliefs)):
                for j in range(i+1, len(agent_beliefs)):
                    div = self.merger.jensen_shannon_divergence(agent_beliefs[i], agent_beliefs[j])
                    divergences.append(div)
            
            divergence_history.append({
                'mean': np.mean(divergences) if divergences else 0,
                'std': np.std(divergences) if divergences else 0,
                'max': np.max(divergences) if divergences else 0
            })
            
            # Check if it's time to merge
            if merge_interval != float('inf') and step > 0 and step % merge_interval == 0:
                # Save beliefs BEFORE merge
                beliefs_before = [b.copy() for b in agent_beliefs]
                
                # Perform belief merging
                merged_belief = self.merger.merge_beliefs_kl(agent_beliefs)
                
                # Update all agents with merged belief
                for i in range(self.n_agents):
                    agent_beliefs[i] = merged_belief.copy()
                
                # Calculate entropy change
                entropy_before = np.mean([entropy(b) for b in beliefs_before])
                entropy_after = entropy(merged_belief)
                
                merge_events.append({
                    'step': step,
                    'entropy_before': entropy_before,
                    'entropy_after': entropy_after,
                    'entropy_reduction': entropy_before - entropy_after
                })
            
            # Get joint action using MPC (TRUE MPC if fast_mode=False)
            joint_action = self.mpc.get_joint_action(
                agent_beliefs,
                agent_positions,
                fast_mode=fast_mode
            )
            
            # Make observations BEFORE moving
            for i, pos in enumerate(agent_positions):
                obs_rand = trial_config['observation_randoms'][step, i]
                
                if pos == target_pos:
                    observation = 1 if obs_rand > self.beta else 0
                    if observation == 1 and not target_found:
                        target_found = True
                        first_discovery_step = step
                    if observation == 1:
                        discovery_count += 1
                else:
                    observation = 1 if obs_rand < self.alpha else 0
                
                # Update individual belief
                agent_beliefs[i] = self.mpc._update_belief_single(
                    agent_beliefs[i], pos, observation
                )
            
            # Execute joint action
            agent_positions = joint_action
            for i, new_pos in enumerate(joint_action):
                agent_trajectories[i].append(new_pos)
        
        # Final merge for no_comm strategy
        if merge_interval == float('inf'):
            final_beliefs = agent_beliefs
            final_merged = self.merger.merge_beliefs_kl(final_beliefs)
        else:
            final_beliefs = agent_beliefs
            final_merged = np.mean(final_beliefs, axis=0)
            final_merged = final_merged / np.sum(final_merged)
        
        # Calculate final metrics
        final_target_pos = trial_config['target_trajectory'][-1]
        
        # Performance metrics
        prob_at_true_target = final_merged[final_target_pos]
        
        # Find position with highest belief
        predicted_pos = np.argmax(final_merged)
        pred_r, pred_c = divmod(predicted_pos, self.grid_size[1])
        true_r, true_c = divmod(final_target_pos, self.grid_size[1])
        prediction_error = np.sqrt((pred_r - true_r)**2 + (pred_c - true_c)**2)
        
        elapsed_time = time.time() - start_time
        
        return {
            'target_found': target_found,
            'first_discovery_step': first_discovery_step,
            'discovery_count': discovery_count,
            'elapsed_time': elapsed_time,
            'final_merged_belief': final_merged,
            'final_entropy': entropy(final_merged),
            'entropy_history': entropy_history,
            'divergence_history': divergence_history,
            'merge_events': merge_events,
            'total_merges': len(merge_events),
            'prob_at_true_target': prob_at_true_target,
            'prediction_error': prediction_error,
            'agent_trajectories': agent_trajectories
        }
    
    def _run_centralized_full_communication(self, trial_config, max_steps, fast_mode=False):
        """Run true full communication - single shared belief, coordinated MPC"""
        start_time = time.time()
        
        # Single shared belief for all agents
        shared_belief = np.ones(self.grid_size[0] * self.grid_size[1]) / (self.grid_size[0] * self.grid_size[1])
        
        # Agent positions
        agent_positions = trial_config['initial_positions'].copy()
        agent_trajectories = [[pos] for pos in agent_positions]
        
        # Tracking variables
        target_found = False
        first_discovery_step = max_steps
        discovery_count = 0
        entropy_history = []
        
        # Add artificial communication overhead
        COMM_OVERHEAD_PER_STEP = 0.001 * self.n_agents * (self.n_agents - 1) / 2
        
        # Run simulation
        for step in range(max_steps):
            # Target position for this step
            target_pos = trial_config['target_trajectory'][step]
            
            # Calculate and store metrics
            entropy_val = entropy(shared_belief)
            entropy_history.append({
                'mean': entropy_val,
                'std': 0,  # No variance - single belief
                'max': entropy_val,
                'min': entropy_val
            })
            
            # Get joint action using MPC with shared belief (TRUE MPC if fast_mode=False)
            joint_action = self.mpc.get_joint_action(
                shared_belief, 
                agent_positions, 
                fast_mode=fast_mode
            )
            
            # ALL agents make observations BEFORE moving
            for i, pos in enumerate(agent_positions):
                obs_rand = trial_config['observation_randoms'][step, i]
                
                if pos == target_pos:
                    observation = 1 if obs_rand > self.beta else 0
                    if observation == 1 and not target_found:
                        target_found = True
                        first_discovery_step = step
                    if observation == 1:
                        discovery_count += 1
                else:
                    observation = 1 if obs_rand < self.alpha else 0
                
                # Update SHARED belief
                shared_belief = self.mpc._update_belief_single(
                    shared_belief, pos, observation
                )
            
            # Execute joint action
            agent_positions = joint_action
            for i, new_pos in enumerate(joint_action):
                agent_trajectories[i].append(new_pos)
            
            # Add communication overhead
            time.sleep(COMM_OVERHEAD_PER_STEP)
        
        # Final metrics
        final_target_pos = trial_config['target_trajectory'][-1]
        
        # Performance metrics
        prob_at_true_target = shared_belief[final_target_pos]
        
        # Find position with highest belief
        predicted_pos = np.argmax(shared_belief)
        pred_r, pred_c = divmod(predicted_pos, self.grid_size[1])
        true_r, true_c = divmod(final_target_pos, self.grid_size[1])
        prediction_error = np.sqrt((pred_r - true_r)**2 + (pred_c - true_c)**2)
        
        elapsed_time = time.time() - start_time
        
        return {
            'target_found': target_found,
            'first_discovery_step': first_discovery_step,
            'discovery_count': discovery_count,
            'elapsed_time': elapsed_time,
            'final_merged_belief': shared_belief,
            'final_entropy': entropy(shared_belief),
            'entropy_history': entropy_history,
            'divergence_history': [{'mean': 0, 'std': 0, 'max': 0} for _ in range(max_steps)],
            'merge_events': [],  # No merge events - always together
            'total_merges': 0,
            'prob_at_true_target': prob_at_true_target,
            'prediction_error': prediction_error,
            'agent_trajectories': agent_trajectories
        }


# ===================================================================
# DISTRIBUTED EXECUTION FRAMEWORK 
# ===================================================================

@dataclass
class ExperimentConfig:
    """Configuration for the experiment - now supports multiple grid sizes and agent numbers"""
    grid_sizes: List[Tuple[int, int]] = field(default_factory=lambda: [(20, 20)])
    n_agents_list: List[int] = field(default_factory=lambda: [4])
    alpha: float = 0.1  # False positive rate
    beta: float = 0.2   # False negative rate
    horizon: int = 2    # MPC horizon
    n_trials: int = 30
    max_steps: int = 1000
    merge_intervals: List[Union[int, float]] = None
    target_patterns: List[str] = None
    fast_mode: bool = False  # TRUE MPC by default
    
    def __post_init__(self):
        if self.merge_intervals is None:
            self.merge_intervals = [0, 10, 25, 50, 100, 200, 500, float('inf')]
        if self.target_patterns is None:
            self.target_patterns = ['random', 'evasive', 'patrol']
    
    def to_dict(self):
        d = asdict(self)
        # Convert tuples to lists for JSON serialization
        d['grid_sizes'] = [[r, c] for r, c in self.grid_sizes]
        return d
    
    @classmethod
    def from_dict(cls, d):
        # Convert lists back to tuples for grid sizes
        if 'grid_sizes' in d:
            d['grid_sizes'] = [(r, c) for r, c in d['grid_sizes']]
        return cls(**d)


@dataclass
class TrialTask:
    """Individual trial task for parallel execution - now includes grid size and n_agents"""
    grid_size: Tuple[int, int]
    n_agents: int
    pattern: str
    trial_id: int
    merge_interval: Union[int, float]
    config: ExperimentConfig
    trial_seed: int
    checkpoint_dir: str
    
    def get_task_id(self):
        """Generate unique task ID including grid size and n_agents"""
        grid_str = f"{self.grid_size[0]}x{self.grid_size[1]}"
        return f"grid{grid_str}_agents{self.n_agents}_{self.pattern}_trial{self.trial_id}_interval{self.merge_interval}_seed{self.trial_seed}"
    
    def get_checkpoint_path(self):
        """Get checkpoint file path for this task"""
        return os.path.join(self.checkpoint_dir, f"{self.get_task_id()}.pkl")


class DistributedExperimentManager:
    """Manages distributed execution with checkpointing - now supports multiple configurations"""
    
    def __init__(self, config: ExperimentConfig, checkpoint_dir: str = "checkpoints", 
                 results_dir: str = "results", max_workers: int = None):
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.results_dir = Path(results_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Determine optimal number of workers
        if max_workers is None:
            # Use 80% of available CPUs, but cap at 500 to avoid overwhelming
            max_workers = min(int(psutil.cpu_count() * 0.8), 500)
        self.max_workers = max_workers
        
        # Setup logging
        self.setup_logging()
        
        # Progress tracking
        self.total_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = self.results_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Main log file
        log_file = log_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Experiment started on {socket.gethostname()}")
        self.logger.info(f"Using {self.max_workers} workers")
        self.logger.info(f"Config: {self.config}")
    
    def generate_all_tasks(self) -> List[TrialTask]:
        """Generate all trial tasks for multiple grid sizes and agent numbers"""
        tasks = []
        
        for grid_size in self.config.grid_sizes:
            for n_agents in self.config.n_agents_list:
                for pattern in self.config.target_patterns:
                    for trial_id in range(self.config.n_trials):
                        # FIXED: Use consistent seed generation instead of hash()
                        trial_seed = generate_consistent_seed(grid_size, n_agents, pattern, trial_id)
                        
                        for merge_interval in self.config.merge_intervals:
                            task = TrialTask(
                                grid_size=grid_size,
                                n_agents=n_agents,
                                pattern=pattern,
                                trial_id=trial_id,
                                merge_interval=merge_interval,
                                config=self.config,
                                trial_seed=trial_seed,
                                checkpoint_dir=str(self.checkpoint_dir)
                            )
                            tasks.append(task)
        
        self.total_tasks = len(tasks)
        self.logger.info(f"Generated {self.total_tasks} total tasks")
        self.logger.info(f"Grid sizes: {self.config.grid_sizes}")
        self.logger.info(f"Agent numbers: {self.config.n_agents_list}")
        return tasks
    
    def filter_incomplete_tasks(self, tasks: List[TrialTask]) -> List[TrialTask]:
        """Filter out already completed tasks"""
        incomplete_tasks = []
        
        for task in tasks:
            checkpoint_path = task.get_checkpoint_path()
            if not os.path.exists(checkpoint_path):
                incomplete_tasks.append(task)
            else:
                # Verify checkpoint integrity
                try:
                    with open(checkpoint_path, 'rb') as f:
                        result = pickle.load(f)
                    if self.validate_result(result):
                        self.completed_tasks += 1
                        continue
                    else:
                        self.logger.warning(f"Invalid checkpoint found: {checkpoint_path}")
                        os.remove(checkpoint_path)
                except Exception as e:
                    self.logger.warning(f"Corrupted checkpoint: {checkpoint_path}, error: {e}")
                    if os.path.exists(checkpoint_path):
                        os.remove(checkpoint_path)
                
                incomplete_tasks.append(task)
        
        self.logger.info(f"Found {self.completed_tasks} completed tasks")
        self.logger.info(f"Remaining tasks: {len(incomplete_tasks)}")
        return incomplete_tasks
    
    def validate_result(self, result: Dict) -> bool:
        """Validate that a result contains all required fields"""
        required_fields = [
            'target_found', 'first_discovery_step', 'discovery_count',
            'elapsed_time', 'final_merged_belief', 'final_entropy',
            'entropy_history', 'divergence_history', 'prediction_error'
        ]
        
        return all(field in result for field in required_fields)
    
    def run_distributed_experiment(self):
        """Run the complete distributed experiment"""
        self.logger.info("Starting distributed experiment")
        start_time = time.time()
        
        # Generate all tasks
        all_tasks = self.generate_all_tasks()
        
        # Filter out completed tasks
        remaining_tasks = self.filter_incomplete_tasks(all_tasks)
        
        if not remaining_tasks:
            self.logger.info("All tasks already completed!")
            return self.collect_results()
        
        # Setup signal handlers for graceful shutdown
        self.setup_signal_handlers()
        
        # Run tasks in parallel
        self.logger.info(f"Starting parallel execution with {self.max_workers} workers")
        
        try:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(run_single_trial_task, task): task 
                    for task in remaining_tasks
                }
                
                # Process completed tasks
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    
                    try:
                        result = future.result()
                        if result is not None:
                            self.completed_tasks += 1
                            self.logger.info(
                                f"Completed {task.get_task_id()} "
                                f"({self.completed_tasks}/{self.total_tasks})"
                            )
                        else:
                            self.failed_tasks += 1
                            self.logger.error(f"Failed: {task.get_task_id()}")
                    
                    except Exception as e:
                        self.failed_tasks += 1
                        self.logger.error(f"Task {task.get_task_id()} failed: {e}")
                    
                    # Progress update
                    if self.completed_tasks % 50 == 0:
                        elapsed = time.time() - start_time
                        rate = self.completed_tasks / elapsed
                        eta = (self.total_tasks - self.completed_tasks) / rate if rate > 0 else 0
                        self.logger.info(
                            f"Progress: {self.completed_tasks}/{self.total_tasks} "
                            f"({100*self.completed_tasks/self.total_tasks:.1f}%), "
                            f"Rate: {rate:.2f} tasks/sec, ETA: {eta/3600:.1f} hours"
                        )
        
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal, shutting down gracefully...")
            return None
        
        total_time = time.time() - start_time
        self.logger.info(
            f"Experiment completed in {total_time/3600:.2f} hours. "
            f"Completed: {self.completed_tasks}, Failed: {self.failed_tasks}"
        )
        
        # Collect and analyze results
        return self.collect_results()
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def collect_results(self):
        """Collect all results from checkpoints - now organized by grid size and n_agents"""
        self.logger.info("Collecting results from checkpoints...")
        
        all_results = {}
        
        for grid_size in self.config.grid_sizes:
            grid_key = f"{grid_size[0]}x{grid_size[1]}"
            all_results[grid_key] = {}
            
            for n_agents in self.config.n_agents_list:
                agent_key = f"{n_agents}_agents"
                all_results[grid_key][agent_key] = {}
                
                for pattern in self.config.target_patterns:
                    pattern_results = {}
                    
                    for merge_interval in self.config.merge_intervals:
                        interval_results = []
                        
                        for trial_id in range(self.config.n_trials):
                            # FIXED: Use consistent seed generation
                            trial_seed = generate_consistent_seed(grid_size, n_agents, pattern, trial_id)
                            
                            task = TrialTask(
                                grid_size=grid_size,
                                n_agents=n_agents,
                                pattern=pattern,
                                trial_id=trial_id,
                                merge_interval=merge_interval,
                                config=self.config,
                                trial_seed=trial_seed,
                                checkpoint_dir=str(self.checkpoint_dir)
                            )
                            
                            checkpoint_path = task.get_checkpoint_path()
                            
                            if os.path.exists(checkpoint_path):
                                try:
                                    with open(checkpoint_path, 'rb') as f:
                                        result = pickle.load(f)
                                    interval_results.append(result)
                                except Exception as e:
                                    self.logger.error(f"Failed to load {checkpoint_path}: {e}")
                        
                        # Store results with proper key naming
                        if merge_interval == 0:
                            key = 'full_comm'
                        elif merge_interval == float('inf'):
                            key = 'no_comm'
                        else:
                            key = f'interval_{merge_interval}'
                        
                        pattern_results[key] = interval_results
                    
                    all_results[grid_key][agent_key][pattern] = pattern_results
        
        # Save consolidated results
        consolidated_path = self.results_dir / f"consolidated_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(consolidated_path, 'wb') as f:
            pickle.dump(all_results, f)
        
        self.logger.info(f"Consolidated results saved to {consolidated_path}")
        return all_results


def run_single_trial_task(task: TrialTask) -> Optional[Dict]:
    """Run a single trial task - designed for parallel execution"""
    
    # Check if already completed
    checkpoint_path = task.get_checkpoint_path()
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'rb') as f:
                result = pickle.load(f)
            return result
        except:
            # Corrupted checkpoint, will regenerate
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
    
    try:
        # Create experiment instance with specific grid size and n_agents
        experiment = ControlledMergingExperiment(
            grid_size=task.grid_size,
            n_agents=task.n_agents,
            alpha=task.config.alpha,
            beta=task.config.beta,
            horizon=task.config.horizon
        )
        
        # Generate trial configuration
        np.random.seed(task.trial_seed)
        
        # Generate target trajectory
        target_policy = TargetMovementPolicy(task.grid_size, task.pattern)
        initial_target = np.random.randint(0, task.grid_size[0] * task.grid_size[1])
        
        target_trajectory = [initial_target]
        current_pos = initial_target
        for step in range(task.config.max_steps):
            current_pos = target_policy.get_next_position(current_pos, step)
            target_trajectory.append(current_pos)
        
        # Generate initial agent positions
        total_states = task.grid_size[0] * task.grid_size[1]
        available_positions = list(range(total_states))
        if initial_target in available_positions:
            available_positions.remove(initial_target)
        
        # Handle case where we have more agents than available positions
        if task.n_agents > len(available_positions):
            raise ValueError(f"Cannot place {task.n_agents} agents in grid of size {task.grid_size} with target at {initial_target}")
        
        initial_positions = np.random.choice(
            available_positions, 
            task.n_agents, 
            replace=False
        ).tolist()
        
        # Pre-generate observation random numbers
        observation_randoms = np.random.random((task.config.max_steps, task.n_agents))
        
        # Create trial configuration
        trial_config = {
            'trial_id': task.trial_id,
            'seed': task.trial_seed,
            'target_trajectory': target_trajectory,
            'initial_positions': initial_positions,
            'observation_randoms': observation_randoms,
            'target_pattern': task.pattern,
            'grid_size': task.grid_size,
            'n_agents': task.n_agents
        }
        
        # Run the experiment with TRUE MPC (unless fast_mode specified in config)
        result = experiment._run_single_experiment(
            trial_config,
            task.merge_interval,
            task.config.max_steps,
            task.config.fast_mode  # FALSE by default for TRUE MPC
        )
        
        # Add metadata
        result['task_metadata'] = {
            'grid_size': task.grid_size,
            'n_agents': task.n_agents,
            'pattern': task.pattern,
            'trial_id': task.trial_id,
            'merge_interval': task.merge_interval,
            'trial_seed': task.trial_seed,
            'hostname': socket.gethostname(),
            'pid': os.getpid(),
            'completion_time': datetime.now().isoformat()
        }
        
        # Atomic save to checkpoint
        save_result_atomic(result, checkpoint_path)
        
        return result
        
    except Exception as e:
        # Log error but don't crash the worker
        error_msg = f"Task {task.get_task_id()} failed: {str(e)}"
        print(f"ERROR: {error_msg}")
        
        # Save error info
        error_path = checkpoint_path.replace('.pkl', '_ERROR.txt')
        with open(error_path, 'w') as f:
            f.write(f"{datetime.now().isoformat()}: {error_msg}\n")
            f.write(f"Exception type: {type(e).__name__}\n")
            f.write(f"Exception details: {str(e)}\n")
            import traceback
            f.write(f"Traceback:\n{traceback.format_exc()}\n")
        
        return None


def save_result_atomic(result: Dict, filepath: str):
    """Atomically save result to avoid corruption"""
    temp_path = filepath + '.tmp'
    try:
        with open(temp_path, 'wb') as f:
            pickle.dump(result, f)
        
        # Atomic move
        os.rename(temp_path, filepath)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Complete Distributed Belief Merging Experiment')
    parser.add_argument('--config-file', type=str, help='JSON config file path')
    parser.add_argument('--max-workers', type=int, help='Maximum number of workers')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--results-dir', type=str, default='results', help='Results directory')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
        config = ExperimentConfig.from_dict(config_dict)
    else:
        # Default configuration with multiple grid sizes and agent numbers
        config = ExperimentConfig(
            grid_sizes=[(10, 10), (20, 20), (30, 30)],  # Multiple grid sizes
            n_agents_list=[2, 3, 4],  # Multiple agent numbers
            alpha=0.1,
            beta=0.2,
            horizon=3,  # Full MPC horizon
            n_trials=50,
            max_steps=1000,
            merge_intervals=[0, 10, 25, 50, 100, 200, 500, float('inf')],
            target_patterns=['random', 'evasive', 'patrol'],
            fast_mode=False  # TRUE MPC for computational accuracy
        )
    
    print("="*80)
    print("COMPLETE DISTRIBUTED BELIEF MERGING EXPERIMENT")
    print("="*80)
    print(f"Configuration: {config}")
    print(f"Grid sizes: {config.grid_sizes}")
    print(f"Agent numbers: {config.n_agents_list}")
    print(f"TRUE MPC Mode: {not config.fast_mode}")
    
    # Create and run experiment manager
    manager = DistributedExperimentManager(
        config=config,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        max_workers=args.max_workers
    )
    
    # Run the experiment
    results = manager.run_distributed_experiment()
    
    if results is not None:
        print("\nExperiment completed successfully!")
        print(f"Results saved in: {manager.results_dir}")
    else:
        print("\nExperiment was interrupted.")


if __name__ == "__main__":
    main()