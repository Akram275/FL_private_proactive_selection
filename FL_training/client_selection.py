"""
Client Selection Strategies for Federated Learning

This module implements in-training client selection approaches:
- UCB-CS: Bandit-based client selection using Upper Confidence Bound (Cho et al., 2020)
- ThresholdCS: Threshold-based participation with Ornstein-Uhlenbeck estimation (Ribero & Vikalo, 2020)

References:
- UCB-CS: https://arxiv.org/abs/2012.08009
- Threshold-based: https://arxiv.org/abs/2007.15197
"""

import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import math


class ClientSelector(ABC):
    """Abstract base class for client selection strategies."""
    
    def __init__(self, n_clients: int, participation_rate: float = 1.0):
        """
        Args:
            n_clients: Total number of clients
            participation_rate: Fraction of clients to select per round (0, 1]
        """
        self.n_clients = n_clients
        self.participation_rate = participation_rate
        self.k = max(1, int(n_clients * participation_rate))  # Number of clients to select
        self.round_num = 0
        
    @abstractmethod
    def select_clients(self, **kwargs) -> List[int]:
        """Select clients for the current round."""
        pass
    
    @abstractmethod
    def update(self, selected_clients: List[int], **kwargs) -> None:
        """Update selection strategy based on round results."""
        pass
    
    def increment_round(self):
        self.round_num += 1


class RandomSelector(ClientSelector):
    """Uniform random client selection (baseline)."""
    
    def __init__(self, n_clients: int, participation_rate: float = 1.0):
        super().__init__(n_clients, participation_rate)
        
    def select_clients(self, **kwargs) -> List[int]:
        """Randomly select k clients."""
        return list(np.random.choice(self.n_clients, self.k, replace=False))
    
    def update(self, selected_clients: List[int], client_losses: Dict[int, float] = None,
               **kwargs) -> None:
        """No update needed for random selection."""
        self.increment_round()


class UCBClientSelector(ClientSelector):
    """
    UCB-CS: Bandit-based Client Selection (Cho et al., 2020)
    
    Uses Upper Confidence Bound to select clients with potentially high local losses.
    The UCB score balances exploitation (high estimated loss) with exploration
    (uncertainty from fewer selections).
    
    UCB_i(t) = μ_i(t) + c * sqrt(log(t) / N_i(t))
    
    where:
    - μ_i(t): Estimated local loss for client i
    - N_i(t): Number of times client i was selected
    - c: Exploration parameter
    """
    
    def __init__(self, n_clients: int, participation_rate: float = 1.0, 
                 exploration_param: float = 1.0, loss_decay: float = 0.9):
        """
        Args:
            n_clients: Total number of clients
            participation_rate: Fraction of clients to select
            exploration_param: UCB exploration coefficient (c)
            loss_decay: Exponential decay for loss estimates
        """
        super().__init__(n_clients, participation_rate)
        self.c = exploration_param
        self.loss_decay = loss_decay
        
        # Initialize estimates
        self.estimated_losses = np.ones(n_clients) * float('inf')  # Optimistic initialization
        self.selection_counts = np.zeros(n_clients)
        self.last_losses = np.zeros(n_clients)
        
    def select_clients(self, **kwargs) -> List[int]:
        """Select clients with highest UCB scores."""
        t = self.round_num + 1  # Avoid log(0)
        
        # Compute UCB scores
        ucb_scores = np.zeros(self.n_clients)
        for i in range(self.n_clients):
            if self.selection_counts[i] == 0:
                # Force exploration of unselected clients
                ucb_scores[i] = float('inf')
            else:
                exploration_bonus = self.c * np.sqrt(np.log(t) / self.selection_counts[i])
                ucb_scores[i] = self.estimated_losses[i] + exploration_bonus
        
        # Select top-k clients by UCB score
        selected = np.argsort(ucb_scores)[-self.k:]
        return list(selected)
    
    def update(self, selected_clients: List[int], client_losses: Dict[int, float] = None, 
               **kwargs) -> None:
        """
        Update loss estimates based on observed losses.
        
        Args:
            selected_clients: List of selected client indices
            client_losses: Dict mapping client_id -> local loss after training
        """
        if client_losses is None:
            client_losses = {}
            
        for client_id in selected_clients:
            self.selection_counts[client_id] += 1
            
            if client_id in client_losses:
                loss = client_losses[client_id]
                # Exponential moving average of losses
                if self.estimated_losses[client_id] == float('inf'):
                    self.estimated_losses[client_id] = loss
                else:
                    self.estimated_losses[client_id] = (
                        self.loss_decay * self.estimated_losses[client_id] + 
                        (1 - self.loss_decay) * loss
                    )
                self.last_losses[client_id] = loss
        
        self.increment_round()
    
    def get_stats(self) -> Dict:
        """Return selection statistics."""
        return {
            'selection_counts': self.selection_counts.copy(),
            'estimated_losses': self.estimated_losses.copy(),
            'last_losses': self.last_losses.copy()
        }


class ThresholdClientSelector(ClientSelector):
    """
    Threshold-based client selection inspired by Ribero & Vikalo (2020).
    
    Simplified version using loss as a proxy for importance.
    Uses O-U process to estimate losses for non-selected clients.
    
    Selection criterion: loss_i > τ (threshold based on percentile)
    Higher loss = more potential for improvement = should be selected.
    """
    
    def __init__(self, n_clients: int, participation_rate: float = 1.0,
                 threshold_percentile: float = 50.0, theta: float = 0.5):
        """
        Args:
            n_clients: Total number of clients
            participation_rate: Max fraction of clients to select (used for budget)
            threshold_percentile: Percentile of losses above which to select
            theta: O-U mean reversion parameter (higher = faster convergence to mean)
        """
        super().__init__(n_clients, participation_rate)
        self.threshold_percentile = threshold_percentile
        self.theta = theta
        
        # Track loss history
        self.estimated_losses = {i: float('inf') for i in range(n_clients)}
        self.last_observed_round = {i: 0 for i in range(n_clients)}
        self.participation_history = []
        self.global_mean_loss = 1.0  # Initial estimate
        
    def select_clients(self, **kwargs) -> List[int]:
        """
        Select clients with estimated losses above the adaptive threshold.
        Higher loss = more potential for improvement = should be selected.
        """
        if self.round_num == 0:
            # First round: select all clients
            return list(range(self.n_clients))
        
        # Apply O-U estimation to update stale loss estimates
        self._update_stale_estimates()
        
        # Get all estimated losses
        losses = np.array([self.estimated_losses[i] for i in range(self.n_clients)])
        
        # Compute adaptive threshold based on percentile
        threshold = np.percentile(losses, self.threshold_percentile)
        
        # Select clients above threshold
        selected = [i for i in range(self.n_clients) if losses[i] >= threshold]
        
        # If too many selected, take top-k by loss
        if len(selected) > self.k:
            sorted_clients = sorted(selected, key=lambda c: self.estimated_losses[c], reverse=True)
            selected = sorted_clients[:self.k]
        
        # Ensure at least one client is selected
        if len(selected) == 0:
            max_client = max(range(self.n_clients), key=lambda c: self.estimated_losses[c])
            selected = [max_client]
        
        return selected
    
    def _update_stale_estimates(self):
        """Apply O-U process decay to estimates for non-recently-selected clients."""
        for client_id in range(self.n_clients):
            rounds_since_observed = self.round_num - self.last_observed_round[client_id]
            if rounds_since_observed > 0 and self.estimated_losses[client_id] != float('inf'):
                # O-U drift toward global mean: loss_t = loss_0 * exp(-θt) + μ * (1 - exp(-θt))
                decay = np.exp(-self.theta * rounds_since_observed)
                self.estimated_losses[client_id] = (
                    self.estimated_losses[client_id] * decay + 
                    self.global_mean_loss * (1 - decay)
                )
    
    def update(self, selected_clients: List[int], 
               client_losses: Dict[int, float] = None,
               **kwargs) -> None:
        """
        Update internal state after a round.
        
        Args:
            selected_clients: Clients that participated this round
            client_losses: Dict mapping client_id -> local loss after training
        """
        if client_losses is not None:
            for client_id, loss in client_losses.items():
                self.estimated_losses[client_id] = loss
                self.last_observed_round[client_id] = self.round_num + 1
            
            # Update global mean loss estimate
            if len(client_losses) > 0:
                self.global_mean_loss = np.mean(list(client_losses.values()))
                
        self.participation_history.append(selected_clients)
        self.increment_round()
    
    def get_stats(self) -> Dict:
        """Return selection statistics."""
        return {
            'estimated_losses': self.estimated_losses.copy(),
            'participation_history': self.participation_history.copy(),
            'round': self.round_num
        }


class PowerOfChoiceSelector(ClientSelector):
    """
    Power-of-Choice Client Selection
    
    A simpler alternative that samples d clients randomly, then selects
    the one with highest estimated contribution (loss).
    """
    
    def __init__(self, n_clients: int, participation_rate: float = 1.0,
                 d_choices: int = 2):
        """
        Args:
            n_clients: Total number of clients
            participation_rate: Fraction of clients to select
            d_choices: Number of random candidates to sample before choosing
        """
        super().__init__(n_clients, participation_rate)
        self.d = min(d_choices, n_clients)
        self.estimated_losses = np.ones(n_clients)  # Initial estimates
        
    def select_clients(self, **kwargs) -> List[int]:
        """Select k clients using power-of-d-choices."""
        selected = []
        available = list(range(self.n_clients))
        
        for _ in range(self.k):
            if len(available) == 0:
                break
                
            # Sample d candidates
            d_actual = min(self.d, len(available))
            candidates = np.random.choice(available, d_actual, replace=False)
            
            # Select candidate with highest estimated loss
            best = max(candidates, key=lambda c: self.estimated_losses[c])
            selected.append(best)
            available.remove(best)
            
        return selected
    
    def update(self, selected_clients: List[int], client_losses: Dict[int, float] = None,
               **kwargs) -> None:
        """Update loss estimates."""
        if client_losses:
            for client_id, loss in client_losses.items():
                self.estimated_losses[client_id] = loss
        self.increment_round()


class FedSamplingSelector(ClientSelector):
    """
    FedSampling: Data-Uniform Sampling for Federated Learning (Qi et al., IJCAI 2023)
    
    Instead of uniform client sampling, uses data-proportional sampling.
    Clients with more data have higher probability of being selected.
    
    Sampling probability: p_i = D_i / sum(D_j) for all j
    
    This ensures data-uniform sampling across the federation, which improves
    performance especially when client data sizes are highly imbalanced.
    
    Reference: https://www.ijcai.org/proceedings/2023/0462
    """
    
    def __init__(self, n_clients: int, participation_rate: float = 1.0,
                 client_data_sizes: List[int] = None):
        """
        Args:
            n_clients: Total number of clients
            participation_rate: Fraction of clients to select per round
            client_data_sizes: List of data sizes for each client (publicly known)
                              If None, falls back to uniform sampling
        """
        super().__init__(n_clients, participation_rate)
        
        if client_data_sizes is not None:
            self.data_sizes = np.array(client_data_sizes, dtype=float)
            total = np.sum(self.data_sizes)
            self.sampling_probs = self.data_sizes / total if total > 0 else np.ones(n_clients) / n_clients
        else:
            # Fallback to uniform if sizes not provided
            self.data_sizes = None
            self.sampling_probs = np.ones(n_clients) / n_clients
            
    def set_data_sizes(self, client_data_sizes: List[int]) -> None:
        """
        Update client data sizes (can be called after initialization).
        
        Args:
            client_data_sizes: List of data sizes for each client
        """
        self.data_sizes = np.array(client_data_sizes, dtype=float)
        total = np.sum(self.data_sizes)
        self.sampling_probs = self.data_sizes / total if total > 0 else np.ones(self.n_clients) / self.n_clients
        
    def select_clients(self, **kwargs) -> List[int]:
        """
        Select k clients with probability proportional to their data size.
        
        Uses sampling without replacement, with probabilities normalized
        to account for already-selected clients.
        """
        # Sample k clients without replacement, weighted by data size
        selected = list(np.random.choice(
            self.n_clients, 
            size=self.k, 
            replace=False, 
            p=self.sampling_probs
        ))
        return selected
    
    def update(self, selected_clients: List[int], client_losses: Dict[int, float] = None,
               **kwargs) -> None:
        """No state update needed for FedSampling."""
        self.increment_round()
    
    def get_stats(self) -> Dict:
        """Return selection statistics."""
        return {
            'data_sizes': self.data_sizes.copy() if self.data_sizes is not None else None,
            'sampling_probs': self.sampling_probs.copy(),
            'round': self.round_num
        }


# =============================================================================
# Factory function
# =============================================================================

SELECTORS = {
    'random': RandomSelector,
    'ucb': UCBClientSelector,
    'threshold': ThresholdClientSelector,
    'power_of_choice': PowerOfChoiceSelector,
    'fedsampling': FedSamplingSelector,
}


def create_client_selector(method: str, n_clients: int, 
                           participation_rate: float = 1.0, **kwargs) -> ClientSelector:
    """
    Factory function to create a client selector.
    
    Args:
        method: One of 'random', 'ucb', 'threshold', 'power_of_choice', 'fedsampling'
        n_clients: Total number of clients
        participation_rate: Fraction of clients to select per round
        **kwargs: Additional arguments for specific selectors:
            - ucb: exploration_param (float), loss_decay (float)
            - threshold: threshold_percentile (float), theta (float)
            - power_of_choice: d_choices (int)
            - fedsampling: client_data_sizes (List[int])
    
    Returns:
        ClientSelector instance
    """
    method = method.lower()
    if method not in SELECTORS:
        raise ValueError(f"Unknown selection method: {method}. "
                        f"Available: {list(SELECTORS.keys())}")
    
    return SELECTORS[method](n_clients, participation_rate, **kwargs)


def get_available_selectors() -> List[str]:
    """Return list of available selection methods."""
    return list(SELECTORS.keys())


def get_available_selection_methods() -> List[str]:
    """Return list of available selection methods (alias for get_available_selectors)."""
    return ['full'] + list(SELECTORS.keys())
