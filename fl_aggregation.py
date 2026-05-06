"""
Federated Learning Aggregation Methods

This module implements various FL aggregation strategies:
- FedAvg: Standard Federated Averaging
- FedProx: FedAvg with proximal regularization for non-IID data
- FedAdam: Adaptive server optimizer using Adam
- SCAFFOLD: Stochastic Controlled Averaging for variance reduction

Usage:
    from fl_aggregation import Aggregator, FedAvgAggregator, FedProxAggregator, ...
"""

import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple


class Aggregator(ABC):
    """Abstract base class for FL aggregation strategies."""
    
    def __init__(self, input_shape: Tuple, model_fn, learning_rate: float = 0.0001):
        """
        Args:
            input_shape: Shape of input features (excluding batch dimension)
            model_fn: Function that creates a new model given input_shape and initializer
            learning_rate: Learning rate for local training
        """
        self.input_shape = input_shape
        self.model_fn = model_fn
        self.learning_rate = learning_rate
        self.global_model = None
        self.round_num = 0
        
    @abstractmethod
    def aggregate(self, local_models: List, client_weights: List[float]) -> tf.keras.Model:
        """Aggregate local models into a new global model."""
        pass
    
    @abstractmethod
    def prepare_local_training(self, client_id: int) -> tf.keras.Model:
        """Prepare a local model for training (may include additional state)."""
        pass
    
    @abstractmethod
    def post_local_training(self, client_id: int, local_model: tf.keras.Model, 
                           x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Post-processing after local training (e.g., update control variates)."""
        pass
    
    def initialize_global_model(self, init_distrib='he_uniform'):
        """Initialize the global model."""
        if isinstance(init_distrib, str):
            init_distrib = tf.initializers.HeUniform(seed=42)
        self.global_model = self.model_fn(self.input_shape, init_distrib)
        return self.global_model
    
    def get_global_model(self) -> tf.keras.Model:
        return self.global_model
    
    def increment_round(self):
        self.round_num += 1


# =============================================================================
# FedAvg: Standard Federated Averaging
# =============================================================================

class FedAvgAggregator(Aggregator):
    """
    Standard Federated Averaging (McMahan et al., 2017)
    
    Simple weighted average of client model parameters.
    """
    
    def __init__(self, input_shape: Tuple, model_fn, learning_rate: float = 0.0001):
        super().__init__(input_shape, model_fn, learning_rate)
        
    def aggregate(self, local_models: List, client_weights: List[float]) -> tf.keras.Model:
        """Weighted average of model parameters."""
        n_clients = len(local_models)
        
        # Normalize weights
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]
        
        # Scale and sum weights
        scaled_weights = []
        for i, model in enumerate(local_models):
            model_weights = model.get_weights()
            scaled = [normalized_weights[i] * w for w in model_weights]
            scaled_weights.append(scaled)
        
        # Average weights
        avg_weights = []
        for layer_weights in zip(*scaled_weights):
            layer_avg = np.sum(layer_weights, axis=0)
            avg_weights.append(layer_avg)
        
        # Create new global model
        self.global_model = self.model_fn(self.input_shape, 'zeros')
        self.global_model.set_weights(avg_weights)
        self.increment_round()
        
        return self.global_model
    
    def prepare_local_training(self, client_id: int) -> tf.keras.Model:
        """Clone global model for local training."""
        local_model = tf.keras.models.clone_model(self.global_model)
        local_model.build((None,) + self.input_shape)
        local_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.Recall(name='Recall'),
                tf.keras.metrics.Precision(name='Precision')
            ]
        )
        local_model.set_weights(self.global_model.get_weights())
        return local_model
    
    def post_local_training(self, client_id: int, local_model: tf.keras.Model,
                           x_train: np.ndarray, y_train: np.ndarray) -> None:
        """No post-processing needed for FedAvg."""
        pass


# =============================================================================
# FedProx: Federated Optimization with Proximal Term
# =============================================================================

class FedProxAggregator(Aggregator):
    """
    FedProx (Li et al., 2020)
    
    Adds a proximal term to local optimization to limit drift from global model.
    Local objective: min F_k(w) + (mu/2) * ||w - w_global||^2
    """
    
    def __init__(self, input_shape: Tuple, model_fn, learning_rate: float = 0.0001,
                 mu: float = 0.01):
        """
        Args:
            mu: Proximal term coefficient. Higher = more regularization toward global model.
        """
        super().__init__(input_shape, model_fn, learning_rate)
        self.mu = mu
        
    def aggregate(self, local_models: List, client_weights: List[float]) -> tf.keras.Model:
        """Same as FedAvg - just weighted average."""
        n_clients = len(local_models)
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]
        
        scaled_weights = []
        for i, model in enumerate(local_models):
            model_weights = model.get_weights()
            scaled = [normalized_weights[i] * w for w in model_weights]
            scaled_weights.append(scaled)
        
        avg_weights = []
        for layer_weights in zip(*scaled_weights):
            layer_avg = np.sum(layer_weights, axis=0)
            avg_weights.append(layer_avg)
        
        self.global_model = self.model_fn(self.input_shape, 'zeros')
        self.global_model.set_weights(avg_weights)
        self.increment_round()
        
        return self.global_model
    
    def prepare_local_training(self, client_id: int) -> tf.keras.Model:
        """Create model with FedProx regularization."""
        # Store global weights for proximal term
        global_weights = self.global_model.get_weights()
        
        # Create custom model with proximal regularization
        local_model = self._create_fedprox_model(global_weights)
        local_model.set_weights(global_weights)
        return local_model
    
    def _create_fedprox_model(self, global_weights: List[np.ndarray]) -> tf.keras.Model:
        """Create a model with FedProx proximal regularization."""
        
        # Custom training step that adds proximal term
        class FedProxModel(tf.keras.Model):
            def __init__(self, base_model, global_weights, mu, **kwargs):
                super().__init__(**kwargs)
                self.base_model = base_model
                self.global_weights_tensors = [tf.constant(w, dtype=tf.float32) for w in global_weights]
                self.mu = mu
                
            def call(self, inputs, training=None):
                return self.base_model(inputs, training=training)
            
            def train_step(self, data):
                x, y = data
                
                with tf.GradientTape() as tape:
                    y_pred = self(x, training=True)
                    loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
                    
                    # Add proximal term: (mu/2) * ||w - w_global||^2
                    prox_term = 0.0
                    for w, w_g in zip(self.base_model.trainable_variables, self.global_weights_tensors):
                        prox_term += tf.reduce_sum(tf.square(w - w_g))
                    prox_term = (self.mu / 2.0) * prox_term
                    
                    total_loss = loss + prox_term
                
                gradients = tape.gradient(total_loss, self.base_model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.base_model.trainable_variables))
                self.compiled_metrics.update_state(y, y_pred)
                
                return {m.name: m.result() for m in self.metrics}
            
            def get_weights(self):
                return self.base_model.get_weights()
            
            def set_weights(self, weights):
                self.base_model.set_weights(weights)
        
        # Create base model
        base_model = self.model_fn(self.input_shape, 'zeros')
        
        # Wrap with FedProx
        fedprox_model = FedProxModel(base_model, global_weights, self.mu)
        fedprox_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.Recall(name='Recall'),
                tf.keras.metrics.Precision(name='Precision')
            ]
        )
        
        return fedprox_model
    
    def post_local_training(self, client_id: int, local_model: tf.keras.Model,
                           x_train: np.ndarray, y_train: np.ndarray) -> None:
        """No post-processing needed."""
        pass


# =============================================================================
# FedAdam: Adaptive Server Optimizer
# =============================================================================

class FedAdamAggregator(Aggregator):
    """
    FedAdam (Reddi et al., 2021)
    
    Uses Adam optimizer at the server level for aggregating pseudo-gradients.
    Server maintains momentum (m) and second moment (v) estimates.
    """
    
    def __init__(self, input_shape: Tuple, model_fn, learning_rate: float = 0.0001,
                 server_lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.99,
                 tau: float = 1e-3):
        """
        Args:
            server_lr: Server-side learning rate
            beta1: First moment decay rate
            beta2: Second moment decay rate  
            tau: Small constant for numerical stability
        """
        super().__init__(input_shape, model_fn, learning_rate)
        self.server_lr = server_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        
        # Server optimizer state
        self.m = None  # First moment estimate
        self.v = None  # Second moment estimate
        
    def aggregate(self, local_models: List, client_weights: List[float]) -> tf.keras.Model:
        """Aggregate using Adam-style update on pseudo-gradients."""
        n_clients = len(local_models)
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]
        
        global_weights = self.global_model.get_weights()
        
        # Compute pseudo-gradient: delta = sum_i p_i * (w_global - w_i)
        # This is the negative of the model update direction
        delta = []
        for layer_idx in range(len(global_weights)):
            layer_delta = np.zeros_like(global_weights[layer_idx])
            for i, model in enumerate(local_models):
                local_w = model.get_weights()[layer_idx]
                layer_delta += normalized_weights[i] * (global_weights[layer_idx] - local_w)
            delta.append(layer_delta)
        
        # Initialize optimizer state if needed
        if self.m is None:
            self.m = [np.zeros_like(d) for d in delta]
            self.v = [np.zeros_like(d) for d in delta]
        
        # Adam update
        new_weights = []
        for layer_idx in range(len(global_weights)):
            # Update biased first moment estimate
            self.m[layer_idx] = self.beta1 * self.m[layer_idx] + (1 - self.beta1) * delta[layer_idx]
            
            # Update biased second raw moment estimate
            self.v[layer_idx] = self.beta2 * self.v[layer_idx] + (1 - self.beta2) * np.square(delta[layer_idx])
            
            # Compute update (note: we subtract because delta is the gradient direction)
            update = self.server_lr * self.m[layer_idx] / (np.sqrt(self.v[layer_idx]) + self.tau)
            new_w = global_weights[layer_idx] - update
            new_weights.append(new_w)
        
        self.global_model = self.model_fn(self.input_shape, 'zeros')
        self.global_model.set_weights(new_weights)
        self.increment_round()
        
        return self.global_model
    
    def prepare_local_training(self, client_id: int) -> tf.keras.Model:
        """Same as FedAvg."""
        local_model = tf.keras.models.clone_model(self.global_model)
        local_model.build((None,) + self.input_shape)
        local_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.Recall(name='Recall'),
                tf.keras.metrics.Precision(name='Precision')
            ]
        )
        local_model.set_weights(self.global_model.get_weights())
        return local_model
    
    def post_local_training(self, client_id: int, local_model: tf.keras.Model,
                           x_train: np.ndarray, y_train: np.ndarray) -> None:
        """No post-processing needed."""
        pass


# =============================================================================
# SCAFFOLD: Stochastic Controlled Averaging
# =============================================================================

class SCAFFOLDAggregator(Aggregator):
    """
    SCAFFOLD (Karimireddy et al., 2020)
    
    Uses control variates to correct client drift in non-IID settings.
    
    Important: SCAFFOLD was designed for vanilla SGD, not Adam. This implementation
    uses SGD for stability, as the control variate update formula assumes a linear
    relationship between gradients and weight updates.
    """
    
    def __init__(self, input_shape: Tuple, model_fn, learning_rate: float = 0.01,
                 n_clients: int = 5, local_lr: float = None):
        """
        Args:
            n_clients: Number of clients (needed to initialize control variates)
            local_lr: Local learning rate (defaults to learning_rate)
        """
        super().__init__(input_shape, model_fn, learning_rate)
        self.n_clients = n_clients
        self.local_lr = local_lr if local_lr is not None else learning_rate
        
        # Control variates
        self.c = None  # Server control variate
        self.c_clients = {}  # Client control variates {client_id: control_variate}
        
        # Storage for initial weights before local training
        self.client_init_weights = {}
        
        # Storage for control variate updates
        self.delta_c_clients = {}
        
    def _initialize_control_variates(self, weights_shape: List):
        """Initialize control variates to zero."""
        self.c = [np.zeros_like(w) for w in weights_shape]
        self.c_clients = {}
        for i in range(self.n_clients):
            self.c_clients[i] = [np.zeros_like(w) for w in weights_shape]
    
    def aggregate(self, local_models: List, client_weights: List[float]) -> tf.keras.Model:
        """Standard FedAvg aggregation - SCAFFOLD correction happens during local training."""
        n_clients = len(local_models)
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]
        
        global_weights = self.global_model.get_weights()
        
        # Initialize control variates if needed
        if self.c is None:
            self._initialize_control_variates(global_weights)
        
        # Compute weighted average of model parameters (same as FedAvg)
        new_weights = []
        for layer_idx in range(len(global_weights)):
            layer_avg = np.zeros_like(global_weights[layer_idx])
            for i, model in enumerate(local_models):
                local_w = model.get_weights()[layer_idx]
                layer_avg += normalized_weights[i] * local_w
            new_weights.append(layer_avg)
        
        # Update server control variate: c = c + (1/N) * sum_i (c_i^+ - c_i)
        if self.delta_c_clients:
            for layer_idx in range(len(self.c)):
                delta_c = np.zeros_like(self.c[layer_idx])
                for client_id, delta in self.delta_c_clients.items():
                    delta_c += delta[layer_idx]
                self.c[layer_idx] += delta_c / self.n_clients
            self.delta_c_clients = {}
        
        self.global_model = self.model_fn(self.input_shape, 'zeros')
        self.global_model.set_weights(new_weights)
        self.increment_round()
        
        return self.global_model
    
    def prepare_local_training(self, client_id: int) -> tf.keras.Model:
        """Prepare model for local training with SCAFFOLD correction."""
        global_weights = self.global_model.get_weights()
        
        # Initialize control variates if needed
        if self.c is None:
            self._initialize_control_variates(global_weights)
        
        if client_id not in self.c_clients:
            self.c_clients[client_id] = [np.zeros_like(w) for w in global_weights]
        
        # Store initial weights for control variate update
        self.client_init_weights[client_id] = [w.copy() for w in global_weights]
        
        # Create model with SCAFFOLD training
        local_model = self._create_scaffold_model(client_id)
        local_model.set_weights(global_weights)
        
        return local_model
    
    def _create_scaffold_model(self, client_id: int) -> tf.keras.Model:
        """Create model with SCAFFOLD gradient correction using SGD."""
        
        c_global = [tf.constant(c, dtype=tf.float32) for c in self.c]
        c_local = [tf.constant(c, dtype=tf.float32) for c in self.c_clients[client_id]]
        
        class SCAFFOLDModel(tf.keras.Model):
            def __init__(self, base_model, **kwargs):
                super().__init__(**kwargs)
                self.base_model = base_model
                self.c_global = c_global
                self.c_local = c_local
                
            def call(self, inputs, training=None):
                return self.base_model(inputs, training=training)
            
            def train_step(self, data):
                x, y = data
                
                with tf.GradientTape() as tape:
                    y_pred = self(x, training=True)
                    loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
                
                # Get raw gradients
                gradients = tape.gradient(loss, self.base_model.trainable_variables)
                
                # Apply SCAFFOLD correction: g_corrected = g - c_i + c
                corrected_gradients = []
                for g, cg, cl in zip(gradients, self.c_global, self.c_local):
                    corrected = g - cl + cg
                    corrected_gradients.append(corrected)
                
                self.optimizer.apply_gradients(zip(corrected_gradients, self.base_model.trainable_variables))
                self.compiled_metrics.update_state(y, y_pred)
                
                return {m.name: m.result() for m in self.metrics}
            
            def get_weights(self):
                return self.base_model.get_weights()
            
            def set_weights(self, weights):
                self.base_model.set_weights(weights)
        
        base_model = self.model_fn(self.input_shape, 'zeros')
        scaffold_model = SCAFFOLDModel(base_model)
        
        # Use SGD for SCAFFOLD (as in original paper)
        scaffold_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=self.local_lr),
            loss='binary_crossentropy',
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.Recall(name='Recall'),
                tf.keras.metrics.Precision(name='Precision')
            ]
        )
        
        return scaffold_model
    
    def post_local_training(self, client_id: int, local_model: tf.keras.Model,
                           x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Update client control variate using Option 2 (weight difference).
        
        For SGD without momentum: delta_w = -eta * sum(g_t)
        So sum(g_t) = (w_init - w_final) / eta
        And avg_gradient = (w_init - w_final) / (K * eta)
        
        c_i^+ = c_i - c + avg_gradient
        """
        init_weights = self.client_init_weights[client_id]
        final_weights = local_model.get_weights()
        
        # Estimate K = number of gradient steps
        batch_size = 32
        K = max(1, len(x_train) // batch_size)
        eta = self.local_lr
        
        old_c = self.c_clients[client_id]
        new_c = []
        delta_c = []
        
        for layer_idx in range(len(old_c)):
            # Estimated average gradient from weight change
            # avg_g = (w_init - w_final) / (K * eta)
            avg_grad = (init_weights[layer_idx] - final_weights[layer_idx]) / (K * eta)
            
            # SCAFFOLD Option 2: c_i^+ = c_i - c + avg_grad
            new_c_layer = old_c[layer_idx] - self.c[layer_idx] + avg_grad
            
            new_c.append(new_c_layer)
            delta_c.append(new_c_layer - old_c[layer_idx])
        
        self.c_clients[client_id] = new_c
        self.delta_c_clients[client_id] = delta_c


# =============================================================================
# Factory function to create aggregators
# =============================================================================

AGGREGATORS = {
    'fedavg': FedAvgAggregator,
    'fedprox': FedProxAggregator,
    'fedadam': FedAdamAggregator,
    'scaffold': SCAFFOLDAggregator,
}

def create_aggregator(method: str, input_shape: Tuple, model_fn, 
                      learning_rate: float = 0.0001, **kwargs) -> Aggregator:
    """
    Factory function to create an aggregator.
    
    Args:
        method: One of 'fedavg', 'fedprox', 'fedadam', 'scaffold'
        input_shape: Shape of input features
        model_fn: Function to create model
        learning_rate: Local learning rate
        **kwargs: Additional arguments for specific aggregators:
            - fedprox: mu (float)
            - fedadam: server_lr, beta1, beta2, tau
            - scaffold: n_clients (int), local_lr (float, default=0.01 for SGD)
    
    Returns:
        Aggregator instance
    """
    method = method.lower()
    if method not in AGGREGATORS:
        raise ValueError(f"Unknown aggregation method: {method}. "
                        f"Available: {list(AGGREGATORS.keys())}")
    
    # SCAFFOLD uses higher default LR since it uses SGD
    if method == 'scaffold' and 'local_lr' not in kwargs:
        kwargs['local_lr'] = 0.01
    
    return AGGREGATORS[method](input_shape, model_fn, learning_rate, **kwargs)


def get_available_methods() -> List[str]:
    """Return list of available aggregation methods."""
    return list(AGGREGATORS.keys())
