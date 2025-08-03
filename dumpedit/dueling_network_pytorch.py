# dueling_network_pytorch.py - PyTorch implementation of Dueling DQN
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Dict, Tuple, Optional

class DuelingNetworkPyTorch(nn.Module):
    """PyTorch implementation of Dueling DQN Network with save/load functionality"""
    
    def __init__(self, input_size: int, hidden_sizes: list, output_size: int, 
                 learning_rate: float = 0.001, device: Optional[str] = None):
        super(DuelingNetworkPyTorch, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Detect device (GPU if available)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Build feature extraction layers
        self.feature_layers = nn.ModuleList()
        layer_sizes = [input_size] + hidden_sizes[:-1]
        
        for i in range(len(layer_sizes) - 1):
            self.feature_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        
        # Dueling streams
        feature_size = hidden_sizes[-2] if len(hidden_sizes) > 1 else hidden_sizes[0]
        stream_size = hidden_sizes[-1]
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(feature_size, stream_size),
            nn.LeakyReLU(0.01),
            nn.Linear(stream_size, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_size, stream_size),
            nn.LeakyReLU(0.01),
            nn.Linear(stream_size, output_size)
        )
        
        # Move to device
        self.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Adaptive learning tracking
        self.training_performance_history = []
        self.adaptive_lr_enabled = True
        
        print(f"🧠 PyTorch Dueling Network created: {input_size}→{hidden_sizes}→V(1)+A({output_size})")
        print(f"   Device: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        # Ensure state is on correct device
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        elif state.device != self.device:
            state = state.to(self.device)
        
        # Ensure correct shape
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        # Feature extraction
        x = state
        for i, layer in enumerate(self.feature_layers):
            x = layer(x)
            x = F.leaky_relu(x, 0.01)
        
        # Dueling streams
        value = self.value_stream(x)
        advantages = self.advantage_stream(x)
        
        # Combine value and advantages
        q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))
        
        return q_values
    
    def get_action(self, state: np.ndarray) -> int:
        """Get action from state (no gradient calculation needed)"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.forward(state_tensor)
            return q_values.argmax(dim=-1).item()
    
    def update_weights(self, state: np.ndarray, target_q_values: np.ndarray, 
                      action_taken: int, adaptive_params: dict = None) -> float:
        """Update network weights using target Q-values with adaptive learning support"""
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        target_tensor = torch.FloatTensor(target_q_values).unsqueeze(0).to(self.device)
        
        # Forward pass
        current_q_values = self.forward(state_tensor)
        
        # Calculate loss (only for the action taken)
        loss = F.mse_loss(current_q_values[0, action_taken], target_tensor[0, action_taken])
        
        # Apply adaptive learning rate if provided
        if adaptive_params and 'learning_rate_multiplier' in adaptive_params:
            # Temporarily adjust learning rate
            original_lr = self.optimizer.param_groups[0]['lr']
            self.optimizer.param_groups[0]['lr'] = original_lr * adaptive_params['learning_rate_multiplier']
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Adaptive gradient clipping based on performance
        clip_value = 1.0
        if adaptive_params and 'gradient_clip_multiplier' in adaptive_params:
            clip_value *= adaptive_params['gradient_clip_multiplier']
        
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_value)
        
        # Update weights
        self.optimizer.step()
        
        # Restore original learning rate if it was modified
        if adaptive_params and 'learning_rate_multiplier' in adaptive_params:
            self.optimizer.param_groups[0]['lr'] = original_lr
        
        return loss.item()
    
    def copy_weights_from(self, other_network: 'DuelingNetworkPyTorch'):
        """Copy weights from another network"""
        self.load_state_dict(other_network.state_dict())
    
    def soft_update_from(self, other_network: 'DuelingNetworkPyTorch', tau: float = 0.001, adaptive_params: dict = None):
        """Soft update weights from another network with adaptive tau"""
        # Adaptive tau based on performance
        if adaptive_params and 'tau_multiplier' in adaptive_params:
            tau = tau * adaptive_params['tau_multiplier']
        
        for target_param, source_param in zip(self.parameters(), other_network.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + source_param.data * tau
            )
    
    def set_adaptive_learning_rate(self, performance_trend: float):
        """Adaptively adjust learning rate based on performance"""
        if not self.adaptive_lr_enabled:
            return
        
        base_lr = self.learning_rate
        
        # If performance is improving, slightly increase learning rate
        # If performance is declining, decrease learning rate
        if performance_trend > 0.1:  # Good performance trend
            new_lr = min(base_lr * 1.1, base_lr * 2.0)  # Cap at 2x base rate
        elif performance_trend < -0.1:  # Poor performance trend
            new_lr = max(base_lr * 0.9, base_lr * 0.5)  # Floor at 0.5x base rate
        else:
            new_lr = base_lr  # Stable performance, keep base rate
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
    
    def get_adaptive_params(self, win_rate: float, recent_rewards: list) -> dict:
        """Generate adaptive parameters based on performance metrics"""
        adaptive_params = {}
        
        # Learning rate multiplier based on win rate
        if win_rate > 0.7:  # High win rate - can afford to learn more aggressively
            adaptive_params['learning_rate_multiplier'] = 1.2
            adaptive_params['tau_multiplier'] = 1.2  # Faster target network updates
        elif win_rate < 0.3:  # Low win rate - be more conservative
            adaptive_params['learning_rate_multiplier'] = 0.8
            adaptive_params['tau_multiplier'] = 0.8  # Slower target network updates
        else:
            adaptive_params['learning_rate_multiplier'] = 1.0
            adaptive_params['tau_multiplier'] = 1.0
        
        # Gradient clipping based on reward variance
        if recent_rewards and len(recent_rewards) > 5:
            reward_variance = np.var(recent_rewards)
            if reward_variance > 10:  # High variance - clip more aggressively
                adaptive_params['gradient_clip_multiplier'] = 0.5
            elif reward_variance < 1:  # Low variance - can be less aggressive
                adaptive_params['gradient_clip_multiplier'] = 1.5
            else:
                adaptive_params['gradient_clip_multiplier'] = 1.0
        
        return adaptive_params
    
    def save_model(self, filepath: str):
        """Save model weights and architecture info"""
        save_dict = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'architecture': {
                'input_size': self.input_size,
                'hidden_sizes': self.hidden_sizes,
                'output_size': self.output_size,
                'learning_rate': self.learning_rate
            }
        }
        torch.save(save_dict, filepath)
        print(f"💾 PyTorch model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model weights"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"📂 PyTorch model loaded from {filepath}")
            return True
        else:
            print(f"⚠️ Model file not found: {filepath}")
            return False
    
    @staticmethod
    def create_from_checkpoint(filepath: str, device: Optional[str] = None) -> 'DuelingNetworkPyTorch':
        """Create a new network from a saved checkpoint"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        # Load checkpoint to get architecture
        checkpoint = torch.load(filepath, map_location='cpu')
        arch = checkpoint['architecture']
        
        # Create new network with saved architecture
        network = DuelingNetworkPyTorch(
            input_size=arch['input_size'],
            hidden_sizes=arch['hidden_sizes'],
            output_size=arch['output_size'],
            learning_rate=arch['learning_rate'],
            device=device
        )
        
        # Load weights
        network.load_model(filepath)
        return network


# Compatibility layer to work with existing NumPy-based code
class DuelingNetworkCompatibility:
    """Wrapper to make PyTorch network compatible with existing NumPy interface"""
    
    def __init__(self, pytorch_network: DuelingNetworkPyTorch):
        self.network = pytorch_network

    def update(self, loss_value=None, state=None, action=None, target=None, *args, **kwargs):
        """Add missing update method for compatibility"""
        # Some training code might call update with multiple arguments
        # The actual updates happen in update_weights method
        pass

    def forward(self, state: np.ndarray) -> np.ndarray:
        """NumPy-compatible forward pass"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.network.device)
            if len(state_tensor.shape) == 1:
                state_tensor = state_tensor.unsqueeze(0)
            q_values = self.network.forward(state_tensor)
            return q_values.cpu().numpy().squeeze()
    
    def update_weights(self, state: np.ndarray, target_q_values: np.ndarray, 
                      action_taken: int, adaptive_params: dict = None) -> float:
        """NumPy-compatible weight update with adaptive learning support"""
        return self.network.update_weights(state, target_q_values, action_taken, adaptive_params)
    
    def copy_weights_from(self, other_network):
        """Copy weights from another network (handles both PyTorch and wrapped versions)"""
        if isinstance(other_network, DuelingNetworkCompatibility):
            self.network.copy_weights_from(other_network.network)
        elif isinstance(other_network, DuelingNetworkPyTorch):
            self.network.copy_weights_from(other_network)
        else:
            raise TypeError("Incompatible network type for weight copying")
    
    def soft_update_from(self, other_network, tau: float = 0.001, adaptive_params: dict = None):
        """Soft update from another network with adaptive parameters"""
        if isinstance(other_network, DuelingNetworkCompatibility):
            self.network.soft_update_from(other_network.network, tau, adaptive_params)
        elif isinstance(other_network, DuelingNetworkPyTorch):
            self.network.soft_update_from(other_network, tau, adaptive_params)
        else:
            raise TypeError("Incompatible network type for soft update")
    
    def get_adaptive_params(self, win_rate: float, recent_rewards: list) -> dict:
        """Get adaptive parameters for learning"""
        return self.network.get_adaptive_params(win_rate, recent_rewards)
    
    def set_adaptive_learning_rate(self, performance_trend: float):
        """Set adaptive learning rate"""
        return self.network.set_adaptive_learning_rate(performance_trend)


# Test the implementation
if __name__ == "__main__":
    print("🧪 Testing PyTorch Dueling Network...")
    
    # Create network
    net = DuelingNetworkPyTorch(
        input_size=14,
        hidden_sizes=[64, 32, 16],
        output_size=3,
        learning_rate=0.001
    )
    
    # Test forward pass
    test_state = np.random.randn(14)
    q_values = net.get_action(test_state)
    print(f"Test action: {q_values}")
    
    # Test training
    target_q = np.array([0.5, 0.8, 0.3])
    loss = net.update_weights(test_state, target_q, action_taken=1)
    print(f"Training loss: {loss:.4f}")
    
    # Test save/load
    net.save_model("test_model.pth")
    
    # Create new network and load
    net2 = DuelingNetworkPyTorch(14, [64, 32, 16], 3)
    net2.load_model("test_model.pth")
    
    # Test loaded network produces same output
    q_values2 = net2.get_action(test_state)
    print(f"Loaded network action: {q_values2}")
    
    # Clean up test file
    import os
    if os.path.exists("test_model.pth"):
        os.remove("test_model.pth")
    
    print("✅ PyTorch Dueling Network tests passed!")