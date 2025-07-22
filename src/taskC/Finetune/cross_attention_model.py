"""
Cross-Attention Model based on Qwen3
Takes two sets of vectors and returns an attention matrix.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Tuple, Optional, Dict, Any
import json
import os


class Qwen3RMSNorm(nn.Module):
    """RMS Layer Normalization from Qwen3"""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class CrossAttentionModel(nn.Module):
    """
    Cross-Attention model based on Qwen3
    
    Args:
        config: model configuration (Qwen3Config or dict)
        layer_idx: layer index for initialization (default: last)
    """
    
    def __init__(self, config, layer_idx: int = -1):
        super().__init__()
        
        # Save configuration
        if hasattr(config, 'to_dict'):
            self.config = config.to_dict()
        else:
            self.config = config
            
        # Architecture parameters
        self.hidden_size = self.config['hidden_size']
        self.num_attention_heads = self.config['num_attention_heads']
        self.num_key_value_heads = self.config['num_key_value_heads']
        self.head_dim = self.config.get('head_dim', self.hidden_size // self.num_attention_heads)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.layer_idx = layer_idx
        
        # Projections for query and key (WITHOUT value and output)
        self.q_proj = nn.Linear(
            self.hidden_size, 
            self.num_attention_heads * self.head_dim, 
            bias=self.config.get('attention_bias', False)
        )
        self.k_proj = nn.Linear(
            self.hidden_size, 
            self.num_key_value_heads * self.head_dim, 
            bias=self.config.get('attention_bias', False)
        )
        
        # Normalization for query and key
        rms_norm_eps = self.config.get('rms_norm_eps', 1e-6)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=rms_norm_eps)
        
        # Flag for tracking weight source
        self.initialized_from_qwen3 = False
        
    def forward(self, vectors_1: torch.Tensor, vectors_2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of cross-attention model
        
        Args:
            vectors_1: tensor of size (batch_size_1, hidden_size) - query
            vectors_2: tensor of size (batch_size_2, hidden_size) - key
            
        Returns:
            attention_matrix: attention matrix of size (batch_size_1, batch_size_2)
        """
        batch_size_1, hidden_size_1 = vectors_1.shape
        batch_size_2, hidden_size_2 = vectors_2.shape
        
        # Dimension check
        assert hidden_size_1 == self.hidden_size, f"vectors_1 dimension {hidden_size_1} != {self.hidden_size}"
        assert hidden_size_2 == self.hidden_size, f"vectors_2 dimension {hidden_size_2} != {self.hidden_size}"
        
        # Projection to query and key
        # vectors_1 -> query: (batch_size_1, hidden_size) -> (batch_size_1, num_heads, head_dim)
        query_states = self.q_proj(vectors_1).view(batch_size_1, -1, self.head_dim).transpose(0, 1)
        
        # vectors_2 -> key: (batch_size_2, hidden_size) -> (batch_size_2, num_heads, head_dim)  
        key_states = self.k_proj(vectors_2).view(batch_size_2, -1, self.head_dim).transpose(0, 1)
        
        # Normalization of query and key
        query_states = self.q_norm(query_states)  # (num_heads, batch_size_1, head_dim)
        key_states = self.k_norm(key_states)      # (num_key_value_heads, batch_size_2, head_dim)
        
        # Repeat key states for head groups (if GQA is used)
        if self.num_key_value_groups > 1:
            key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        
        # Compute attention scores
        # query @ key^T: (num_heads, batch_size_1, head_dim) @ (num_heads, head_dim, batch_size_2)
        # -> (num_heads, batch_size_1, batch_size_2)
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scaling
        
        # Softmax over last dimension (key dimension)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Average over heads: (num_heads, batch_size_1, batch_size_2) -> (batch_size_1, batch_size_2)
        attention_matrix = attention_weights.mean(dim=0)
        
        return attention_matrix
    
    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeat key/value states for head groups (GQA)
        
        Args:
            hidden_states: (num_key_value_heads, seq_len, head_dim)
            n_rep: number of repetitions
            
        Returns:
            repeated_states: (num_attention_heads, seq_len, head_dim)
        """
        if n_rep == 1:
            return hidden_states
        
        num_key_value_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, None, :, :].expand(num_key_value_heads, n_rep, seq_len, head_dim)
        return hidden_states.reshape(num_key_value_heads * n_rep, seq_len, head_dim)
    
    def save_pretrained(self, save_directory: str):
        """
        Save trained model weights
        
        Args:
            save_directory: path to directory for saving
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model state
        model_state = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'layer_idx': self.layer_idx,
            'initialized_from_qwen3': self.initialized_from_qwen3,
            'model_type': 'cross_attention'
        }
        
        torch.save(model_state, os.path.join(save_directory, 'pytorch_model.bin'))
        
        # Save configuration separately
        with open(os.path.join(save_directory, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
            
        print(f"Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, model_path: str):
        """
        Load trained model weights
        
        Args:
            model_path: path to directory or file with weights
            
        Returns:
            model: CrossAttentionModel instance with loaded weights
        """
        if os.path.isdir(model_path):
            # Load from directory
            model_file = os.path.join(model_path, 'pytorch_model.bin')
            config_file = os.path.join(model_path, 'config.json')
            
            # Load configuration
            with open(config_file, 'r') as f:
                config = json.load(f)
        else:
            # Load from file
            model_file = model_path
            try:
                checkpoint = torch.load(model_file, map_location='cpu', weights_only=True)
            except Exception:
                checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
            config = checkpoint['config']
        
        # Create model
        model = cls(config)
        
        # Load weights
        try:
            checkpoint = torch.load(model_file, map_location='cpu', weights_only=True)
        except Exception:
            checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.initialized_from_qwen3 = checkpoint.get('initialized_from_qwen3', False)
        
        print(f"Model loaded from {model_path}")
        return model
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            info: dictionary with model information
        """
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'CrossAttentionModel',
            'hidden_size': self.hidden_size,
            'num_attention_heads': self.num_attention_heads,
            'num_key_value_heads': self.num_key_value_heads,
            'head_dim': self.head_dim,
            'num_key_value_groups': self.num_key_value_groups,
            'scaling': self.scaling,
            'layer_idx': self.layer_idx,
            'initialized_from_qwen3': self.initialized_from_qwen3,
            'num_trainable_params': num_params,
            'device': next(self.parameters()).device,
            'dtype': next(self.parameters()).dtype
        }
    
    def __repr__(self):
        info = self.get_model_info()
        return f"CrossAttentionModel(hidden_size={info['hidden_size']}, " \
               f"num_heads={info['num_attention_heads']}, " \
               f"params={info['num_trainable_params']:,})"


def create_model_from_config(config_dict: Dict[str, Any]) -> CrossAttentionModel:
    """
    Create model from configuration dictionary
    
    Args:
        config_dict: dictionary with configuration parameters
        
    Returns:
        model: CrossAttentionModel instance
    """
    return CrossAttentionModel(config_dict)


if __name__ == "__main__":
    # Test example
    print("🚀 Testing CrossAttentionModel...")
    
    # Create test configuration
    test_config = {
        'hidden_size': 512,
        'num_attention_heads': 8,
        'num_key_value_heads': 8,
        'rms_norm_eps': 1e-6,
        'attention_bias': False
    }
    
    # Create model
    model = CrossAttentionModel(test_config)
    print(f"Created model: {model}")
    
    # Test data
    vectors_1 = torch.randn(3, 512)  # 3 vectors of dimension 512
    vectors_2 = torch.randn(5, 512)  # 5 vectors of dimension 512
    
    # Forward pass
    with torch.no_grad():
        attention_matrix = model(vectors_1, vectors_2)
    
    print(f"Input data:")
    print(f"  vectors_1: {vectors_1.shape}")
    print(f"  vectors_2: {vectors_2.shape}")
    print(f"Output attention matrix: {attention_matrix.shape}")
    print(f"Sum over rows (should be ~1.0): {attention_matrix.sum(dim=1)}")
    
    # Model information
    info = model.get_model_info()
    print(f"\nModel information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Testing completed successfully!") 