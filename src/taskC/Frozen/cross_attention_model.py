"""
Cross-Attention Model based on Qwen3
Takes two sets of vectors and returns an attention matrix.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union
from transformers import Qwen2Config


class RMSNorm(nn.Module):
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


class CrossAttentionModel(nn.Module):
    """
    Cross-Attention model based on Qwen3
    
    Args:
        config: model configuration (Qwen3Config or dict)
        layer_idx: layer index for initialization (default last)
    """
    def __init__(self, config: Union[Qwen2Config, dict], layer_idx: Optional[int] = None):
        super().__init__()
        
        # Save configuration
        if isinstance(config, dict):
            self.config = config
        else:
            self.config = config.to_dict()
        
        # Architecture parameters
        self.hidden_size = self.config.get('hidden_size', 4096)
        self.num_attention_heads = self.config.get('num_attention_heads', 32)
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.rope_theta = self.config.get('rope_theta', 10000.0)
        self.max_position_embeddings = self.config.get('max_position_embeddings', 32768)
        
        # Projections for query and key (WITHOUT value and output)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Normalization for query and key
        self.q_norm = RMSNorm(self.hidden_size, eps=self.config.get('rms_norm_eps', 1e-6))
        self.k_norm = RMSNorm(self.hidden_size, eps=self.config.get('rms_norm_eps', 1e-6))
        
        # Flag for tracking weight source
        self.weights_initialized = False

    def forward(self, vectors_1: torch.Tensor, vectors_2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Cross-Attention model
        
        Args:
            vectors_1: (batch_size_1, hidden_size) - first set of vectors (query)
            vectors_2: (batch_size_2, hidden_size) - second set of vectors (key)
            
        Returns:
            attention_matrix: (batch_size_1, batch_size_2) - attention matrix
        """
        # Check dimensions
        batch_size_1, hidden_size_1 = vectors_1.shape
        batch_size_2, hidden_size_2 = vectors_2.shape
        
        assert hidden_size_1 == self.hidden_size, f"vectors_1 dimension {hidden_size_1} != {self.hidden_size}"
        assert hidden_size_2 == self.hidden_size, f"vectors_2 dimension {hidden_size_2} != {self.hidden_size}"
        
        # Project to query and key
        query = self.q_proj(vectors_1)  # (batch_size_1, hidden_size)
        key = self.k_proj(vectors_2)    # (batch_size_2, hidden_size)
        
        # Normalize query and key
        query = self.q_norm(query)
        key = self.k_norm(key)
        
        # Reshape to (batch_size, num_heads, seq_len, head_dim)
        query = query.view(batch_size_1, self.num_attention_heads, 1, self.head_dim)
        key = key.view(batch_size_2, self.num_attention_heads, 1, self.head_dim)
        
        # Transpose for attention computation
        query = query.transpose(1, 2)  # (batch_size_1, 1, num_heads, head_dim)
        key = key.transpose(1, 2)      # (batch_size_2, 1, num_heads, head_dim)
        
        # Compute attention scores
        # We need to compute attention between all pairs (batch_size_1, batch_size_2)
        # For this, we expand query and key to create all combinations
        
        # Expand query: (batch_size_1, 1, num_heads, head_dim) -> (batch_size_1, batch_size_2, num_heads, head_dim)
        query_expanded = query.unsqueeze(1).expand(-1, batch_size_2, -1, -1, -1)
        
        # Expand key: (batch_size_2, 1, num_heads, head_dim) -> (batch_size_1, batch_size_2, num_heads, head_dim)
        key_expanded = key.unsqueeze(0).expand(batch_size_1, -1, -1, -1, -1)
        
        # Compute attention scores
        attention_scores = torch.matmul(query_expanded, key_expanded.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # Apply softmax along the last dimension (head_dim)
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Average across attention heads
        attention_matrix = attention_probs.mean(dim=2)  # (batch_size_1, batch_size_2, 1, 1)
        attention_matrix = attention_matrix.squeeze(-1).squeeze(-1)  # (batch_size_1, batch_size_2)
        
        return attention_matrix

    def get_num_params(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size_mb(self) -> float:
        """Get model size in MB"""
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    def load_from_qwen3(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", layer_idx: int = -1):
        """
        Load weights from Qwen3 model
        
        Args:
            model_name: name of the Qwen3 model
            layer_idx: layer index to load from (default: last layer)
        """
        from transformers import Qwen2ForCausalLM
        
        print(f"🔄 Loading weights from {model_name}, layer {layer_idx}")
        
        # Load Qwen3 model
        model = Qwen2ForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Get the specified layer
        if layer_idx < 0:
            layer_idx = len(model.model.layers) + layer_idx
        
        layer = model.model.layers[layer_idx]
        
        # Load attention weights
        self.q_proj.weight.data = layer.self_attn.q_proj.weight.data.clone()
        self.k_proj.weight.data = layer.self_attn.k_proj.weight.data.clone()
        
        # Load normalization weights
        self.q_norm.weight.data = layer.self_attn.q_norm.weight.data.clone()
        self.k_norm.weight.data = layer.self_attn.k_norm.weight.data.clone()
        
        # Mark weights as initialized
        self.weights_initialized = True
        
        print(f"✅ Weights loaded successfully from layer {layer_idx}")
        
        # Free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def initialize_weights(self, method: str = "xavier_uniform"):
        """
        Initialize model weights
        
        Args:
            method: initialization method ("xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal")
        """
        if self.weights_initialized:
            print("⚠️ Weights already initialized from Qwen3")
            return
        
        print(f"🔄 Initializing weights with {method}")
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if method == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight)
                elif method == "xavier_normal":
                    nn.init.xavier_normal_(module.weight)
                elif method == "kaiming_uniform":
                    nn.init.kaiming_uniform_(module.weight)
                elif method == "kaiming_normal":
                    nn.init.kaiming_normal_(module.weight)
                else:
                    raise ValueError(f"Unknown initialization method: {method}")
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        self.weights_initialized = True
        print("✅ Weights initialized successfully")


if __name__ == "__main__":
    # Test the model
    print("🧪 Testing Cross-Attention Model...")
    
    # Create test configuration
    config = {
        'hidden_size': 512,
        'num_attention_heads': 8,
        'rope_theta': 10000.0,
        'max_position_embeddings': 32768,
        'rms_norm_eps': 1e-6
    }
    
    # Create model
    model = CrossAttentionModel(config)
    
    # Test forward pass
    batch_size_1, batch_size_2 = 4, 6
    hidden_size = 512
    
    vectors_1 = torch.randn(batch_size_1, hidden_size)
    vectors_2 = torch.randn(batch_size_2, hidden_size)
    
    with torch.no_grad():
        attention_matrix = model(vectors_1, vectors_2)
    
    print(f"✅ Test passed!")
    print(f"   Input shapes: {vectors_1.shape}, {vectors_2.shape}")
    print(f"   Output shape: {attention_matrix.shape}")
    print(f"   Model parameters: {model.get_num_params():,}")
    print(f"   Model size: {model.get_model_size_mb():.2f} MB") 