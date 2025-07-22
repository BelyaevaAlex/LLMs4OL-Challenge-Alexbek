"""
Script for creating embeddings from text descriptions
Uses Qwen3-Embedding-4B model for generating embeddings.
"""

import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any
import argparse
from pathlib import Path
import tqdm


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Extract last token embeddings with proper handling of padding"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def load_embedding_model():
    """Load model for creating embeddings"""
    model_name = "Qwen/Qwen3-Embedding-4B"
    
    # Configure tokenizer according to official documentation
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    
    # Load model with optimizations
    model = AutoModel.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    
    return model, tokenizer


def get_embeddings_mean(texts: List[str], model, tokenizer, batch_size: int = 32) -> List[List[float]]:
    """Generate embeddings using mean pooling (old method)."""
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(texts), batch_size), desc="Generating embeddings (mean)"):
            batch_texts = texts[i:i+batch_size]
            
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
            embeddings.extend(batch_embeddings.tolist())
    
    return embeddings


def get_embeddings_pooling(texts: List[str], model, tokenizer, batch_size: int = 32) -> List[List[float]]:
    """Generate embeddings using last token pooling with normalization (new method)."""
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(texts), batch_size), desc="Generating embeddings (pooling)"):
            batch_texts = texts[i:i+batch_size]
            
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            outputs = model(**inputs)
            # Use last_token_pool to extract embeddings
            batch_embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            # Apply L2 normalization
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            batch_embeddings = batch_embeddings.cpu().numpy()
            
            embeddings.extend(batch_embeddings.tolist())
    
    return embeddings


def create_embeddings(
    input_file: str,
    output_file: str,
    method: str = "pooling",
    batch_size: int = 32
):
    """
    Create embeddings for entities from input file
    
    Args:
        input_file: path to input JSON file with entities
        output_file: path to output JSON file with embeddings
        method: embedding method ("pooling" or "mean")
        batch_size: batch size for processing
    """
    print(f"📁 Loading entities from {input_file}")
    
    # Load entities
    with open(input_file, 'r', encoding='utf-8') as f:
        entities = json.load(f)
    
    print(f"✅ Loaded {len(entities)} entities")
    
    # Extract text descriptions
    texts = []
    for entity in entities:
        text = entity.get('text_description', '')
        if not text:
            print(f"⚠️ Entity {entity.get('id', 'unknown')} has no text_description")
            text = f"entity_{entity.get('id', 'unknown')}"
        texts.append(text)
    
    # Load embedding model
    print("🔄 Loading embedding model...")
    model, tokenizer = load_embedding_model()
    
    # Generate embeddings
    print(f"⚡ Generating embeddings (method: {method})...")
    if method == "pooling":
        embeddings = get_embeddings_pooling(texts, model, tokenizer, batch_size)
    else:
        embeddings = get_embeddings_mean(texts, model, tokenizer, batch_size)
    
    # Update entities with embeddings
    print("📝 Updating entities with embeddings...")
    for i, entity in enumerate(entities):
        entity['embedding'] = embeddings[i]
    
    # Save updated entities
    print(f"💾 Saving to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Successfully created embeddings for {len(entities)} entities")
    print(f"   Embedding dimension: {len(embeddings[0])}")
    print(f"   Method: {method}")


def main():
    parser = argparse.ArgumentParser(description="Create embeddings for entities")
    parser.add_argument("input_file", help="Path to input JSON file with entities")
    parser.add_argument("output_file", help="Path to output JSON file with embeddings")
    parser.add_argument("--method", choices=["pooling", "mean"], default="pooling", 
                       help="Embedding method (default: pooling)")
    parser.add_argument("--batch_size", type=int, default=32, 
                       help="Batch size for processing (default: 32)")
    
    args = parser.parse_args()
    
    create_embeddings(
        input_file=args.input_file,
        output_file=args.output_file,
        method=args.method,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main() 