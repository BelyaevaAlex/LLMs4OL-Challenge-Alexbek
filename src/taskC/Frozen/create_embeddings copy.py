import os
import json
from pathlib import Path
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModel
import tqdm

def load_model_and_tokenizer():
    """Load Qwen3-Embedding-4B model and tokenizer."""
    model_name = "Qwen/Qwen3-Embedding-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model, tokenizer

def get_embeddings(texts: List[str], model, tokenizer) -> List[List[float]]:
    """Generate embeddings for a list of texts."""
    embeddings = []
    
    with torch.no_grad():
        for text in tqdm.tqdm(texts, desc="Generating embeddings"):
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().tolist()
            embeddings.append(embedding)
    
    return embeddings

def process_directory(base_path: str):
    """Process all subdirectories containing train/txt files."""
    base_path = Path(base_path)
    
    # Load model and tokenizer once
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    
    # Process each subdirectory
    for subdir in base_path.iterdir():
        if not subdir.is_dir():
            continue
            
        train_dir = subdir / "train"
        if not train_dir.exists():
            print(f"No train directory in {subdir}")
            continue
            
        # Find .txt file in train directory
        txt_files = list(train_dir.glob("*.txt"))
        if not txt_files:
            print(f"No .txt file found in {train_dir}")
            continue
            
        txt_file = txt_files[0]
        print(f"\nProcessing {txt_file}")
        
        # Read terms from .txt file
        with open(txt_file, 'r', encoding='utf-8') as f:
            terms = [line.strip() for line in f if line.strip()]
            
        # Generate embeddings
        embeddings = get_embeddings(terms, model, tokenizer)
        
        # Create output data
        output_data = []
        for idx, (term, embedding) in enumerate(zip(terms, embeddings)):
            output_data.append({
                "id": idx,
                "text_description": term,
                "embedding": embedding
            })
            
        # Save embeddings
        output_file = train_dir / f"{txt_file.stem}_embeddings.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
            
        print(f"Saved embeddings to {output_file}")

if __name__ == "__main__":
    base_path = "2025/TaskC-TaxonomyDiscovery"
    print(f"Processing directories in {base_path}")
    process_directory(base_path) 