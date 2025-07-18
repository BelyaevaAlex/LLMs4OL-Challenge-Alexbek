import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

class FFN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 1024):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

class CustomZeroShotDistMultWithFFN:
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-4B", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        
        # Get embedding dimension from model config
        self.embedding_dim = self.model.config.hidden_size
        
        # Initialize FFN
        self.ffn = FFN(self.embedding_dim).to(device)

    def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        embeddings = []
        
        with torch.no_grad():
            for text in tqdm(texts, desc="Computing embeddings"):
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                # Use mean pooling
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings.append(embedding)
                
        return torch.cat(embeddings, dim=0)

    def _format_prompted(self, terms: List[str], types: List[str]) -> tuple:
        # More informative prompts for Qwen
        formatted_terms = [f"Define the term: {term}" for term in terms]
        formatted_types = [f"Category description: {type_}" for type_ in types]
        return formatted_terms, formatted_types

    def _compute_distmult_scores(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute DistMult scores for all term-type pairs
        h: term embeddings [n_terms x dim]
        r: relation embeddings [n_types x dim]
        t: type embeddings [n_types x dim]
        Returns: scores [n_terms x n_types]
        """
        # Compute h * r for each term-type pair
        hr = h.unsqueeze(1) * r.unsqueeze(0)  # [n_terms x n_types x dim]
        
        # Multiply with t and sum over dimension
        scores = torch.sum(hr * t.unsqueeze(0), dim=-1)  # [n_terms x n_types]
        
        return scores

    def _select_types_adaptive(self, scores: np.ndarray) -> List[int]:
        """
        Select single type with highest score
        """
        # Get index of highest score
        selected_index = np.argmax(scores)
        return [selected_index]  # Return as list for compatibility

    def predict(self, terms_data: List[Dict[str, Any]], types: List[str], output_file: str) -> None:
        # Extract terms
        terms = [item["term"] for item in terms_data]
        
        # Format inputs with prompting
        formatted_terms, formatted_types = self._format_prompted(terms, types)
        
        # Get embeddings
        print("Computing term embeddings...")
        term_embeddings = self._get_embeddings(formatted_terms)  # [n_terms x dim]
        
        print("Computing type embeddings...")
        type_embeddings = self._get_embeddings(formatted_types)  # [n_types x dim]
        
        # Transform type embeddings through FFN to get relation embeddings
        with torch.no_grad():
            relation_embeddings = self.ffn(type_embeddings)  # [n_types x dim]
        
        # Compute DistMult scores
        print("Computing DistMult scores...")
        scores = self._compute_distmult_scores(
            term_embeddings, relation_embeddings, type_embeddings
        ).cpu().numpy()
        
        # Make predictions with adaptive thresholding
        predictions = []
        for i, term_data in enumerate(terms_data):
            term_scores = scores[i]
            selected_indices = self._select_types_adaptive(term_scores)
            selected_types = [types[idx] for idx in selected_indices]
            
            predictions.append({
                "id": term_data["id"],
                "types": selected_types
            })
        
        # Save predictions
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        
        print(f"Predictions saved to {output_file}")

def run_pipeline(data_dir: str, output_dir: str):
    # Get domain name from data_dir
    domain = Path(data_dir).name  # Get B4_blind from B4_blind
    domain_short = domain.split('_')[0]  # Get B4 from B4_blind
    
    # Load data with correct file names
    with open(Path(data_dir) / f"{domain_short}-Blind-Terms.json", 'r', encoding='utf-8') as f:
        terms_data = json.load(f)
    
    with open(Path(data_dir) / f"{domain_short}-Blind-Types.txt", 'r', encoding='utf-8') as f:
        types = [line.strip() for line in f if line.strip()]
    
    # Initialize model
    model = CustomZeroShotDistMultWithFFN()
    
    # Run prediction
    print("\nRunning Custom DistMult with FFN...")
    model.predict(
        terms_data=terms_data,
        types=types,
        output_file=str(Path(output_dir) / "predictions_zs_distmult_custom.json")
    )

if __name__ == "__main__":
    # Example usage
    data_dir = "/path/to/data"
    output_dir = "/path/to/output"
    run_pipeline(data_dir, output_dir) 