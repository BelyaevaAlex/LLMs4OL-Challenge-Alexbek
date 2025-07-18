import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from cosine_similarity_zs import CosineSimilarityZeroShot, last_token_pool

# Domain-specific prompts for different datasets
DOMAIN_PROMPTS = {
    "B4": {  # Earth sciences and geography
        "term_prefix": "In Earth sciences and geography, explain the concept of",
        "type_prefix": "This Earth science category represents",
        "instruction": "Analyze this geographical or Earth science term and explain its key features"
    },
    "B5": {  # Linguistics and grammar
        "term_prefix": "In linguistics and grammar, define the term",
        "type_prefix": "This linguistic category encompasses",
        "instruction": "Analyze this linguistic or grammatical term and explain its function"
    },
    "B6": {  # Units and measurements
        "term_prefix": "In the context of units and measurements, describe",
        "type_prefix": "This measurement-related category represents",
        "instruction": "Analyze this unit or measurement-related term and explain its purpose"
    }
}

class EnsembleZeroShot:
    def __init__(
        self,
        device: str = "cuda",
        confidence_threshold: float = 0.7,
        use_domain_prompts: bool = True
    ):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.use_domain_prompts = use_domain_prompts
        
        # Initialize base models
        print("Initializing MPNet model...")
        self.mpnet = CosineSimilarityZeroShot(
            model_name="sentence-transformers/all-mpnet-base-v2",
            device=device,
            use_qwen=False
        )
        
        print("Initializing Qwen model...")
        self.qwen = CosineSimilarityZeroShot(
            model_name="Qwen/Qwen3-Embedding-4B",
            device=device,
            use_qwen=True
        )
        
        print("Initializing BGE model...")
        self.bge = SentenceTransformer('BAAI/bge-large-en-v1.5').to(device)
        
        # Initialize default weights
        self.default_weights = {
            'mpnet': 0.3,
            'qwen': 0.4,
            'bge': 0.3
        }

    def _get_bge_embeddings(self, texts: List[str], is_query: bool = True) -> np.ndarray:
        """Get embeddings using BGE model"""
        # Add instruction prefix for queries
        if is_query:
            texts = [f"Represent this sentence for retrieval: {text}" for text in texts]
        
        # Compute embeddings
        with torch.no_grad():
            embeddings = self.bge.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_tensor=True
            )
        return embeddings.cpu().numpy()

    def _get_confidence_scores(
        self,
        similarity_matrix: np.ndarray,
        temperature: float = 1.0
    ) -> Tuple[np.ndarray, float]:
        """
        Calculate confidence scores based on similarity distribution
        Returns both the confidence score and the entropy of the distribution
        """
        # Apply softmax with temperature
        probs = F.softmax(torch.tensor(similarity_matrix) / temperature, dim=-1).numpy()
        
        # Calculate max probability and entropy
        max_prob = np.max(probs, axis=-1)
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=-1)
        
        # Normalize entropy to [0, 1]
        entropy_normalized = entropy / np.log(probs.shape[-1])
        
        # Combine max probability and entropy for confidence score
        confidence = max_prob * (1 - entropy_normalized)
        
        return confidence, np.mean(entropy)

    def _adjust_weights(
        self,
        confidences: Dict[str, np.ndarray],
        entropies: Dict[str, float]
    ) -> Dict[str, float]:
        """Adjust model weights based on confidence scores and entropies"""
        # Calculate confidence-based weights
        total_confidence = sum(np.mean(conf) for conf in confidences.values())
        weights = {
            model: np.mean(conf) / total_confidence 
            for model, conf in confidences.items()
        }
        
        # Adjust weights based on entropy
        total_entropy = sum(entropies.values())
        entropy_weights = {
            model: (1 - entropy / total_entropy) 
            for model, entropy in entropies.items()
        }
        
        # Combine both factors
        final_weights = {
            model: 0.7 * weights[model] + 0.3 * entropy_weights[model]
            for model in weights
        }
        
        # Normalize weights
        total_weight = sum(final_weights.values())
        final_weights = {
            model: weight / total_weight 
            for model, weight in final_weights.items()
        }
        
        return final_weights

    def _get_domain_specific_prompts(self, domain: str, terms: List[str], types: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """Apply domain-specific prompts to terms and types"""
        if not self.use_domain_prompts or domain not in DOMAIN_PROMPTS:
            return terms, types, types
        
        prompts = DOMAIN_PROMPTS[domain]
        formatted_terms = [f"{prompts['term_prefix']} {term}" for term in terms]
        formatted_types = [f"{prompts['type_prefix']} {type_}" for type_ in types]
        
        return formatted_terms, formatted_types, types

    def predict(
        self,
        terms_data: List[Dict[str, Any]],
        types: List[str],
        output_file: str,
        domain: str = None
    ) -> None:
        # Extract terms
        terms = [item["term"] for item in terms_data]
        
        # Apply domain-specific prompts if enabled
        if self.use_domain_prompts and domain:
            terms_prompted, types_prompted, original_types = self._get_domain_specific_prompts(domain, terms, types)
        else:
            terms_prompted, types_prompted, original_types = terms, types, types
        
        # Get predictions from each model
        predictions = {}
        confidences = {}
        entropies = {}
        
        # MPNet predictions
        print("\nComputing MPNet predictions...")
        mpnet_similarities = cosine_similarity(
            self.mpnet._get_embeddings(terms_prompted, is_query=True),
            self.mpnet._get_embeddings(types_prompted, is_query=True)
        )
        confidences['mpnet'], entropies['mpnet'] = self._get_confidence_scores(mpnet_similarities)
        predictions['mpnet'] = mpnet_similarities
        
        # Qwen predictions
        print("\nComputing Qwen predictions...")
        qwen_similarities = cosine_similarity(
            self.qwen._get_embeddings(terms_prompted, is_query=True),
            self.qwen._get_embeddings(types_prompted, is_query=True)
        )
        confidences['qwen'], entropies['qwen'] = self._get_confidence_scores(qwen_similarities)
        predictions['qwen'] = qwen_similarities
        
        # BGE predictions
        print("\nComputing BGE predictions...")
        bge_similarities = cosine_similarity(
            self._get_bge_embeddings(terms_prompted),
            self._get_bge_embeddings(types_prompted)
        )
        confidences['bge'], entropies['bge'] = self._get_confidence_scores(bge_similarities)
        predictions['bge'] = bge_similarities
        
        # Adjust weights based on confidences
        weights = self._adjust_weights(confidences, entropies)
        print("\nAdjusted weights:", weights)
        
        # Combine predictions
        ensemble_predictions = sum(
            weights[model] * pred_matrix
            for model, pred_matrix in predictions.items()
        )
        
        # Make final predictions
        final_predictions = []
        detailed_predictions = []
        
        for i, term_data in enumerate(terms_data):
            scores = ensemble_predictions[i]
            selected_index = np.argmax(scores)
            
            # Save only required fields in the output using original type
            final_predictions.append({
                "id": term_data["id"],
                "types": [original_types[selected_index]]
            })
            
            # Save detailed predictions for analysis
            detailed_predictions.append({
                "id": term_data["id"],
                "types": [original_types[selected_index]],
                "confidence": float(scores[selected_index]),
                "model_weights": weights,
                "individual_predictions": {
                    "mpnet": original_types[np.argmax(predictions['mpnet'][i])],
                    "qwen": original_types[np.argmax(predictions['qwen'][i])],
                    "bge": original_types[np.argmax(predictions['bge'][i])]
                }
            })
        
        # Save main predictions in required format
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_predictions, f, indent=2, ensure_ascii=False)
        
        # Save detailed analysis
        analysis_file = output_file.replace('.json', '_analysis.json')
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_predictions, f, indent=2, ensure_ascii=False)
        
        print(f"\nPredictions saved to {output_file}")
        print(f"Detailed analysis saved to {analysis_file}")

def run_ensemble_pipeline(data_dir: str, output_dir: str):
    # Get domain name from data_dir
    domain = Path(data_dir).name  # Get B4_blind from B4_blind
    domain_short = domain.split('_')[0]  # Get B4 from B4_blind
    
    # Load data
    with open(Path(data_dir) / f"{domain_short}-Blind-Terms.json", 'r', encoding='utf-8') as f:
        terms_data = json.load(f)
    
    with open(Path(data_dir) / f"{domain_short}-Blind-Types.txt", 'r', encoding='utf-8') as f:
        types = [line.strip() for line in f if line.strip()]
    
    # Initialize ensemble model
    print(f"\nInitializing Ensemble model for domain {domain}...")
    model = EnsembleZeroShot(use_domain_prompts=True)
    
    # Run prediction
    model.predict(
        terms_data=terms_data,
        types=types,
        output_file=str(Path(output_dir) / "predictions_ensemble.json"),
        domain=domain
    ) 