"""
Inference script for Cross-Attention model
Applies trained model to new data for building term hierarchies
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModel
import argparse
from pathlib import Path
import logging
from datetime import datetime
import tqdm
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Import model and utilities
from cross_attention_model import CrossAttentionModel


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Extract last token embeddings with proper handling of padding"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def setup_logging(log_file: str = None) -> logging.Logger:
    """
    Setup logging
    
    Args:
        log_file: path to log file
        
    Returns:
        logger: configured logger
    """
    logger = logging.getLogger("cross_attention_inference")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Log formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_embedding_model():
    """
    Load model for creating embeddings
    
    Returns:
        model, tokenizer: model and tokenizer for embeddings
    """
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


def get_term_embeddings_mean(terms: List[str], model, tokenizer, batch_size: int = 32) -> List[List[float]]:
    """
    Generate embeddings for list of terms using mean pooling (old method)
    
    Args:
        terms: list of terms
        model: model for embeddings
        tokenizer: tokenizer
        batch_size: batch size
        
    Returns:
        embeddings: list of embeddings
    """
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(terms), batch_size), desc="Generating embeddings (mean)"):
            batch_terms = terms[i:i+batch_size]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_terms, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Get embeddings
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
            embeddings.extend(batch_embeddings.tolist())
    
    return embeddings


def get_term_embeddings_pooling(terms: List[str], model, tokenizer, batch_size: int = 32) -> List[List[float]]:
    """
    Generate embeddings for list of terms using last token pooling with normalization (new method)
    
    Args:
        terms: list of terms
        model: model for embeddings
        tokenizer: tokenizer
        batch_size: batch size
        
    Returns:
        embeddings: list of embeddings
    """
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(terms), batch_size), desc="Generating embeddings (pooling)"):
            batch_terms = terms[i:i+batch_size]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_terms, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Get embeddings
            outputs = model(**inputs)
            # Use last_token_pool to extract embeddings
            batch_embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            # Apply L2 normalization
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            batch_embeddings = batch_embeddings.cpu().numpy()
            
            embeddings.extend(batch_embeddings.tolist())
    
    return embeddings


def get_term_embeddings(terms: List[str], model, tokenizer, batch_size: int = 32, method: str = "pooling") -> List[List[float]]:
    """
    Generate embeddings for list of terms
    
    Args:
        terms: list of terms
        model: model for embeddings
        tokenizer: tokenizer
        batch_size: batch size
        method: embedding extraction method ("pooling" or "mean")
        
    Returns:
        embeddings: list of embeddings
    """
    if method == "pooling" or method == "pool":
        return get_term_embeddings_pooling(terms, model, tokenizer, batch_size)
    else:
        return get_term_embeddings_mean(terms, model, tokenizer, batch_size)


def load_trained_model(results_dir: str) -> Tuple[CrossAttentionModel, Dict, float]:
    """
    Load trained model and information about best results
    
    Args:
        results_dir: directory with training results
        
    Returns:
        model: loaded model
        best_results: information about best results (combined from two files)
        f1_threshold: best threshold for F1 score
    """
    results_path = Path(results_dir)
    
    # 1. Load best model information (correct thresholds)
    best_model_results_file = results_path / "best_models" / "best_model_evaluation_summary.json"
    if not best_model_results_file.exists():
        raise FileNotFoundError(f"best_model_evaluation_summary.json not found in {results_dir}/best_models/")
    
    with open(best_model_results_file, 'r') as f:
        best_model_results = json.load(f)
    
    # 2. Load general experiment information (including dataset)
    experiment_results_file = results_path / "best_results.json"
    if not experiment_results_file.exists():
        raise FileNotFoundError(f"best_results.json not found in {results_dir}")
    
    with open(experiment_results_file, 'r') as f:
        experiment_results = json.load(f)
    
    # 3. Combine information from two files
    try:
        best_results = {
            # Take correct results and thresholds from best model
            'best_results': best_model_results['best_results'],
            # Take experiment information from general file
            'experiment_info': experiment_results['experiment_info'],
            # Add best model information
            'best_model_info': best_model_results.get('best_model_info', {})
        }
    except KeyError:
        best_results = {
            'best_results': experiment_results['best_results'],
            'experiment_info': experiment_results['experiment_info'],
            'best_model_info': best_model_results.get('best_model_info', {})
        }
    
    try:
        # Get best F1 threshold from best model
        f1_threshold = best_model_results['best_results']['best_thresholds']['f1_score']['threshold']
    except KeyError:
        try:
            f1_threshold = best_results['best_thresholds']['f1']
        except KeyError:
            f1_threshold = experiment_results['best_results']['best_thresholds']['f1_score']['threshold']
    
    # Load best model information
    best_models_file = results_path / "best_models" / "best_models.json"
    if not best_models_file.exists():
        raise FileNotFoundError(f"best_models.json not found in {results_dir}/best_models/")
    
    with open(best_models_file, 'r') as f:
        best_models_info = json.load(f)
    
    # Select best model (first in list)
    best_model_info = best_models_info['best_models'][0]
    model_path = best_model_info['model_path']
    
    if not os.path.exists(model_path):
        # Try relative path
        model_filename = best_model_info['model_filename']
        model_path = results_path / "best_models" / model_filename
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model with PyTorch 2.6 support
    try:
        # First try to load safely
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    except Exception:
        # If not successful, use old method (for older models)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Create model
    config = checkpoint['config']
    # Ensure config is a dictionary
    if not isinstance(config, dict):
        if hasattr(config, 'to_dict'):
            config = config.to_dict()
        else:
            # If it's an object, convert its attributes to a dictionary
            config = {k: v for k, v in config.__dict__.items() 
                     if not k.startswith('_')}
    
    model = CrossAttentionModel(config, layer_idx=checkpoint.get('layer_idx', 0))
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model config: ", config)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    model.eval()
    
    return model, best_results, f1_threshold


def build_prediction_matrix(
    embeddings: List[List[float]], 
    model: CrossAttentionModel, 
    batch_size: int = 64,
    logger: logging.Logger = None
) -> np.ndarray:
    """
    Build full prediction matrix terms x terms
    
    Args:
        embeddings: list of term embeddings
        model: trained model
        batch_size: batch size for processing
        logger: logger for progress output
        
    Returns:
        pred_matrix: prediction matrix of size (n_terms, n_terms)
    """
    n_terms = len(embeddings)
    pred_matrix = np.zeros((n_terms, n_terms), dtype=np.float32)
    
    # Convert embeddings to tensor
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    if torch.cuda.is_available():
        embeddings_tensor = embeddings_tensor.cuda()
    
    if logger:
        logger.info(f"Building prediction matrix {n_terms}x{n_terms}")
    
    with torch.no_grad():
        # Process in blocks to save memory
        for i in tqdm.tqdm(range(0, n_terms, batch_size), desc="Building prediction matrix"):
            end_i = min(i + batch_size, n_terms)
            batch_embeddings_1 = embeddings_tensor[i:end_i]  # (batch_size, embedding_dim)
            
            # For each block of rows, calculate predictions for all terms
            for j in range(0, n_terms, batch_size):
                end_j = min(j + batch_size, n_terms)
                batch_embeddings_2 = embeddings_tensor[j:end_j]  # (batch_size, embedding_dim)
                
                # Calculate predictions for the block
                # batch_embeddings_1: (batch_i, embedding_dim)
                # batch_embeddings_2: (batch_j, embedding_dim)
                # Need to get: (batch_i, batch_j)
                
                batch_pred = model(batch_embeddings_1, batch_embeddings_2)
                pred_matrix[i:end_i, j:end_j] = batch_pred.cpu().numpy()
    
    return pred_matrix


def apply_threshold_and_extract_pairs(
    pred_matrix: np.ndarray,
    terms: List[str],
    threshold: float,
    logger: logging.Logger = None
) -> List[Dict[str, str]]:
    """
    Apply threshold to prediction matrix and extract child-parent pairs
    
    Args:
        pred_matrix: prediction matrix
        terms: list of terms
        threshold: threshold for binarization
        logger: logger
        
    Returns:
        pairs: list of pairs in format [{"parent": "...", "child": "..."}]
    """
    pairs = []
    
    # Apply threshold
    binary_matrix = (pred_matrix > threshold).astype(int)
    
    # Extract pairs (i, j) where binary_matrix[i, j] = 1
    # i - child index, j - parent index
    child_indices, parent_indices = np.where(binary_matrix == 1)
    
    for child_idx, parent_idx in zip(child_indices, parent_indices):
        # Skip self-connections
        if child_idx == parent_idx:
            continue
            
        pairs.append({
            "parent": terms[parent_idx],
            "child": terms[child_idx]
        })
    
    if logger:
        logger.info(f"Extracted {len(pairs)} pairs with threshold {threshold}")
        
        # Matrix statistics
        total_pairs = len(terms) * (len(terms) - 1)  # Without diagonal
        positive_pairs = len(pairs)
        logger.info(f"Total possible pairs: {total_pairs}")
        logger.info(f"Positive pairs: {positive_pairs} ({positive_pairs/total_pairs*100:.2f}%)")
    
    return pairs


def save_results(
    pred_matrix: np.ndarray,
    pairs: List[Dict[str, str]],
    terms: List[str],
    output_dir: str,
    threshold: float,
    best_results: Dict,
    logger: logging.Logger = None
):
    """
    Save inference results
    
    Args:
        pred_matrix: prediction matrix
        pairs: child-parent pairs
        terms: list of terms
        output_dir: directory to save
        threshold: used threshold
        best_results: information about best results
        logger: logger
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save prediction matrix
    matrix_file = output_path / "prediction_matrix.npy"
    np.save(matrix_file, pred_matrix)
    
    # Save pairs in JSON
    pairs_file = output_path / "predicted_pairs.json"
    with open(pairs_file, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    
    # Save terms
    terms_file = output_path / "terms.json"
    with open(terms_file, 'w', encoding='utf-8') as f:
        json.dump(terms, f, ensure_ascii=False, indent=2)
    
    # Save metadata
    metadata = {
        "inference_info": {
            "timestamp": datetime.now().isoformat(),
            "n_terms": len(terms),
            "n_pairs": len(pairs),
            "threshold_used": threshold,
            "matrix_shape": pred_matrix.shape,
            "matrix_stats": {
                "min": float(pred_matrix.min()),
                "max": float(pred_matrix.max()),
                "mean": float(pred_matrix.mean()),
                "std": float(pred_matrix.std())
            }
        },
        "model_info": {
            "best_f1_threshold": best_results['best_results']['best_thresholds']['f1_score']['threshold'],
            "best_f1_value": best_results['best_results']['best_thresholds']['f1_score']['value'],
            "model_roc_auc": best_results['best_results']['roc_auc'],
            "dataset": best_results['experiment_info']['dataset']
        }
    }
    
    metadata_file = output_path / "inference_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # Plot prediction distribution
    plot_prediction_distribution(pred_matrix, threshold, output_path)
    
    if logger:
        logger.info(f"Results saved to {output_dir}")
        logger.info(f"  📊 Prediction matrix: {matrix_file}")
        logger.info(f"  🔗 Pairs: {pairs_file}")
        logger.info(f"  📝 Terms: {terms_file}")
        logger.info(f"  📋 Metadata: {metadata_file}")


def plot_prediction_distribution(pred_matrix: np.ndarray, threshold: float, output_dir: Path, max_samples: int = 1000000):
    """
    Plot prediction distribution with sampling support for large matrices
    
    Args:
        pred_matrix: prediction matrix
        threshold: used threshold
        output_dir: directory to save
        max_samples: maximum number of samples for analysis (default 1M)
    """
    plt.figure(figsize=(12, 8))
    
    # Calculate number of off-diagonal elements
    n = pred_matrix.shape[0]
    total_off_diagonal = n * (n - 1)
    
    # Decide whether to use sampling
    use_sampling = total_off_diagonal > max_samples
    
    if use_sampling:
        print(f"🔄 Using sampling: {max_samples:,} from {total_off_diagonal:,} elements ({max_samples/total_off_diagonal*100:.1f}%)")
        
        # Generate random indices for sampling
        np.random.seed(42)  # For reproducibility
        
        # Generate random pairs (i, j) where i != j
        sample_indices = []
        samples_per_batch = min(max_samples // 10, 100000)  # Generate in batches
        
        while len(sample_indices) < max_samples:
            # Generate random indices
            i_indices = np.random.randint(0, n, samples_per_batch)
            j_indices = np.random.randint(0, n, samples_per_batch)
            
            # Filter only off-diagonal elements
            mask = i_indices != j_indices
            valid_pairs = list(zip(i_indices[mask], j_indices[mask]))
            
            sample_indices.extend(valid_pairs)
            
            if len(sample_indices) >= max_samples:
                sample_indices = sample_indices[:max_samples]
                break
        
        # Extract values by sampled indices
        sample_i, sample_j = zip(*sample_indices)
        off_diagonal = pred_matrix[sample_i, sample_j]
        
        sampling_info = f"Sampling: {len(off_diagonal):,} from {total_off_diagonal:,} elements"
        
    else:
        print(f"📊 Analyzing all elements: {total_off_diagonal:,}")
        
        # Standard approach for small matrices
        mask = np.eye(pred_matrix.shape[0], dtype=bool)
        off_diagonal = pred_matrix[~mask]
        
        sampling_info = f"Full analysis: {len(off_diagonal):,} elements"
    
    # Histogram of predictions
    plt.subplot(2, 2, 1)
    plt.hist(off_diagonal, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.3f}')
    plt.xlabel('Prediction Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Logarithmic scale
    plt.subplot(2, 2, 2)
    plt.hist(off_diagonal, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.3f}')
    plt.xlabel('Prediction Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Scores (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Cumulative distribution
    plt.subplot(2, 2, 3)
    sorted_scores = np.sort(off_diagonal)
    y_vals = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    plt.plot(sorted_scores, y_vals, linewidth=2)
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.3f}')
    plt.xlabel('Prediction Score')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution of Prediction Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Statistics
    plt.subplot(2, 2, 4)
    stats_text = f"""
    Matrix Statistics:
    
    Shape: {pred_matrix.shape}
    Total pairs: {total_off_diagonal:,}
    {sampling_info}
    
    Score Distribution:
    Min: {off_diagonal.min():.4f}
    Max: {off_diagonal.max():.4f}
    Mean: {off_diagonal.mean():.4f}
    Std: {off_diagonal.std():.4f}
    
    Threshold: {threshold:.3f}
    Above threshold: {np.sum(off_diagonal > threshold):,} ({np.sum(off_diagonal > threshold)/len(off_diagonal)*100:.2f}%)
    Below threshold: {np.sum(off_diagonal <= threshold):,} ({np.sum(off_diagonal <= threshold)/len(off_diagonal)*100:.2f}%)
    """
    
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=10, fontfamily='monospace')
    plt.axis('off')
    
    plt.tight_layout()
    plot_path = output_dir / 'prediction_distribution.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Prediction distribution plot saved: {plot_path}")


def load_existing_inference_results(results_dir: str) -> Tuple[np.ndarray, List[str], Dict, float]:
    """
    Load already existing inference results
    
    Args:
        results_dir: directory with existing inference results
        
    Returns:
        pred_matrix: prediction matrix
        terms: list of terms
        metadata: inference metadata
        original_threshold: originally used threshold
    """
    results_path = Path(results_dir)
    
    # Load matrix
    matrix_file = results_path / "prediction_matrix.npy"
    if not matrix_file.exists():
        raise FileNotFoundError(f"Prediction matrix not found: {matrix_file}")
    
    pred_matrix = np.load(matrix_file)
    
    # Load terms
    terms_file = results_path / "terms.json"
    if not terms_file.exists():
        raise FileNotFoundError(f"Terms file not found: {terms_file}")
    
    with open(terms_file, 'r', encoding='utf-8') as f:
        terms = json.load(f)
    
    # Load metadata
    metadata_file = results_path / "inference_metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_file}")
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    original_threshold = metadata['inference_info']['threshold_used']
    
    return pred_matrix, terms, metadata, original_threshold


def run_threshold_experiment(
    existing_results_dir: str,
    new_threshold: float,
    output_dir: str,
    output_suffix: str = None,
    log_file: str = None
) -> Dict:
    """
    Apply new threshold to already existing prediction matrix
    
    Args:
        existing_results_dir: directory with existing inference results
        new_threshold: new threshold for extracting pairs
        output_dir: directory to save new results
        output_suffix: suffix for file names (e.g., "_threshold_05")
        log_file: log file
        
    Returns:
        results: dictionary with results
    """
    # Setup logging
    logger = setup_logging(log_file)
    
    logger.info("🎯 Starting threshold experiment")
    logger.info(f"📁 Existing results: {existing_results_dir}")
    logger.info(f"🎚️ New threshold: {new_threshold}")
    logger.info(f"💾 Output directory: {output_dir}")
    
    try:
        # 1. Load existing results
        logger.info("📖 Loading existing results...")
        pred_matrix, terms, metadata, original_threshold = load_existing_inference_results(existing_results_dir)
        
        logger.info(f"✅ Results loaded:")
        logger.info(f"   Matrix: {pred_matrix.shape}")
        logger.info(f"   Terms: {len(terms)}")
        logger.info(f"   Original threshold: {original_threshold:.3f}")
        logger.info(f"   New threshold: {new_threshold:.3f}")
        
        # 2. Apply new threshold
        logger.info("🎯 Applying new threshold...")
        pairs = apply_threshold_and_extract_pairs(pred_matrix, terms, new_threshold, logger)
        
        # 3. Create updated metadata
        updated_metadata = metadata.copy()
        updated_metadata['threshold_experiment'] = {
            'original_threshold': original_threshold,
            'new_threshold': new_threshold,
            'original_results_dir': existing_results_dir,
            'experiment_timestamp': datetime.now().isoformat()
        }
        updated_metadata['inference_info']['threshold_used'] = new_threshold
        
        # 4. Save results with new threshold
        logger.info("💾 Saving results with new threshold...")
        
        # Determine suffix for files
        if output_suffix is None:
            output_suffix = f"_threshold_{new_threshold:.3f}".replace(".", "")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save pairs
        pairs_file = output_path / f"predicted_pairs{output_suffix}.json"
        with open(pairs_file, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)
        
        # Save terms (copy)
        terms_file = output_path / f"terms{output_suffix}.json"
        with open(terms_file, 'w', encoding='utf-8') as f:
            json.dump(terms, f, ensure_ascii=False, indent=2)
        
        # Save matrix (create symbolic link or copy)
        matrix_file = output_path / f"prediction_matrix{output_suffix}.npy"
        if not matrix_file.exists():
            original_matrix_file = Path(existing_results_dir) / "prediction_matrix.npy"
            try:
                # Try to create a symbolic link (saves space)
                matrix_file.symlink_to(original_matrix_file.absolute())
                logger.info(f"   Created symbolic link to matrix: {matrix_file}")
            except (OSError, NotImplementedError):
                # If not successful, copy the file
                import shutil
                shutil.copy2(original_matrix_file, matrix_file)
                logger.info(f"   Copied matrix: {matrix_file}")
        
        # Save updated metadata
        metadata_file = output_path / f"inference_metadata{output_suffix}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(updated_metadata, f, ensure_ascii=False, indent=2)
        
        # Plot distribution with new threshold
        logger.info("📊 Plotting distribution...")
        plot_prediction_distribution(pred_matrix, new_threshold, output_path)
        
        logger.info("✅ Threshold experiment completed successfully!")
        logger.info(f"📊 Results:")
        logger.info(f"   Original threshold: {original_threshold:.3f}")
        logger.info(f"   New threshold: {new_threshold:.3f}")
        logger.info(f"   Original number of pairs: {len(json.load(open(Path(existing_results_dir) / 'predicted_pairs.json')))}")
        logger.info(f"   New number of pairs: {len(pairs)}")
        logger.info(f"   Difference: {len(pairs) - len(json.load(open(Path(existing_results_dir) / 'predicted_pairs.json')))}")
        logger.info(f"   Results directory: {output_dir}")
        
        # Return results
        return {
            'pred_matrix': pred_matrix,
            'pairs': pairs,
            'terms': terms,
            'threshold': new_threshold,
            'original_threshold': original_threshold,
            'metadata': updated_metadata,
            'statistics': {
                'total_terms': len(terms),
                'total_pairs': len(pairs),
                'original_pairs': len(json.load(open(Path(existing_results_dir) / 'predicted_pairs.json'))),
                'pairs_difference': len(pairs) - len(json.load(open(Path(existing_results_dir) / 'predicted_pairs.json'))),
                'matrix_shape': pred_matrix.shape,
                'matrix_min': float(pred_matrix.min()),
                'matrix_max': float(pred_matrix.max()),
                'matrix_mean': float(pred_matrix.mean()),
                'matrix_std': float(pred_matrix.std())
            },
            'output_dir': output_dir,
            'files_created': {
                'pairs': str(pairs_file),
                'terms': str(terms_file),
                'matrix': str(matrix_file),
                'metadata': str(metadata_file),
                'distribution_plot': str(output_path / 'prediction_distribution.png')
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error during threshold experiment: {e}")
        raise


def run_inference(
    results_dir: str,
    terms_file: str,
    output_dir: str,
    embedding_batch_size: int = 32,
    prediction_batch_size: int = 64,
    custom_threshold: float = None,
    log_file: str = None,
    embedding_method: str = "pooling",
    # New parameters for working with a pre-built matrix
    existing_results_dir: str = None,
    threshold_only: bool = False
) -> Dict:
    """
    Main inference function for use in Jupyter notebook
    
    Args:
        results_dir: Directory with training results (only needed if existing_results_dir is None)
        terms_file: Path to .txt file with terms (only needed if existing_results_dir is None)
        output_dir: Directory to save inference results
        embedding_batch_size: Batch size for embedding generation
        prediction_batch_size: Batch size for prediction matrix construction
        custom_threshold: Custom threshold (if None, uses best F1 threshold from training)
        log_file: Path to log file (optional)
        embedding_method: Method for extracting embeddings ("pooling" or "mean")
        existing_results_dir: Directory with existing inference results (optional)
        threshold_only: Whether to return only the threshold and statistics (optional)
        
    Returns:
        results: Dictionary with inference results and statistics
    """
    # Setup logging
    logger = setup_logging(log_file)
    
    # Mode 1: Work with pre-built matrix (only new threshold)
    if existing_results_dir is not None:
        logger.info("🎯 Working with pre-built matrix")
        
        # Load trained model to get threshold (if needed)
        if custom_threshold is None:
            logger.info("🔄 Loading model to get threshold...")
            _, _, f1_threshold = load_trained_model(results_dir)
            threshold = f1_threshold
        else:
            threshold = custom_threshold
        
        # Run threshold experiment
        return run_threshold_experiment(
            existing_results_dir=existing_results_dir,
            new_threshold=threshold,
            output_dir=output_dir,
            log_file=log_file
        )
    
    # Mode 2: Full inference
    logger.info("🚀 Starting full Cross-Attention model inference")
    logger.info(f"📁 Results directory: {results_dir}")
    logger.info(f"📝 Terms file: {terms_file}")
    logger.info(f"💾 Output directory: {output_dir}")
    
    try:
        # 1. Load trained model
        logger.info("🔄 Loading trained model...")
        model, best_results, f1_threshold = load_trained_model(results_dir)
        
        # Choose threshold
        threshold = custom_threshold if custom_threshold is not None else f1_threshold
        
        logger.info(f"✅ Model loaded:")
        logger.info(f"   ROC AUC: {best_results['best_results']['roc_auc']:.4f}")
        logger.info(f"   Best F1 threshold: {f1_threshold:.3f}")
        logger.info(f"   Using threshold: {threshold:.3f}")
        
        # 2. Read terms
        logger.info("📖 Reading terms...")
        with open(terms_file, 'r', encoding='utf-8') as f:
            terms = [line.strip() for line in f if line.strip()]
        
        logger.info(f"✅ Loaded {len(terms)} terms")
        
        # 3. Generate embeddings
        logger.info("🔄 Loading embedding model...")
        embedding_model, tokenizer = load_embedding_model()
        
        logger.info(f"⚡ Generating embeddings (method: {embedding_method})...")
        embeddings = get_term_embeddings(terms, embedding_model, tokenizer, embedding_batch_size, embedding_method)
        
        logger.info(f"✅ Embeddings created: {len(embeddings)} x {len(embeddings[0])}")
        
        # Free memory from embedding model
        del embedding_model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 4. Build prediction matrix
        logger.info("🔮 Building prediction matrix...")
        pred_matrix = build_prediction_matrix(embeddings, model, prediction_batch_size, logger)
        
        logger.info(f"✅ Prediction matrix built: {pred_matrix.shape}")
        logger.info(f"   Statistics: min={pred_matrix.min():.4f}, max={pred_matrix.max():.4f}, mean={pred_matrix.mean():.4f}")
        
        # 5. Apply threshold and extract pairs
        logger.info("�� Applying threshold and extracting pairs...")
        pairs = apply_threshold_and_extract_pairs(pred_matrix, terms, threshold, logger)
        
        # 6. Save results
        logger.info("💾 Saving results...")
        save_results(pred_matrix, pairs, terms, output_dir, threshold, best_results, logger)
        
        logger.info("✅ Inference completed successfully!")
        logger.info(f"📊 Results:")
        logger.info(f"   Total terms: {len(terms)}")
        logger.info(f"   Found pairs: {len(pairs)}")
        logger.info(f"   Used threshold: {threshold:.3f}")
        logger.info(f"   Results directory: {output_dir}")
        
        # Return results for further analysis
        return {
            'pred_matrix': pred_matrix,
            'pairs': pairs,
            'terms': terms,
            'threshold': threshold,
            'best_results': best_results,
            'model_info': {
                'roc_auc': best_results['best_results']['roc_auc'],
                'f1_threshold': f1_threshold,
                'dataset': best_results['experiment_info']['dataset']
            },
            'statistics': {
                'total_terms': len(terms),
                'total_pairs': len(pairs),
                'matrix_shape': pred_matrix.shape,
                'matrix_min': float(pred_matrix.min()),
                'matrix_max': float(pred_matrix.max()),
                'matrix_mean': float(pred_matrix.mean()),
                'matrix_std': float(pred_matrix.std())
            },
            'output_dir': output_dir
        }
        
    except Exception as e:
        logger.error(f"❌ Error during inference: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Cross-Attention Model Inference")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory with training results (e.g., results/20250707_192128_MatOnto_...)")
    parser.add_argument("--terms_file", type=str, required=True,
                        help="Path to .txt file with terms")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save inference results")
    parser.add_argument("--embedding_batch_size", type=int, default=32,
                        help="Batch size for embedding generation")
    parser.add_argument("--prediction_batch_size", type=int, default=64,
                        help="Batch size for prediction matrix construction")
    parser.add_argument("--custom_threshold", type=float, default=None,
                        help="Custom threshold (if not provided, uses best F1 threshold from training)")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Path to log file")
    parser.add_argument("--embedding_method", type=str, default="pooling", 
                        choices=["pooling", "mean"],
                        help="Method for extracting embeddings: pooling (last token + normalization) or mean")
    parser.add_argument("--existing_results_dir", type=str, default=None,
                        help="Directory with existing inference results (optional)")
    parser.add_argument("--threshold_only", action="store_true",
                        help="Whether to return only the threshold and statistics")
    
    args = parser.parse_args()
    
    # Call main inference function
    results = run_inference(
        results_dir=args.results_dir,
        terms_file=args.terms_file,
        output_dir=args.output_dir,
        embedding_batch_size=args.embedding_batch_size,
        prediction_batch_size=args.prediction_batch_size,
        custom_threshold=args.custom_threshold,
        log_file=args.log_file,
        embedding_method=args.embedding_method,
        existing_results_dir=args.existing_results_dir,
        threshold_only=args.threshold_only
    )
    
    # Print brief summary
    print(f"\n📈 Brief summary:")
    print(f"   Terms: {results['statistics']['total_terms']}")
    print(f"   Pairs: {results['statistics']['total_pairs']}")
    print(f"   Threshold: {results['threshold']:.3f}")
    print(f"   Model ROC AUC: {results['model_info']['roc_auc']:.4f}")
    print(f"   Results: {results['output_dir']}")


if __name__ == "__main__":
    main() 