#!/usr/bin/env python3
"""Inference script for universal terms and types extraction model with structured output support."""

import json
import argparse
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from fuzzywuzzy import fuzz
from pydantic import BaseModel, Field

# Required outlines import for structured output
try:
    import outlines
    from outlines.models import Transformers
    from outlines.generate import json as generate_json
    OUTLINES_AVAILABLE = True
except ImportError:
    OUTLINES_AVAILABLE = False
    print("ERROR: outlines library is required for structured output but not available.")
    print("Please install it with: pip install outlines")
    import sys
    sys.exit(1)

from .data import (
    load_few_shot_examples, 
    load_docs2terms_types_dataset,
    build_conversation_for_inference,
    extract_data_from_document,
    SYSTEM_PROMPT
)

def load_jsonl(file_path):
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add file handler
file_handler = logging.FileHandler('inference.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Add console handler if not already present
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


class ExtractedTermsAndTypes(BaseModel):
    """Schema for extracted terms and types output"""
    terms: List[str] = Field(description="List of extracted ontology terms from the document")
    types: List[str] = Field(description="List of extracted ontology types from the document")


def extract_terms_and_types_from_generated_text(text: str) -> Tuple[List[str], List[str]]:
    """Extract terms and types from generated text by finding JSON in the response."""
    try:
        # Look for JSON in text
        json_match = re.search(r'\{[^}]*"terms"[^}]*"types"[^}]*\}', text)
        if not json_match:
            # Alternative search
            json_match = re.search(r'\{[^}]*"types"[^}]*"terms"[^}]*\}', text)
        
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
            return parsed.get("terms", []), parsed.get("types", [])
    except:
        pass
    return [], []


def calculate_metrics(true_items: List[str], pred_items: List[str], similarity_threshold: int = 90) -> Dict[str, float]:
    """
    Calculate exact and soft metrics (Precision, Recall, F1) for term/type extraction.
    - Exact metrics are based on exact string matches.
    - Soft metrics use fuzzy string matching to account for minor variations.
    """
    
    def _get_scores(true_positives: int, pred_count: int, true_count: int) -> Dict[str, float]:
        precision = true_positives / pred_count if pred_count > 0 else 0.0
        recall = true_positives / true_count if true_count > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return {'precision': precision, 'recall': recall, 'f1': f1}

    # Normalize and unique items
    norm_true_items = sorted(list(set([t.lower().strip() for t in true_items])))
    norm_pred_items = sorted(list(set([p.lower().strip() for p in pred_items])))

    if not norm_true_items and not norm_pred_items:
        return {
            'exact_precision': 1.0, 'exact_recall': 1.0, 'exact_f1': 1.0,
            'soft_precision': 1.0, 'soft_recall': 1.0, 'soft_f1': 1.0,
        }

    # --- Exact match calculation ---
    exact_true_positives = len(set(norm_true_items) & set(norm_pred_items))
    exact_metrics = _get_scores(exact_true_positives, len(norm_pred_items), len(norm_true_items))

    # --- Soft match calculation (using fuzzy matching) ---
    soft_true_positives = 0
    
    # Greedily match predicted items to true items
    unmatched_true = list(norm_true_items)
    
    for pred_item in norm_pred_items:
        best_match_score = 0
        best_match_true_item = None
        
        # Find the best matching true item above the threshold
        for true_item in unmatched_true:
            score = fuzz.token_set_ratio(pred_item, true_item)
            if score > best_match_score:
                best_match_score = score
                best_match_true_item = true_item
                
        if best_match_score >= similarity_threshold:
            soft_true_positives += 1
            unmatched_true.remove(best_match_true_item) # Remove to avoid re-matching

    soft_metrics = _get_scores(soft_true_positives, len(norm_pred_items), len(norm_true_items))

    return {
        'exact_precision': exact_metrics['precision'],
        'exact_recall': exact_metrics['recall'],
        'exact_f1': exact_metrics['f1'],
        'soft_precision': soft_metrics['precision'],
        'soft_recall': soft_metrics['recall'],
        'soft_f1': soft_metrics['f1'],
    }


def load_model_and_tokenizer(model_path: str):
    """Load trained model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda:1" if torch.cuda.is_available() else None,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )
    
    return model, tokenizer


def extract_terms_and_types_from_document_structured(
    model,
    tokenizer, 
    title: str, 
    text: str, 
    few_shot_examples: List[Dict] = None,
    use_tfidf: bool = False,
    tfidf_suggestions: List[str] = None,
    random_few_shot_count: Optional[int] = None,
    seed: int = 42
) -> Tuple[List[str], List[str], str]:
    """Extract terms and types from a single document using structured output when available."""
    
    # Set seed for reproducible generation
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Build conversation with random few-shot support
    conversation = build_conversation_for_inference(
        title, text, few_shot_examples, use_tfidf, 
        tfidf_suggestions, random_few_shot_count
    )
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False
    )
    
    if not OUTLINES_AVAILABLE:
        raise ImportError("Outlines library is required for structured output but not available. Please install: pip install outlines")
    
    logger.info(f"Prompt: {prompt}\n\n")
    
    # Use outlines for structured generation (mandatory)
    outlines_model = Transformers(model, tokenizer)
    generator = generate_json(outlines_model, ExtractedTermsAndTypes)
    
    result = generator(prompt, max_tokens=4096)
    extracted_terms = result.terms
    extracted_types = result.types
    generated_text = json.dumps(result.model_dump(), ensure_ascii=False)
    logger.info(f"Generated text (structured output): {generated_text}")
    
    return extracted_terms, extracted_types, generated_text


def extract_terms_and_types_from_document(
    model, 
    tokenizer, 
    title: str, 
    text: str, 
    few_shot_examples: List[Dict] = None,
    use_tfidf: bool = False,
    tfidf_suggestions: List[str] = None,
    random_few_shot_count: Optional[int] = None,
    seed: int = 42
) -> Tuple[List[str], List[str], str]:
    """Extract terms and types from a single document (always uses structured output)."""
    
    return extract_terms_and_types_from_document_structured(
        model, tokenizer, title, text, few_shot_examples,
        use_tfidf, tfidf_suggestions,
        random_few_shot_count=random_few_shot_count,
        seed=seed
    )


def process_documents(
    model, 
    tokenizer, 
    documents: List[Dict], 
    few_shot_examples: List[Dict] = None,
    use_tfidf: bool = False,
    random_few_shot_count: Optional[int] = None,
    seed: int = 42
) -> Tuple[List[Dict], Dict[str, float]]:
    """Process multiple documents and calculate summary metrics."""
    
    results = []
    all_terms_metrics = []
    all_types_metrics = []
    
    for doc in tqdm(documents, desc="Processing documents"):
        # Extract data from document (new format with RAG)
        title = doc.get("title", "")
        text = doc.get("text", "")
        true_terms = doc.get("terms", [])  # May be empty for test docs
        true_types = doc.get("types", [])  # May be empty for test docs
        tfidf_suggestions = doc.get("TF-IDF", [])
        
        # Use RAG examples as few-shot if available
        doc_few_shot_examples = doc.get("RAG", few_shot_examples)
        
        # Extract terms and types
        pred_terms, pred_types, generated_text = extract_terms_and_types_from_document_structured(
            model, tokenizer, title, text, doc_few_shot_examples,
            use_tfidf,
            tfidf_suggestions if use_tfidf else None,
            random_few_shot_count, seed
        )
        
        # Calculate metrics for terms and types (only if true values exist)
        if true_terms or true_types:
            terms_metrics = calculate_metrics(true_terms, pred_terms)
            types_metrics = calculate_metrics(true_types, pred_types)
        else:
            # For test documents without ground truth
            terms_metrics = {'exact_precision': 0.0, 'exact_recall': 0.0, 'exact_f1': 0.0,
                           'soft_precision': 0.0, 'soft_recall': 0.0, 'soft_f1': 0.0}
            types_metrics = {'exact_precision': 0.0, 'exact_recall': 0.0, 'exact_f1': 0.0,
                           'soft_precision': 0.0, 'soft_recall': 0.0, 'soft_f1': 0.0}
        
        # Combine metrics with prefixes
        combined_metrics = {}
        for key, value in terms_metrics.items():
            combined_metrics[f"terms_{key}"] = value
        for key, value in types_metrics.items():
            combined_metrics[f"types_{key}"] = value
        
        result = {
            "document_id": doc.get("document_id", doc.get("id", "")),
            "title": title,
            "text": text,
            "generated_text": generated_text,
            "extracted_terms": pred_terms,
            "extracted_types": pred_types,
            "true_terms": true_terms,
            "true_types": true_types,
            "metrics": combined_metrics,
            "tfidf_suggestions_count": len(tfidf_suggestions) if tfidf_suggestions else 0
        }
        
        results.append(result)
        all_terms_metrics.append(terms_metrics)
        all_types_metrics.append(types_metrics)
    
    # Calculate summary metrics
    def average_metrics(metrics_list):
        if not metrics_list:
            return {}
        
        avg_metrics = {}
        for key in metrics_list[0].keys():
            avg_metrics[key] = sum(m[key] for m in metrics_list) / len(metrics_list)
        return avg_metrics
    
    summary_terms_metrics = average_metrics(all_terms_metrics)
    summary_types_metrics = average_metrics(all_types_metrics)
    
    # Combine summary metrics with prefixes
    summary_metrics = {}
    for key, value in summary_terms_metrics.items():
        summary_metrics[f"terms_{key}"] = value
    for key, value in summary_types_metrics.items():
        summary_metrics[f"types_{key}"] = value
    
    return results, summary_metrics


def generate_output_filename(
    model_path: str,
    input_path: str,
    few_shot_path: Optional[str] = None,
    use_tfidf: bool = False,
    random_few_shot_count: Optional[int] = None,
    seed: int = 42
) -> str:
    """Generate descriptive output filename based on parameters."""
    
    # Extract model name
    model_name = Path(model_path).name.replace("/", "_")
    
    # Extract input filename without extension
    input_name = Path(input_path).stem
    
    # Build components
    components = [f"method_v5", model_name, input_name]
    
    # Add few-shot info
    if few_shot_path:
        few_shot_name = Path(few_shot_path).stem
        if random_few_shot_count:
            components.append(f"fs_{few_shot_name}_rand{random_few_shot_count}")
        else:
            components.append(f"fs_{few_shot_name}")
    elif random_few_shot_count:
        components.append(f"rand_fs{random_few_shot_count}")
    
    # Add feature flags
    if use_tfidf:
        components.append("tfidf")
    # Structured output is always enabled
    
    # Add seed
    components.append(f"seed{seed}")
    
    return "_".join(components) + ".json"


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Extract terms and types from documents using trained model")
    
    # Required arguments
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the trained model")
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file with documents")
    
    # Optional arguments
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file (auto-generated if not specified)")
    parser.add_argument("--few-shot", type=str, default=None,
                        help="JSONL file with few-shot examples")
    parser.add_argument("--random-few-shot", type=int, default=None,
                        help="Number of random few-shot examples to use")
    
    # Feature flags
    parser.add_argument("--use-tfidf", action="store_true",
                        help="Use TF-IDF suggestions from documents")
    
    # Generation parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible generation")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    # Load documents
    print(f"Loading documents from {args.input}...")
    documents = load_jsonl(Path(args.input))
    print(f"Loaded {len(documents)} documents")
    
    # Load few-shot examples if specified
    few_shot_examples = None
    if args.few_shot:
        print(f"Loading few-shot examples from {args.few_shot}...")
        few_shot_examples = load_few_shot_examples(Path(args.few_shot))
        print(f"Loaded {len(few_shot_examples)} few-shot examples")
    
    # Generate output filename if not specified
    if not args.output:
        args.output = generate_output_filename(
            args.model_path, args.input, args.few_shot,
            args.use_tfidf,
            args.random_few_shot, args.seed
        )
    
    print(f"Output will be saved to: {args.output}")
    print(f"Configuration:")
    print(f"  Model: {args.model_path}")
    print(f"  TF-IDF suggestions: {args.use_tfidf}")
    print(f"  Structured output: Always enabled")
    print(f"  Random few-shot: {args.random_few_shot}")
    print(f"  Seed: {args.seed}")
    
    # Process documents
    print("Starting inference...")
    results, summary_metrics = process_documents(
        model, tokenizer, documents, few_shot_examples,
        args.use_tfidf,
        args.random_few_shot, args.seed
    )
    
    # Create output
    output_data = {
        "results": results,
        "summary_metrics": summary_metrics,
        "config": {
            "model_path": args.model_path,
            "input_file": args.input,
            "few_shot_file": args.few_shot,
            "use_tfidf": args.use_tfidf,
            "use_structured_output": True,
            "random_few_shot": args.random_few_shot,
            "seed": args.seed,
            "total_documents": len(documents),
            "few_shot_examples_count": len(few_shot_examples) if few_shot_examples else 0
        }
    }
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print(f"\nInference completed!")
    print(f"Processed {len(results)} documents")
    print(f"Results saved to: {args.output}")
    
    print(f"\nSummary Metrics:")
    print(f"Terms - Exact: P={summary_metrics['terms_exact_precision']:.3f}, R={summary_metrics['terms_exact_recall']:.3f}, F1={summary_metrics['terms_exact_f1']:.3f}")
    print(f"Terms - Soft:  P={summary_metrics['terms_soft_precision']:.3f}, R={summary_metrics['terms_soft_recall']:.3f}, F1={summary_metrics['terms_soft_f1']:.3f}")
    print(f"Types - Exact: P={summary_metrics['types_exact_precision']:.3f}, R={summary_metrics['types_exact_recall']:.3f}, F1={summary_metrics['types_exact_f1']:.3f}")
    print(f"Types - Soft:  P={summary_metrics['types_soft_precision']:.3f}, R={summary_metrics['types_soft_recall']:.3f}, F1={summary_metrics['types_soft_f1']:.3f}")


if __name__ == "__main__":
    main() 