#!/usr/bin/env python3
"""Inference script for TaskB-TermTyping with RAG support.

New features:
- Batch processing: Process multiple terms simultaneously for faster inference
- Retry logic: Automatically retry failed structured generation attempts (up to 3 times by default)
- Enhanced error handling: Graceful fallback to regular generation when structured output fails
- Flash Attention 2: Improved memory efficiency and speed

Usage examples:
- Basic inference: python inference.py --model-path /path/to/model --input terms.json
- Batch processing: python inference.py --model-path /path/to/model --input terms.json --batch-size 8
- With retries: python inference.py --model-path /path/to/model --input terms.json --max-retries 5
- RAG with batching: python inference.py --model-path /path/to/model --input terms.json --use-rag --batch-size 4
"""

import json
import argparse
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel, Field

# Optional outlines import
try:
    import outlines
    from outlines.models import Transformers
    from outlines.generate import json as generate_json
    OUTLINES_AVAILABLE = True
except ImportError:
    OUTLINES_AVAILABLE = False
    print("Warning: outlines not available. Falling back to regular generation.")

from .data import (
    load_few_shot_examples, 
    build_conversation_for_inference,
    SYSTEM_PROMPT
)

MAX_TOKENS = 4096

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add file handler
file_handler = logging.FileHandler('inference_taskB.log')
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


class PredictedTypes(BaseModel):
    """Schema for predicted types output"""
    types: List[str] = Field(description="List of predicted semantic types for the term")


def extract_types_from_generated_text(text: str) -> List[str]:
    """Extract types from generated text by finding JSON in the response."""
    try:
        # Look for JSON in the text
        json_match = re.search(r'\{[^}]*"types"[^}]*\}', text)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
            return parsed.get("types", [])
    except:
        pass
    return []


def calculate_metrics(true_types_list: List[List[str]], pred_types_list: List[List[str]]) -> Dict[str, float]:
    """
    Calculate metrics for type prediction.
    Each item is a list of types for a term.
    """
    
    def jaccard_similarity(set1, set2):
        """Calculate Jaccard similarity between two sets"""
        if not set1 and not set2:
            return 1.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
    
    def f1_score(set1, set2):
        """Calculate F1 score between two sets"""
        if not set1 and not set2:
            return 1.0
        intersection = len(set1.intersection(set2))
        precision = intersection / len(set2) if len(set2) > 0 else 0.0
        recall = intersection / len(set1) if len(set1) > 0 else 0.0
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Convert to sets for easier comparison
    true_sets = [set(types) for types in true_types_list]
    pred_sets = [set(types) for types in pred_types_list]
    
    # Calculate metrics
    jaccard_scores = [jaccard_similarity(true_set, pred_set) for true_set, pred_set in zip(true_sets, pred_sets)]
    f1_scores = [f1_score(true_set, pred_set) for true_set, pred_set in zip(true_sets, pred_sets)]
    
    # Calculate exact match (all types match exactly)
    exact_matches = [true_set == pred_set for true_set, pred_set in zip(true_sets, pred_sets)]
    
    return {
        'jaccard_similarity': sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0.0,
        'f1_score': sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        'exact_match_accuracy': sum(exact_matches) / len(exact_matches) if exact_matches else 0.0,
    }


def load_model_and_tokenizer(model_path: str):
    """Load trained model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        attn_implementation="flash_attention_2"
    )
    
    return model, tokenizer


def predict_types_for_terms_batch(
    model,
    tokenizer,
    batch_data: List[Dict],
    few_shot_examples: List[Dict] = None,
    use_structured_output: bool = True,
    few_shot_amount: Optional[int] = None,
    seed: int = 42,
    use_rag: bool = False,
    max_retries: int = 3
) -> List[Tuple[List[str], str]]:
    """Predict types for a batch of terms using structured output when available."""
    
    # Set seed for reproducible generation
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Prepare prompts for all terms in batch
    prompts = []
    for term_data in batch_data:
        term = term_data.get("term", "")
        
        # Extract RAG examples from the term data if use_rag is enabled
        rag_examples = None
        if use_rag and "RAG" in term_data and term_data["RAG"]:
            rag_examples = term_data["RAG"]
            # Apply first N examples if few_shot_amount is specified (they are sorted by semantic similarity)
            if few_shot_amount is not None and len(rag_examples) > few_shot_amount:
                rag_examples = rag_examples[:few_shot_amount]
        
        # Build conversation with RAG support
        conversation = build_conversation_for_inference(
            term, few_shot_examples, few_shot_amount, use_rag, rag_examples
        )
        
        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        prompts.append(prompt)
    
    results = []
    
    if use_structured_output and OUTLINES_AVAILABLE:
        try:
            # Use outlines for structured generation
            outlines_model = Transformers(model, tokenizer)
            generator = generate_json(outlines_model, PredictedTypes)
            
            # Process each prompt in the batch
            for i, prompt in enumerate(prompts):
                success = False
                predicted_types = []
                generated_text = ""
                
                # Try structured generation with retries
                for attempt in range(max_retries):
                    try:
                        logger.info(f"Batch item {i+1}/{len(prompts)} - Attempt {attempt+1}/{max_retries} - Prompt: {prompt[:200]}...\n")
                        result = generator(prompt, max_tokens=MAX_TOKENS)
                        predicted_types = result.types
                        generated_text = json.dumps(result.model_dump(), ensure_ascii=False)
                        logger.info(f"Generated text: {generated_text}")
                        success = True
                        break
                    except Exception as e:
                        logger.warning(f"Structured generation attempt {attempt+1} failed for item {i+1}: {e}")
                        if attempt < max_retries - 1:
                            print(f"Warning: Structured generation attempt {attempt+1} failed for item {i+1} ({e}), retrying...")
                        else:
                            print(f"Warning: All {max_retries} structured generation attempts failed for item {i+1} ({e}), falling back to regular generation")
                
                if success:
                    results.append((predicted_types, generated_text))
                else:
                    # Fallback for this specific item after all retries failed
                    fallback_result = _fallback_generation_single(model, tokenizer, prompt)
                    results.append(fallback_result)
            
        except Exception as e:
            print(f"Warning: Batch structured generation setup failed ({e}), falling back to regular generation")
            use_structured_output = False
    
    if not use_structured_output or not OUTLINES_AVAILABLE:
        # Fallback to regular batch generation
        results = _regular_batch_generation(model, tokenizer, prompts)
    
    return results


def _regular_batch_generation(model, tokenizer, prompts: List[str]) -> List[Tuple[List[str], str]]:
    """Regular batch generation without structured output."""
    
    # Tokenize all prompts
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKENS,
        padding=True
    )
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,  # Deterministic generation
            pad_token_id=tokenizer.eos_token_id,
            temperature=None,  # Not used when do_sample=False
            top_p=None,       # Not used when do_sample=False
            num_beams=1,      # Greedy decoding
        )
    
    results = []
    for i, (input_ids, output_ids) in enumerate(zip(inputs["input_ids"], outputs)):
        # Decode only new tokens
        new_tokens = output_ids[len(input_ids):]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Extract types using regex parsing
        predicted_types = extract_types_from_generated_text(generated_text)
        results.append((predicted_types, generated_text))
    
    return results


def _fallback_generation_single(model, tokenizer, prompt: str) -> Tuple[List[str], str]:
    """Fallback generation for a single prompt."""
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKENS
    )
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,  # Deterministic generation
            pad_token_id=tokenizer.eos_token_id,
            temperature=None,  # Not used when do_sample=False
            top_p=None,       # Not used when do_sample=False
            num_beams=1,      # Greedy decoding
        )
    
    # Decode only new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Extract types using regex parsing
    predicted_types = extract_types_from_generated_text(generated_text)
    
    return predicted_types, generated_text


def predict_types_for_term(
    model,
    tokenizer, 
    term: str,
    few_shot_examples: List[Dict] = None,
    use_structured_output: bool = True,
    few_shot_amount: Optional[int] = None,
    seed: int = 42,
    use_rag: bool = False,
    rag_examples: List[Dict] = None,
    max_retries: int = 3
) -> Tuple[List[str], str]:
    """Predict types for a single term using structured output when available."""
    
    # Create batch data for single term
    term_data = {"term": term}
    if use_rag and rag_examples:
        term_data["RAG"] = rag_examples
    
    batch_results = predict_types_for_terms_batch(
        model, tokenizer, [term_data], few_shot_examples,
        use_structured_output, few_shot_amount, seed, use_rag, max_retries
    )
    
    return batch_results[0]


def process_terms(
    model, 
    tokenizer, 
    terms_data: List[Dict], 
    few_shot_examples: List[Dict] = None,
    use_structured_output: bool = True,
    few_shot_amount: Optional[int] = None,
    use_rag: bool = False,
    batch_size: int = 1,
    max_retries: int = 3
) -> Tuple[List[Dict], Dict[str, float]]:
    """Process multiple terms with batch support, predict types, and calculate metrics if ground truth is available."""
    results = []
    all_true_types = []
    all_pred_types = []
    
    # Check if ground truth is available in the first term
    has_ground_truth = "types" in terms_data[0] and terms_data[0]["types"] if terms_data else False
    
    # Process terms in batches
    total_batches = (len(terms_data) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(total_batches), desc=f"Processing batches (batch_size={batch_size})"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(terms_data))
        batch_data = terms_data[start_idx:end_idx]
        
        # Validate required fields
        for term_data in batch_data:
            term = term_data.get("term", "")
            if "id" not in term_data:
                raise ValueError(f"Missing required 'id' field in term data for term: '{term}'. All input data must contain 'id' field.")
        
        # Predict types for batch
        batch_results = predict_types_for_terms_batch(
            model, tokenizer, batch_data, few_shot_examples,
            use_structured_output, few_shot_amount, seed=42, use_rag=use_rag, max_retries=max_retries
        )
        
        # Process results
        for term_data, (predicted_types, generated_text) in zip(batch_data, batch_results):
            term_id = term_data["id"]
            true_types = term_data.get("types", [])
            
            # Create result in required format: {"id": "...", "types": [...]}
            result = {
                "id": term_id,
                "types": sorted(list(set(predicted_types))), # Store unique sorted types
            }
            
            if has_ground_truth:
                all_true_types.append(true_types)
                all_pred_types.append(predicted_types)
                
            results.append(result)

    avg_metrics = {}
    if has_ground_truth and all_true_types:
        # Calculate metrics
        avg_metrics = calculate_metrics(all_true_types, all_pred_types)

    return results, avg_metrics


def generate_output_filename(
    model_path: str,
    input_path: str,
    few_shot_path: Optional[str] = None,
    use_structured_output: bool = False,
    few_shot_amount: Optional[int] = None,
    seed: int = 42,
    use_rag: bool = False,
    domain: str = "unknown",
    batch_size: int = 1
) -> str:
    """Generate output filename based on parameters."""
    
    # Extract model name
    model_name = Path(model_path).name.replace("/", "_").replace("-", "_")
    
    # Extract information from input path
    input_path_obj = Path(input_path)
    input_parts = input_path_obj.parts
    
    # Try to extract data type from path
    data_type = "unknown"
    for part in input_parts:
        if "train" in part.lower():
            data_type = "train"
        elif "test" in part.lower():
            data_type = "test"
        elif "val" in part.lower():
            data_type = "val"
    
    # Base name
    base_name = f"taskB_termtyping_{data_type}_results"
    
    # Add few-shot or RAG information
    if use_rag:
        if few_shot_amount:
            base_name += f"_rag_amount_{few_shot_amount}"
        else:
            base_name += "_rag"
    elif few_shot_path:
        if few_shot_amount:
            base_name += f"_few_shot_amount_{few_shot_amount}"
        else:
            base_name += "_few_shot"
    
    # Add domain
    if domain != "unknown":
        base_name += f"_{domain}"
    
    # Add model
    base_name += f"_{model_name}"
    
    # Add additional flags
    flags = []
    if use_structured_output:
        flags.append("structured")
    
    if flags:
        base_name += f"_{'_'.join(flags)}"
    
    # Add batch size if not standard
    if batch_size != 1:
        base_name += f"_batch_{batch_size}"
    
    # Add seed if not standard
    if seed != 42:
        base_name += f"_seed_{seed}"
    
    return f"{base_name}.json"


def main():
    parser = argparse.ArgumentParser(description="Run inference for TaskB-TermTyping")
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--input", type=Path, required=True, help="Input terms JSON file")
    parser.add_argument("--output", type=Path, required=False, help="Output results JSON file (auto-generated if not provided)")
    parser.add_argument("--few-shot", type=Path, required=False, help="Few-shot examples JSON file")
    parser.add_argument("--few-shot-amount", type=int, help="Number of few-shot examples to use (first N examples, sorted by semantic similarity)")
    parser.add_argument("--use-structured-output", action="store_true", help="Use structured output via outlines (if available)")
    parser.add_argument("--use-rag", action="store_true", help="Use RAG examples from the RAG field in input terms instead of few-shot examples")
    parser.add_argument("--domain", type=str, default="unknown", help="Domain name for output filename (e.g., SWEET, MatOnto, OBI)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible generation")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference (default: 1)")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries for structured generation (default: 3)")
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if args.output is None:
        output_filename = generate_output_filename(
            args.model_path, 
            str(args.input), 
            str(args.few_shot) if args.few_shot else None,
            args.use_structured_output, 
            args.few_shot_amount, 
            args.seed,
            args.use_rag,
            args.domain,
            args.batch_size
        )
        print(f"Auto-generated output filename: {output_filename}")
    else:
        output_filename = str(args.output)
        print(f"Using specified output filename: {output_filename}")
    
    # Set global seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    # Load few-shot examples
    few_shot_examples = load_few_shot_examples(args.few_shot)
    if few_shot_examples:
        print(f"Loaded {len(few_shot_examples)} few-shot examples")
    else:
        print("No few-shot examples provided")
        
    # Check RAG usage
    if args.use_rag:
        print("Using RAG examples from input terms")
        if args.few_shot:
            print("Warning: --few-shot parameter will be ignored when --use-rag is enabled")
    else:
        print("Using traditional few-shot examples (if provided)")
    
    # Check structured output availability
    if args.use_structured_output:
        if OUTLINES_AVAILABLE:
            print("Using structured output via outlines")
        else:
            print("Warning: outlines not available, falling back to regular generation")
    else:
        print("Using regular generation (structured output disabled)")
    
    # Load terms
    print(f"Loading terms from {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        terms_data = json.load(f)
    
    if not terms_data:
        print("Input file is empty. Exiting.")
        return
        
    print(f"Loaded {len(terms_data)} terms")
    print(f"Using batch size: {args.batch_size}")
    if args.use_structured_output and OUTLINES_AVAILABLE:
        print(f"Max retries for structured generation: {args.max_retries}")
    
    # Process terms
    results, avg_metrics = process_terms(
        model, tokenizer, terms_data, few_shot_examples, 
        args.use_structured_output, args.few_shot_amount, args.use_rag, args.batch_size, args.max_retries
    )
    
    # Save results in required format
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    
    # Print metrics if available
    if avg_metrics:
        print("\n--- Average Metrics ---")
        print(f"Model: {args.model_path}")
        print(f"Domain: {args.domain}")
        print(f"Batch size: {args.batch_size}")
        if args.use_rag:
            print("Using RAG examples from input terms")
        else:
            print(f"Few-shot examples: {len(few_shot_examples) if few_shot_examples else 0}")
        if args.few_shot_amount:
            print(f"Few-shot amount: {args.few_shot_amount}")
        print(f"Structured output: {args.use_structured_output and OUTLINES_AVAILABLE}")
        if args.use_structured_output and OUTLINES_AVAILABLE:
            print(f"Max retries: {args.max_retries}")
        print(f"Seed: {args.seed}")
        print(f"Jaccard Similarity: {avg_metrics['jaccard_similarity']:.4f}")
        print(f"F1 Score: {avg_metrics['f1_score']:.4f}")
        print(f"Exact Match Accuracy: {avg_metrics['exact_match_accuracy']:.4f}")
        print("-----------------------")
    
    print(f"Results and metrics saved to {output_filename}")


if __name__ == "__main__":
    main() 