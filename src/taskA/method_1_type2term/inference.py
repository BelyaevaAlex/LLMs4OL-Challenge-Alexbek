#!/usr/bin/env python3
"""Inference script for term-to-type task with RAG support."""

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


class PredictedTypes(BaseModel):
    """Schema for predicted types output"""
    types: List[str] = Field(description="List of predicted semantic types for the term")


def extract_types_from_generated_text(text: str) -> List[str]:
    """Extract types from generated text by finding JSON in the response."""
    try:
        # Look for JSON in text
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
    )
    
    return model, tokenizer


def predict_types_for_term(
    model,
    tokenizer, 
    term: str,
    few_shot_examples: List[Dict] = None,
    use_structured_output: bool = True,
    random_few_shot_count: Optional[int] = None,
    seed: int = 42
) -> Tuple[List[str], str]:
    """Predict types for a single term using structured output when available."""
    
    # Set seed for reproducible generation
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Build conversation with random few-shot support
    conversation = build_conversation_for_inference(
        term, few_shot_examples, random_few_shot_count
    )
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False
    )
    
    predicted_types = []
    generated_text = ""
    
    if use_structured_output and OUTLINES_AVAILABLE:
        try:
            logger.info(f"Prompt: {prompt}\n\n")
            # Use outlines for structured generation
            outlines_model = Transformers(model, tokenizer)
            generator = generate_json(outlines_model, PredictedTypes)
            
            result = generator(prompt, max_tokens=4096)
            predicted_types = result.types
            generated_text = json.dumps(result.model_dump(), ensure_ascii=False)
            logger.info(f"Generated text: {generated_text}")
            
        except Exception as e:
            print(f"Warning: Structured generation failed ({e}), falling back to regular generation")
            use_structured_output = False
    
    if not use_structured_output or not OUTLINES_AVAILABLE:
        # Fallback to regular generation
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
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


def process_terms(
    model, 
    tokenizer, 
    terms_data: List[Dict], 
    few_shot_examples: List[Dict] = None,
    use_structured_output: bool = True,
    random_few_shot_count: Optional[int] = None,
    use_rag: bool = False
) -> Tuple[List[Dict], Dict[str, float]]:
    """Process multiple terms, predict types, and calculate metrics if ground truth is available."""
    results = []
    all_true_types = []
    all_pred_types = []
    
    # Check if ground truth is available in the first term
    has_ground_truth = "types" in terms_data[0] and terms_data[0]["types"] if terms_data else False
    
    for i, term_data in tqdm(enumerate(terms_data), total=len(terms_data), desc="Processing terms"):
        term = term_data.get("term", "")
        true_types = term_data.get("types", [])
        
        # Extract RAG examples from the term data if use_rag is enabled
        examples_to_use = few_shot_examples
        if use_rag and "RAG" in term_data and term_data["RAG"]:
            # Use RAG examples instead of few_shot_examples
            examples_to_use = term_data["RAG"]
            # Apply random sampling if needed
            if random_few_shot_count is not None and len(examples_to_use) > random_few_shot_count:
                import random
                examples_to_use = random.sample(examples_to_use, random_few_shot_count)
        
        predicted_types, generated_text = predict_types_for_term(
            model, tokenizer, term, examples_to_use, 
            use_structured_output, random_few_shot_count, seed=42
        )
        
        result = {
            "term": term,
            "generated_text": generated_text,
            "predicted_types": sorted(list(set(predicted_types))), # Store unique sorted types
        }
        
        if use_rag and examples_to_use != few_shot_examples:
            result["rag_examples_count"] = len(examples_to_use)
        
        if has_ground_truth:
            result["true_types"] = sorted(list(set(true_types)))
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
    random_few_shot_count: Optional[int] = None,
    seed: int = 42,
    use_rag: bool = False
) -> str:
    """Generate output filename based on parameters."""
    
    # Extract model name
    model_name = Path(model_path).name.replace("/", "_").replace("-", "_")
    
    # Extract information from input path
    input_path_obj = Path(input_path)
    input_parts = input_path_obj.parts
    
    # Try to extract domain and data type from path
    domain = "unknown"
    data_type = "unknown"
    
    for i, part in enumerate(input_parts):
        if "engineering" in part.lower() or "scholarly" in part.lower() or "ecology" in part.lower():
            domain = part.lower()
        elif "train" in part.lower():
            data_type = "train"
        elif "test" in part.lower():
            data_type = "test"
        elif "val" in part.lower():
            data_type = "val"
    
    # Base name
    base_name = f"terms2types_{data_type}_results"
    
    # Add few-shot or RAG information
    if use_rag:
        if random_few_shot_count:
            base_name += f"_rag_random_{random_few_shot_count}"
        else:
            base_name += "_rag"
    elif few_shot_path:
        if random_few_shot_count:
            base_name += f"_few_shot_random_{random_few_shot_count}"
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
    
    # Add seed if not standard
    if seed != 42:
        base_name += f"_seed_{seed}"
    
    return f"{base_name}.json"


def main():
    parser = argparse.ArgumentParser(description="Run inference for term-to-type task")
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--input", type=Path, required=True, help="Input terms JSON file")
    parser.add_argument("--output", type=Path, required=False, help="Output results JSON file (auto-generated if not provided)")
    parser.add_argument("--few-shot", type=Path, required=False, help="Few-shot examples JSON file")
    parser.add_argument("--random-few-shot", type=int, help="Number of random few-shot examples to use from the provided file")
    parser.add_argument("--use-structured-output", action="store_true", help="Use structured output via outlines (if available)")
    parser.add_argument("--use-rag", action="store_true", help="Use RAG examples from the RAG field in input terms instead of few-shot examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible generation")
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if args.output is None:
        output_filename = generate_output_filename(
            args.model_path, 
            str(args.input), 
            str(args.few_shot) if args.few_shot else None,
            args.use_structured_output, 
            args.random_few_shot, 
            args.seed,
            args.use_rag
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
    
    # Process terms
    results, avg_metrics = process_terms(
        model, tokenizer, terms_data, few_shot_examples, 
        args.use_structured_output, args.random_few_shot, args.use_rag
    )
    
    # Prepare output
    output_data = {"results": results}
    if avg_metrics:
        print("\n--- Average Metrics ---")
        print(f"Model: {args.model_path}")
        if args.use_rag:
            print("Using RAG examples from input terms")
        else:
            print(f"Few-shot examples: {len(few_shot_examples) if few_shot_examples else 0}")
        if args.random_few_shot:
            print(f"Random few-shot count: {args.random_few_shot}")
        print(f"Structured output: {args.use_structured_output and OUTLINES_AVAILABLE}")
        print(f"Seed: {args.seed}")
        print(f"Jaccard Similarity: {avg_metrics['jaccard_similarity']:.4f}")
        print(f"F1 Score: {avg_metrics['f1_score']:.4f}")
        print(f"Exact Match Accuracy: {avg_metrics['exact_match_accuracy']:.4f}")
        print("-----------------------")
        output_data["summary_metrics"] = avg_metrics
        output_data["config"] = {
            "model_path": args.model_path,
            "few_shot_examples": len(few_shot_examples) if few_shot_examples else 0,
            "random_few_shot_count": args.random_few_shot,
            "use_structured_output": args.use_structured_output,
            "structured_output_available": OUTLINES_AVAILABLE,
            "use_rag": args.use_rag,
            "seed": args.seed,
            "deterministic": True
        }
        
    # Save results
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Results and metrics saved to {output_filename}")


if __name__ == "__main__":
    main() 