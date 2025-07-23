#!/usr/bin/env python3
"""Inference script for universal term extraction model with structured output support."""

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



class ExtractedTerms(BaseModel):
    """Schema for extracted terms output"""
    terms: List[str] = Field(description="List of extracted ontology terms from the document")


def extract_terms_from_generated_text(text: str) -> List[str]:
    """Extract terms from generated text by finding JSON in the response."""
    try:
        # Look for JSON in text
        json_match = re.search(r'\{[^}]*"terms"[^}]*\}', text)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
            return parsed.get("terms", [])
    except:
        pass
    return []


def calculate_metrics(true_terms: List[str], pred_terms: List[str], similarity_threshold: int = 90) -> Dict[str, float]:
    """
    Calculate exact and soft metrics (Precision, Recall, F1) for term extraction.
    - Exact metrics are based on exact string matches.
    - Soft metrics use fuzzy string matching to account for minor variations.
    """
    
    def _get_scores(true_positives: int, pred_count: int, true_count: int) -> Dict[str, float]:
        precision = true_positives / pred_count if pred_count > 0 else 0.0
        recall = true_positives / true_count if true_count > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return {'precision': precision, 'recall': recall, 'f1': f1}

    # Normalize and unique terms
    norm_true_terms = sorted(list(set([t.lower().strip() for t in true_terms])))
    norm_pred_terms = sorted(list(set([p.lower().strip() for p in pred_terms])))

    if not norm_true_terms and not norm_pred_terms:
        return {
            'exact_precision': 1.0, 'exact_recall': 1.0, 'exact_f1': 1.0,
            'soft_precision': 1.0, 'soft_recall': 1.0, 'soft_f1': 1.0,
        }

    # --- Exact match calculation ---
    exact_true_positives = len(set(norm_true_terms) & set(norm_pred_terms))
    exact_metrics = _get_scores(exact_true_positives, len(norm_pred_terms), len(norm_true_terms))

    # --- Soft match calculation (using fuzzy matching) ---
    soft_true_positives = 0
    
    # Greedily match predicted terms to true terms
    unmatched_true = list(norm_true_terms)
    
    for pred_term in norm_pred_terms:
        best_match_score = 0
        best_match_true_term = None
        
        # Find the best matching true term above the threshold
        for true_term in unmatched_true:
            score = fuzz.token_set_ratio(pred_term, true_term)
            if score > best_match_score:
                best_match_score = score
                best_match_true_term = true_term
                
        if best_match_score >= similarity_threshold:
            soft_true_positives += 1
            unmatched_true.remove(best_match_true_term) # Remove to avoid re-matching

    soft_metrics = _get_scores(soft_true_positives, len(norm_pred_terms), len(norm_true_terms))

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
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    return model, tokenizer


def extract_terms_from_document_structured(
    model,
    tokenizer, 
    title: str, 
    text: str, 
    few_shot_examples: List[Dict] = None,
    use_tfidf: bool = False,
    tfidf_suggestions: List[str] = None,
    use_structured_output: bool = True,
    random_few_shot_count: Optional[int] = None,
    seed: int = 42
) -> Tuple[List[str], str]:
    """Extract terms from a single document using structured output when available."""
    
    # Set seed for reproducible generation
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Build conversation with random few-shot support
    conversation = build_conversation_for_inference(
        title, text, few_shot_examples, use_tfidf, tfidf_suggestions, 
        random_few_shot_count
    )
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False
    )
    
    extracted_terms = []
    generated_text = ""
    
    if use_structured_output and OUTLINES_AVAILABLE:
        try:
            logger.info(f"Prompt: {prompt}\n\n")
            # Use outlines for structured generation
            outlines_model = Transformers(model, tokenizer)
            generator = generate_json(outlines_model, ExtractedTerms)
            
            result = generator(prompt, max_tokens=4096)
            extracted_terms = result.terms
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
        
        # Extract terms using regex parsing
        extracted_terms = extract_terms_from_generated_text(generated_text)
    
    return extracted_terms, generated_text


def extract_terms_from_document(
    model, 
    tokenizer, 
    title: str, 
    text: str, 
    few_shot_examples: List[Dict] = None,
    use_tfidf: bool = False,
    tfidf_suggestions: List[str] = None,
    random_few_shot_count: Optional[int] = None,
    seed: int = 42
) -> Tuple[List[str], str]:
    """Legacy function for backward compatibility."""
    return extract_terms_from_document_structured(
        model, tokenizer, title, text, few_shot_examples, 
        use_tfidf, tfidf_suggestions, use_structured_output=False, 
        random_few_shot_count=random_few_shot_count, seed=seed
    )


def process_documents(
    model, 
    tokenizer, 
    documents: List[Dict], 
    few_shot_examples: List[Dict] = None,
    use_tfidf: bool = False,
    use_structured_output: bool = True,
    random_few_shot_count: Optional[int] = None,
    use_rag: bool = False
) -> Tuple[List[Dict], Dict[str, float]]:
    """Process multiple documents, extract terms, and calculate metrics if ground truth is available."""
    results = []
    all_metrics = []
    
    # Check if ground truth is available in the first document
    has_ground_truth = "OL" in documents[0] if documents else False
    
    for i, doc in tqdm(enumerate(documents), total=len(documents), desc="Processing documents"):
        title = doc.get("title", "")
        text = doc.get("text", "")
        doc_id = doc.get("id", str(i))
        tfidf_suggestions = doc.get("TF-IDF", []) if use_tfidf else None
        
        # Extract RAG examples from the document if use_rag is enabled
        examples_to_use = few_shot_examples
        if use_rag and "RAG" in doc and doc["RAG"]:
            # Use RAG examples instead of few_shot_examples
            examples_to_use = doc["RAG"]
            # Apply random sampling if needed
            if random_few_shot_count is not None and len(examples_to_use) > random_few_shot_count:
                import random
                examples_to_use = random.sample(examples_to_use, random_few_shot_count)
        
        extracted_terms, generated_text = extract_terms_from_document_structured(
            model, tokenizer, title, text, examples_to_use, 
            use_tfidf, tfidf_suggestions, use_structured_output, 
            random_few_shot_count, seed=42
        )
        
        result = {
            "id": doc_id,
            "title": title,
            "text": text,
            "generated_text": generated_text,
            "extracted_terms": sorted(list(set(extracted_terms))), # Store unique sorted terms
        }
        
        if use_tfidf and tfidf_suggestions:
            result["tfidf_suggestions"] = tfidf_suggestions
            
        if use_rag and examples_to_use != few_shot_examples:
            result["rag_examples_count"] = len(examples_to_use)
        
        if has_ground_truth:
            true_terms = doc.get("OL", [])
            result["true_terms"] = sorted(list(set(true_terms)))
            metrics = calculate_metrics(true_terms, extracted_terms)
            result["metrics"] = metrics
            all_metrics.append(metrics)
            
        results.append(result)

    avg_metrics = {}
    if has_ground_truth and all_metrics:
        # Average all collected metrics
        avg_metrics = {key: sum(m[key] for m in all_metrics) / len(all_metrics) for key in all_metrics[0]}

    return results, avg_metrics


def generate_output_filename(
    model_path: str,
    input_path: str,
    few_shot_path: Optional[str] = None,
    use_tfidf: bool = False,
    use_structured_output: bool = False,
    random_few_shot_count: Optional[int] = None,
    seed: int = 42,
    use_rag: bool = False
) -> str:
    """
    Automatically generates output filename based on parameters.
    
    Args:
        model_path: Path to model
        input_path: Path to input data
        few_shot_path: Path to few-shot examples
        use_tfidf: Whether to use TF-IDF
        use_structured_output: Whether to use structured output
        random_few_shot_count: Number of random few-shot examples
        seed: Seed for reproducibility
        use_rag: Whether to use RAG examples
        
    Returns:
        Generated filename
    """
    # Extract model name
    model_name = Path(model_path).name.replace("/", "_").replace("-", "_")
    
    # Extract input data information
    input_path_obj = Path(input_path)
    input_parts = input_path_obj.parts
    
    # Try to extract domain, subject area and data type from path
    domain = "unknown"
    subject_area = "unknown"
    data_type = "unknown"
    split_type = "unknown"
    
    for i, part in enumerate(input_parts):
        # Look for domains (ecology, engineering)
        if "ecology" in part.lower() or "engineering" in part.lower():
            domain = part.lower()
        # Look for subject areas (scholarly, etc.)
        elif "scholarly" in part.lower():
            subject_area = part.lower()
        # Look for data types
        elif "train" in part.lower():
            data_type = "train"
            # Check folder before train - this might be subject area
            if i > 0 and subject_area == "unknown":
                prev_part = input_parts[i-1]
                # If previous folder is not domain and not standard folder
                if (prev_part.lower() not in ["ecology", "engineering", "2025"] and 
                    not prev_part.startswith("TaskA")):
                    subject_area = prev_part.lower()
        elif "test" in part.lower():
            data_type = "test"
            # Similarly for test
            if i > 0 and subject_area == "unknown":
                prev_part = input_parts[i-1]
                if (prev_part.lower() not in ["ecology", "engineering", "2025"] and 
                    not prev_part.startswith("TaskA")):
                    subject_area = prev_part.lower()
        elif "val" in part.lower():
            data_type = "val"
            # Similarly for val
            if i > 0 and subject_area == "unknown":
                prev_part = input_parts[i-1]
                if (prev_part.lower() not in ["ecology", "engineering", "2025"] and 
                    not prev_part.startswith("TaskA")):
                    subject_area = prev_part.lower()
        elif "split" in part.lower():
            split_type = part
    
    # Base filename
    base_name = f"docs2terms_{data_type}_results"
    
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
    
    # Add split information
    if split_type != "unknown":
        base_name += f"_{split_type}"
    
    # Add subject area (scholarly, etc.)
    if subject_area != "unknown":
        base_name += f"_{subject_area}"
    
    # Add domain (engineering, ecology)
    if domain != "unknown":
        base_name += f"_{domain}"
    
    # Add model
    base_name += f"_{model_name}"
    
    # Add additional flags
    flags = []
    if use_tfidf:
        flags.append("tfidf")
    if use_structured_output:
        flags.append("structured")
    
    if flags:
        base_name += f"_{'_'.join(flags)}"
    
    # Add seed if not standard
    if seed != 42:
        base_name += f"_seed_{seed}"
    
    return f"{base_name}.jsonl"


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained model")
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--input", type=Path, required=True, help="Input documents JSONL file")
    parser.add_argument("--output", type=Path, required=False, help="Output results JSON file (auto-generated if not provided)")
    parser.add_argument("--few-shot", type=Path, required=False, help="Few-shot examples JSONL file")
    parser.add_argument("--random-few-shot", type=int, help="Number of random few-shot examples to use from the provided file")
    parser.add_argument("--use-tfidf", action="store_true", help="Use TF-IDF suggestions from input documents")
    parser.add_argument("--use-structured-output", action="store_true", help="Use structured output via outlines (if available)")
    parser.add_argument("--use-rag", action="store_true", help="Use RAG examples from the RAG field in input documents instead of few-shot examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible generation")
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if args.output is None:
        output_filename = generate_output_filename(
            args.model_path, 
            str(args.input), 
            str(args.few_shot) if args.few_shot else None,
            args.use_tfidf, 
            args.use_structured_output, 
            args.random_few_shot, 
            args.seed,
            args.use_rag
        )
        print(f"Automatically generated output filename: {output_filename}")
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
        print("Using RAG examples from input documents")
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
    
    # Load documents
    print(f"Loading documents from {args.input}")
    documents = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            documents.append(json.loads(line))
    
    if not documents:
        print("Input file is empty. Exiting.")
        return
        
    print(f"Loaded {len(documents)} documents")
    
    # Process documents
    results, avg_metrics = process_documents(
        model, tokenizer, documents, few_shot_examples, 
        args.use_tfidf, args.use_structured_output, args.random_few_shot, args.use_rag
    )
    
    # Prepare output
    output_data = {"results": results}
    if avg_metrics:
        print("\n--- Average Metrics ---")
        print(f"Model: {args.model_path}")
        if args.use_rag:
            print("Using RAG examples from input documents")
        else:
            print(f"Few-shot examples: {len(few_shot_examples) if few_shot_examples else 0}")
        if args.random_few_shot:
            print(f"Random few-shot count: {args.random_few_shot}")
        print(f"TF-IDF enabled: {args.use_tfidf}")
        print(f"Structured output: {args.use_structured_output and OUTLINES_AVAILABLE}")
        print(f"Seed: {args.seed}")
        print("\n[Exact Match]")
        print(f"  Precision: {avg_metrics['exact_precision']:.4f}")
        print(f"  Recall:    {avg_metrics['exact_recall']:.4f}")
        print(f"  F1-score:  {avg_metrics['exact_f1']:.4f}")
        print("\n[Soft Match (Fuzzy)]")
        print(f"  Precision: {avg_metrics['soft_precision']:.4f}")
        print(f"  Recall:    {avg_metrics['soft_recall']:.4f}")
        print(f"  F1-score:  {avg_metrics['soft_f1']:.4f}")
        print("-----------------------")
        output_data["summary_metrics"] = avg_metrics
        output_data["config"] = {
            "model_path": args.model_path,
            "few_shot_examples": len(few_shot_examples) if few_shot_examples else 0,
            "random_few_shot_count": args.random_few_shot,
            "use_tfidf": args.use_tfidf,
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