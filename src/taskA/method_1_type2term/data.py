#!/usr/bin/env python3
"""Data processing module for term-to-type task with RAG support."""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Any

# System prompt for term-to-type task
SYSTEM_PROMPT = """You are an expert in ontology and semantic type classification. Your task is to predict the semantic types for given terms based on their context and similar examples.

Given a term, you should predict its semantic types from the domain-specific ontology. Use the provided examples to understand the patterns and relationships between terms and their types.

Output your response as a JSON object with the following structure:
{
  "types": ["type1", "type2", ...]
}

The types should be relevant semantic categories that best describe the given term."""


def load_few_shot_examples(few_shot_path: Optional[Path]) -> List[Dict]:
    """Load few-shot examples from JSONL file."""
    if not few_shot_path or not few_shot_path.exists():
        return []
    
    examples = []
    with open(few_shot_path, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    
    return examples


def build_conversation_for_training(
    term: str,
    types: List[str],
    few_shot_examples: List[Dict] = None,
    random_few_shot_count: Optional[int] = None
) -> List[Dict[str, str]]:
    """Build conversation format for training."""
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add few-shot examples if provided
    if few_shot_examples:
        examples_to_use = few_shot_examples
        if random_few_shot_count is not None and len(examples_to_use) > random_few_shot_count:
            examples_to_use = random.sample(examples_to_use, random_few_shot_count)
        
        for example in examples_to_use:
            example_term = example.get("term", "")
            example_types = example.get("types", [])
            
            conversation.append({
                "role": "user",
                "content": f"Term: {example_term}"
            })
            conversation.append({
                "role": "assistant",
                "content": json.dumps({"types": example_types}, ensure_ascii=False)
            })
    
    # Add the actual training example
    conversation.append({
        "role": "user",
        "content": f"Term: {term}"
    })
    conversation.append({
        "role": "assistant",
        "content": json.dumps({"types": types}, ensure_ascii=False)
    })
    
    return conversation


def build_conversation_for_inference(
    term: str,
    few_shot_examples: List[Dict] = None,
    random_few_shot_count: Optional[int] = None
) -> List[Dict[str, str]]:
    """Build conversation format for inference."""
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add few-shot examples if provided
    if few_shot_examples:
        examples_to_use = few_shot_examples
        if random_few_shot_count is not None and len(examples_to_use) > random_few_shot_count:
            examples_to_use = random.sample(examples_to_use, random_few_shot_count)
        
        for example in examples_to_use:
            example_term = example.get("term", "")
            example_types = example.get("types", [])
            
            conversation.append({
                "role": "user",
                "content": f"Term: {example_term}"
            })
            conversation.append({
                "role": "assistant",
                "content": json.dumps({"types": example_types}, ensure_ascii=False)
            })
    
    # Add the actual query
    conversation.append({
        "role": "user",
        "content": f"Term: {term}"
    })
    
    return conversation


def build_hf_dataset(
    data_path: Path,
    tokenizer,
    few_shot_examples: List[Dict] = None,
    random_few_shot_count: Optional[int] = None
):
    """Build HuggingFace dataset for training."""
    from datasets import Dataset
    
    # Load the data
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    conversations = []
    
    for item in data:
        term = item.get("term", "")
        types = item.get("types", [])
        
        # Build conversation
        conversation = build_conversation_for_training(
            term, types, few_shot_examples, random_few_shot_count
        )
        
        # Apply chat template
        formatted_conversation = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=False,
            tokenize=False
        )
        
        conversations.append({"text": formatted_conversation})
    
    return Dataset.from_list(conversations)


def process_dataset_for_training(
    train_path: Path,
    tokenizer,
    few_shot_path: Optional[Path] = None,
    random_few_shot_count: Optional[int] = None
):
    """Process dataset for training with proper formatting."""
    
    # Load few-shot examples
    few_shot_examples = load_few_shot_examples(few_shot_path)
    
    # Build dataset
    dataset = build_hf_dataset(
        train_path,
        tokenizer,
        few_shot_examples,
        random_few_shot_count
    )
    
    return dataset 