#!/usr/bin/env python3
"""Data processing module for TaskB-TermTyping with RAG support."""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Any

# System prompt для задачи TaskB-TermTyping с подробными инструкциями
SYSTEM_PROMPT = """You are an expert in ontologies and semantic term classification. Your task is to determine semantic types for given terms based on the domain-specific ontology.

CRITICAL INSTRUCTIONS:
1. Analyze the provided term carefully
2. Determine the most appropriate semantic types from the domain-specific ontology
3. A semantic type is a GENERALIZING CONCEPT or CATEGORY that the term belongs to
4. Use the provided examples to understand the patterns and relationships between terms and their types
5. Respond ONLY in JSON format using \" quotes
6. The reasoning should be concise and to the point, not too long. Write not more than 100 words in reasoning.

RESPONSE FORMAT:
{
  "term": "term name",
  "reasoning": "term name is similar to provided examples, but is not like the first example because ..., so it is not type1, but type2",
  "types": ["type2"]
}

The types should be relevant semantic categories that best describe the given term according to the domain ontology."""


def load_few_shot_examples(few_shot_path: Optional[Path]) -> List[Dict]:
    """Load few-shot examples from JSON file."""
    if not few_shot_path or not few_shot_path.exists():
        return []
    
    with open(few_shot_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to list if it's a single dict
    if isinstance(data, dict):
        return [data]
    elif isinstance(data, list):
        return data
    else:
        return []


def create_user_prompt_with_examples(term: str, examples: List[Dict] = None) -> str:
    """Create user prompt with few-shot examples embedded in the prompt."""
    
    # Few-shot examples
    examples_text = ""
    if examples:
        examples_text = "\n\nCLASSIFICATION EXAMPLES:\n"
        for i, example in enumerate(examples, 1):
            example_term = example.get("term", "")
            example_types = example.get("types", [])
            examples_text += f"{i}. Term: '{example_term}' → Types: {example_types}\n"
        examples_text += "\nEND OF EXAMPLES.\n"
    
    user_prompt = f"""{examples_text}
TERM: {term}

TASK: Determine semantic types for the given term based on the domain ontology. Remember that semantic types should be generalizing categories, not the term itself. Provide reasoning for your choice and respond in JSON format."""
    
    return user_prompt


def build_conversation_for_training(
    term: str,
    types: List[str],
    reasoning: str = "",
    few_shot_examples: List[Dict] = None,
    few_shot_amount: Optional[int] = None,
    use_rag: bool = False,
    rag_examples: List[Dict] = None
) -> List[Dict[str, str]]:
    """Build conversation format for training."""
    
    # Choose examples to use
    examples_to_use = few_shot_examples
    if use_rag and rag_examples:
        examples_to_use = rag_examples
    
    # Limit examples if needed
    if examples_to_use and few_shot_amount is not None and len(examples_to_use) > few_shot_amount:
        examples_to_use = examples_to_use[:few_shot_amount]  # Take first N examples (sorted by semantic similarity)
    
    # Create user prompt with embedded examples
    user_prompt = create_user_prompt_with_examples(term, examples_to_use)
    
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": json.dumps({
            "reasoning": reasoning,
            "types": types
        }, ensure_ascii=False)}
    ]
    
    return conversation


def build_conversation_for_inference(
    term: str,
    few_shot_examples: List[Dict] = None,
    few_shot_amount: Optional[int] = None,
    use_rag: bool = False,
    rag_examples: List[Dict] = None
) -> List[Dict[str, str]]:
    """Build conversation format for inference."""
    
    # Choose examples to use
    examples_to_use = few_shot_examples
    if use_rag and rag_examples:
        examples_to_use = rag_examples
    
    # Limit examples if needed
    if examples_to_use and few_shot_amount is not None and len(examples_to_use) > few_shot_amount:
        examples_to_use = examples_to_use[:few_shot_amount]  # Take first N examples (sorted by semantic similarity)
    
    # Create user prompt with embedded examples
    user_prompt = create_user_prompt_with_examples(term, examples_to_use)
    
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    return conversation


def build_hf_dataset(
    data_path: Path,
    tokenizer,
    few_shot_examples: List[Dict] = None,
    few_shot_amount: Optional[int] = None,
    use_rag: bool = False
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
        reasoning = item.get("reasoning", "")  # Extract reasoning if available
        
        # Extract RAG examples if use_rag is enabled
        rag_examples = None
        if use_rag and "RAG" in item:
            rag_examples = item["RAG"]
        
        # Build conversation
        conversation = build_conversation_for_training(
            term, types, reasoning, few_shot_examples, few_shot_amount, use_rag, rag_examples
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
    few_shot_amount: Optional[int] = None,
    use_rag: bool = False
):
    """Process dataset for training with proper formatting."""
    
    # Load few-shot examples
    few_shot_examples = load_few_shot_examples(few_shot_path)
    
    # Build dataset
    dataset = build_hf_dataset(
        train_path,
        tokenizer,
        few_shot_examples,
        few_shot_amount,
        use_rag
    )
    
    return dataset 