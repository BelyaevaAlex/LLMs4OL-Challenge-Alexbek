import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import torch

SYSTEM_PROMPT = (
    "You are an expert in ontology term extraction.\n\n"
    "TASK: Extract specific, relevant ontology terms from scientific documents.\n\n"
    "INSTRUCTIONS:\n"
    "- The following conversation contains few-shot examples showing correct term extraction patterns\n"
    "- Study these examples carefully to understand the extraction style and approach\n"
    "- Follow the EXACT same pattern and style demonstrated in the examples\n"
    "- Extract only terms that actually appear in the document text\n"
    "- Focus on domain-specific terminology, concepts, and technical terms\n\n"
    "- The first three user-assistant conversation pairs serve as few-shot examples\n"
    "- Each example shows: user provides a document, assistant extracts relevant terms\n"
    "- Pay attention to the extraction patterns and term selection criteria in these examples\n\n"
    "DO:\n"
    "- Extract terms that are EXPLICITLY mentioned in the LAST document\n"
    "- Follow the SAME extraction pattern as shown in examples\n"
    "- Return unique terms without duplicates\n"
    "- Use the same JSON format as demonstrated\n\n"
    "DON'T:\n"
    "- Hallucinate or invent terms not present in last the document\n"
    "- Repeat the same term multiple times\n"
    "- Deviate from the extraction style shown in examples\n\n"
    "OUTPUT FORMAT: Return a JSON object with a single field 'terms' containing a list of extracted terms."
)


def load_reverse_dataset(path: Path) -> List[Dict]:
    """Load reverse dataset with OL field."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_few_shot_examples(path: Optional[Path]) -> List[Dict]:
    """Load few-shot examples from JSONL file."""
    if not path or not path.exists():
        return []
    
    examples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def format_document_text(title: str, text: str, tfidf_suggestions: List[str] = None) -> str:
    """Format document text with optional TF-IDF suggestions."""
    formatted_text = f"Title: {title}\n{text}"
    
    if tfidf_suggestions:
        formatted_text += f"\nTF-IDF based suggestions: {tfidf_suggestions}"
    
    return formatted_text


def build_conversation_for_training(doc: Dict, few_shot_examples: List[Dict] = None, use_tfidf: bool = False, random_few_shot_count: Optional[int] = None) -> List[Dict]:
    """Build conversation format for training (universal for all models)."""
    conversation = [{
        "role": "system",
        "content": SYSTEM_PROMPT
    }]
    
    # Add few-shot examples as previous conversation turns
    if few_shot_examples:
        import random
        examples_to_use = few_shot_examples
        if random_few_shot_count is not None and len(few_shot_examples) > random_few_shot_count:
            examples_to_use = random.sample(few_shot_examples, random_few_shot_count)
        
        for example in examples_to_use:
            # Include TF-IDF in examples if available and requested
            tfidf_suggestions = None
            if use_tfidf and "TF-IDF" in example:
                tfidf_suggestions = example.get("TF-IDF", [])
            
            user_input = f"Title: {example['title']}\n{example['text']}"

            if tfidf_suggestions:
                user_input += f"\nTF-IDF based suggestions: {tfidf_suggestions}"
            assistant_output = json.dumps({"terms": example['OL']}, ensure_ascii=False)
            
            conversation.extend([
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant", 
                    "content": assistant_output
                }
            ])
    
    # Add current document
    title = doc.get("title", "")
    text = doc.get("text", "")
    terms = doc.get("OL", [])
    
    # Include TF-IDF for current document if available and requested
    tfidf_suggestions = None
    if use_tfidf and "TF-IDF" in doc:
        tfidf_suggestions = doc.get("TF-IDF", [])
    
    user_input = format_document_text(title, text, tfidf_suggestions)
    assistant_output = json.dumps({"terms": terms}, ensure_ascii=False)
    
    conversation.extend([
        {
            "role": "user",
            "content": user_input
        },
        {
            "role": "assistant",
            "content": assistant_output
        }
    ])
    
    return conversation


def build_conversation_for_inference(title: str, text: str, few_shot_examples: List[Dict] = None, use_tfidf: bool = False, tfidf_suggestions: List[str] = None, random_few_shot_count: Optional[int] = None) -> List[Dict]:
    """Build conversation format for inference (universal for all models)."""
    conversation = [{
        "role": "system",
        "content": SYSTEM_PROMPT
    }]
    
    # Add few-shot examples as previous conversation turns
    if few_shot_examples:
        import random
        examples_to_use = few_shot_examples
        if random_few_shot_count is not None and len(few_shot_examples) > random_few_shot_count:
            examples_to_use = random.sample(few_shot_examples, random_few_shot_count)
        
        for example in examples_to_use:
            # Include TF-IDF in examples if available and requested
            example_tfidf = None
            if use_tfidf and "TF-IDF" in example:
                example_tfidf = example.get("TF-IDF", [])
            
            user_input = format_document_text(
                example['title'], 
                example['text'], 
                example_tfidf
            )
            assistant_output = json.dumps({"terms": example['OL']}, ensure_ascii=False)
            
            conversation.extend([
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant", 
                    "content": assistant_output
                }
            ])
    
    # Add current document for inference (without assistant response)
    user_input = format_document_text(title, text, tfidf_suggestions if use_tfidf else None)
    conversation.append({
        "role": "user",
        "content": user_input
    })
    
    return conversation


def create_attention_mask_for_training(tokenized_text: str, tokenizer, conversation: List[Dict], mask_few_shot: bool = True, mask_only_terms: bool = False) -> Tuple[List[int], List[int]]:
    """
    Create attention mask for training to mask few-shot examples and user input.
    
    Args:
        tokenized_text: Full tokenized conversation text
        tokenizer: Tokenizer instance
        conversation: Original conversation structure
        mask_few_shot: Whether to mask few-shot examples
        mask_only_terms: Whether to mask everything except terms from OL
        
    Returns:
        input_ids: Tokenized input
        labels: Labels with masked tokens set to -100
    """
    tokens = tokenizer(tokenized_text, return_tensors="pt", add_special_tokens=False)
    input_ids = tokens["input_ids"][0].tolist()
    labels = input_ids.copy()
    
    if mask_few_shot or mask_only_terms:
        # Reconstruct conversation parts to identify what to mask
        current_pos = 0
        
        for i, turn in enumerate(conversation):
            turn_text = tokenizer.apply_chat_template(
                [turn], 
                tokenize=False, 
                add_generation_prompt=(turn["role"] == "user" and i == len(conversation) - 1)
            )
            turn_tokens = tokenizer(turn_text, add_special_tokens=False)["input_ids"]
            turn_length = len(turn_tokens)
            
            # Determine if this turn should be masked
            should_mask = False
            
            if mask_few_shot:
                # Mask few-shot examples (all but the last user-assistant pair)
                if i < len(conversation) - 2:  # Not the last training pair
                    should_mask = True
                # Also mask user input in the last pair
                elif i == len(conversation) - 2 and turn["role"] == "user":
                    should_mask = True
            
            if should_mask:
                # Mask this entire turn
                for j in range(current_pos, min(current_pos + turn_length, len(labels))):
                    labels[j] = -100
            elif mask_only_terms and turn["role"] == "assistant":
                # For assistant responses, only keep terms from OL, mask everything else
                # This is more complex and requires parsing the JSON structure
                assistant_content = turn["content"]
                try:
                    parsed_content = json.loads(assistant_content)
                    terms = parsed_content.get("terms", [])
                    
                    # Find term positions in the tokenized assistant response
                    # For simplicity, mask everything except the terms
                    # This is a simplified approach - could be improved
                    term_positions = []
                    for term in terms:
                        term_tokens = tokenizer(term, add_special_tokens=False)["input_ids"]
                        # Find term positions in the turn (simplified)
                        # In practice, you'd want more sophisticated matching
                    
                    # For now, keep the entire assistant response if mask_only_terms
                    # A more sophisticated implementation would selectively mask
                    pass
                except json.JSONDecodeError:
                    # If can't parse JSON, mask everything
                    for j in range(current_pos, min(current_pos + turn_length, len(labels))):
                        labels[j] = -100
            
            current_pos += turn_length
    
    return input_ids, labels


def build_hf_dataset(data: List[Dict], tokenizer, few_shot_examples: List[Dict] = None, use_tfidf=False, mask_few_shot: bool = True, random_few_shot_count: Optional[int] = None):
    """Convert raw documents to a HuggingFace Dataset using conversation format with proper masking."""
    from datasets import Dataset

    all_input_ids = []
    all_labels = []
    
    for doc in data:
        # Handle TF-IDF suggestions if needed
        doc_copy = doc.copy()
        if use_tfidf and "TF-IDF" in doc_copy:
            # TF-IDF is already in the document, will be used in conversation building
            pass
        
        # Build conversation with random few-shot selection
        conversation = build_conversation_for_training(doc_copy, few_shot_examples, use_tfidf, random_few_shot_count)
        
        # Apply chat template to get formatted text
        formatted_text = tokenizer.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        if mask_few_shot:
            # Create properly masked labels
            input_ids, labels = create_attention_mask_for_training(
                formatted_text, tokenizer, conversation, mask_few_shot=True
            )
            all_input_ids.append(input_ids)
            all_labels.append(labels)
        else:
            # Standard approach without masking
            tokens = tokenizer(
                formatted_text, 
                truncation=True, 
                max_length=2048,
                add_special_tokens=False
            )
            all_input_ids.append(tokens["input_ids"])
            all_labels.append(tokens["input_ids"])  # For causal LM, labels = input_ids

    return Dataset.from_dict({
        "input_ids": all_input_ids,
        "labels": all_labels
    })


# Legacy function for backward compatibility with Llama-specific format
def build_inference_prompt(title: str, text: str, tfidf_bs: List[str] = []) -> str:
    """Build prompt for inference with Llama model (legacy function)."""
    instruction = "Extract ontology terms from the document. Return a JSON object with a single field 'terms' listing the terms."
    user_input = f"Title: {title}\n Text: {text}"
    if tfidf_bs:
        user_input += f"\n TF-IDF based suggestions: {tfidf_bs}"
    
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
