#!/usr/bin/env python3
"""
Term-to-Type RAG Implementation
Creates similarity matrices and terms2types.json with RAG examples for ontology learning.
"""

import json
import pandas as pd
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import gc
from typing import List, Dict, Any


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Pooling function for obtaining embeddings"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Creates instruction for embedding model"""
    return f'Instruct: {task_description}\nQuery: {query}'


def get_embeddings_batch(texts: List[str], model, tokenizer, max_length=8192, batch_size=8):
    """Get embeddings for texts in batches"""
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize batch
        batch_tokenized = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch_tokenized = {k: v.to(model.device) for k, v in batch_tokenized.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**batch_tokenized)
            batch_embeddings = last_token_pool(outputs.last_hidden_state, batch_tokenized['attention_mask'])
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            all_embeddings.append(batch_embeddings.cpu())
        
        # Clean memory
        del batch_tokenized, outputs, batch_embeddings
        torch.cuda.empty_cache()
        gc.collect()
    
    return torch.cat(all_embeddings, dim=0)


def compute_similarity_matrix(embeddings1, embeddings2, batch_size=100):
    """Compute similarity matrix between two sets of embeddings in batches"""
    n1, n2 = embeddings1.shape[0], embeddings2.shape[0]
    similarity_matrix = torch.zeros(n1, n2)
    
    for i in tqdm(range(0, n1, batch_size), desc="Computing similarity"):
        end_i = min(i + batch_size, n1)
        batch1 = embeddings1[i:end_i]
        
        for j in range(0, n2, batch_size):
            end_j = min(j + batch_size, n2)
            batch2 = embeddings2[j:end_j]
            
            # Compute similarity for batch
            similarity_batch = torch.mm(batch1, batch2.T)
            similarity_matrix[i:end_i, j:end_j] = similarity_batch
    
    return similarity_matrix


def load_terms2types_data(file_path: str) -> List[Dict[str, Any]]:
    """Load terms2types data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_test_terms(file_path: str) -> List[str]:
    """Load test terms from txt file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def find_top_similar_terms(scores_df: pd.DataFrame, term: str, top_k: int = 10, exclude_self: bool = True) -> List[str]:
    """Find top_k most similar terms"""
    if term not in scores_df.index:
        print(f"Warning: term '{term}' not found in scores matrix")
        return []
    
    scores_row = scores_df.loc[term]
    
    # Exclude self if needed
    if exclude_self and term in scores_df.columns:
        scores_row = scores_row.drop(term, errors='ignore')
    
    if scores_row.empty:
        return []
    
    top_terms = scores_row.nlargest(top_k).index.tolist()
    return top_terms


def process_domain(domain: str, train_data: List[Dict], test_data: List[Dict] = None, 
                  model=None, tokenizer=None, output_dir: str = "./data"):
    """Process one domain: get embeddings and similarity matrices"""
    print(f"\n=== Processing domain: {domain} ===")
    
    # Create output directory
    domain_dir = os.path.join(output_dir, domain)
    os.makedirs(domain_dir, exist_ok=True)
    
    # Prepare texts for embeddings
    instruction = "Given a term, find similar terms that have related semantic types or categories."
    
    # Train data
    train_terms = [item['term'] for item in train_data]
    train_texts_with_types = [f"Term: {item['term']}, Types: {', '.join(item['types'])}" for item in train_data]
    train_texts_instruct = [get_detailed_instruct(instruction, text) for text in train_texts_with_types]
    
    print(f"Train: {len(train_terms)} terms")
    
    # Get embeddings for train
    print("Getting train embeddings...")
    train_embeddings = get_embeddings_batch(train_texts_instruct, model, tokenizer, batch_size=8)
    
    # Train/Train matrix
    print("Computing Train/Train matrix...")
    train_train_scores = compute_similarity_matrix(train_embeddings, train_embeddings, batch_size=100)
    
    # Save Train/Train matrix
    train_train_df = pd.DataFrame(
        train_train_scores.numpy(), 
        columns=train_terms, 
        index=train_terms
    )
    train_train_path = os.path.join(domain_dir, f"{domain}_train_train_scores.csv")
    train_train_df.to_csv(train_train_path)
    print(f"Saved: {train_train_path}")
    
    # Test data (if available)
    if test_data:
        test_terms = [item['term'] for item in test_data]
        test_texts_with_types = [f"Term: {item['term']}" for item in test_data]  # Test has no types
        test_texts_instruct = [get_detailed_instruct(instruction, text) for text in test_texts_with_types]
        
        print(f"Test: {len(test_terms)} terms")
        
        # Get embeddings for test
        print("Getting test embeddings...")
        test_embeddings = get_embeddings_batch(test_texts_instruct, model, tokenizer, batch_size=8)
        
        # Test/Train matrix
        print("Computing Test/Train matrix...")
        test_train_scores = compute_similarity_matrix(test_embeddings, train_embeddings, batch_size=100)
        
        # Save Test/Train matrix
        test_train_df = pd.DataFrame(
            test_train_scores.numpy(), 
            columns=train_terms, 
            index=test_terms
        )
        test_train_path = os.path.join(domain_dir, f"{domain}_test_train_scores.csv")
        test_train_df.to_csv(test_train_path)
        print(f"Saved: {test_train_path}")
    
    # Clean memory
    del train_embeddings, train_train_scores
    if test_data:
        del test_embeddings, test_train_scores
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"Completed processing domain: {domain}")


def create_rag_data(domain: str, train_data: List[Dict], test_data: List[Dict] = None, 
                   output_dir: str = "./data"):
    """Create terms2types.json with RAG examples"""
    print(f"\n=== Creating RAG data for {domain} ===")
    
    domain_dir = os.path.join(output_dir, domain)
    
    # Create train dictionary without RAG field to avoid circular references
    train_dict = {}
    for item in train_data:
        clean_item = {k: v for k, v in item.items() if k != 'RAG'}
        train_dict[item['term']] = clean_item
    
    # Load train/train scores matrix
    train_scores_path = os.path.join(domain_dir, f"{domain}_train_train_scores.csv")
    if not os.path.exists(train_scores_path):
        print(f"Train scores matrix not found: {train_scores_path}")
        return
    
    train_scores_df = pd.read_csv(train_scores_path, index_col=0)
    
    # Process train data
    print(f"Processing {len(train_data)} train examples...")
    for item in tqdm(train_data, desc=f"Processing {domain} train"):
        term = item['term']
        
        # Find 10 most similar terms
        similar_terms = find_top_similar_terms(train_scores_df, term, top_k=10, exclude_self=True)
        
        # Add RAG examples (clean copies without RAG field)
        rag_examples = []
        for similar_term in similar_terms:
            if similar_term in train_dict:
                rag_examples.append(train_dict[similar_term].copy())
        
        item['RAG'] = rag_examples
    
    # Save train data
    train_output_path = os.path.join(domain_dir, "train", "terms2types.json")
    os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
    with open(train_output_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=1)
    print(f"Saved train data: {train_output_path}")
    
    # Process test data (if available)
    if test_data:
        test_scores_path = os.path.join(domain_dir, f"{domain}_test_train_scores.csv")
        
        if os.path.exists(test_scores_path):
            test_scores_df = pd.read_csv(test_scores_path, index_col=0)
            
            print(f"Processing {len(test_data)} test examples...")
            for item in tqdm(test_data, desc=f"Processing {domain} test"):
                term = item['term']
                
                if term in test_scores_df.index:
                    scores_row = test_scores_df.loc[term]
                    top_train_terms = scores_row.nlargest(10).index.tolist()
                    
                    rag_examples = []
                    for train_term in top_train_terms:
                        if train_term in train_dict:
                            rag_examples.append(train_dict[train_term].copy())
                    
                    item['RAG'] = rag_examples
                else:
                    item['RAG'] = []
            
            # Save test data
            test_output_path = os.path.join(domain_dir, "test", "terms2types.json")
            os.makedirs(os.path.dirname(test_output_path), exist_ok=True)
            with open(test_output_path, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, ensure_ascii=False, indent=1)
            print(f"Saved test data: {test_output_path}")
        else:
            print(f"Test scores not found: {test_scores_path}")
    
    print(f"Completed RAG data creation for {domain}")


def main():
    """Main function to process all domains"""
    # Data paths (relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_path = os.path.join(script_dir, "../../../2025/TaskA-Text2Onto")
    output_dir = os.path.join(script_dir, "./data")
    
    domains = ["ecology", "engineering", "scholarly"]
    
    # Load train and test data
    train_data = {}
    test_data = {}
    
    print("Loading train data...")
    for domain in domains:
        train_path = os.path.join(base_data_path, domain, "train", "terms2types.json")
        if os.path.exists(train_path):
            train_data[domain] = load_terms2types_data(train_path)
            print(f"{domain} train: {len(train_data[domain])} examples")
        else:
            print(f"Train data not found for {domain}")
    
    print("\nLoading test data...")
    for domain in domains:
        test_path = os.path.join(base_data_path, domain, "test", "docs2terms_test_results_rag_random_3_text2onto_{domain}_test_documents.jsonl_Qwen2.5_14B_Instruct_tfidf_structured.txt")
        if os.path.exists(test_path):
            test_terms = load_test_terms(test_path)
            test_data[domain] = [{"term": term, "types": []} for term in test_terms]
            print(f"{domain} test: {len(test_data[domain])} terms")
        else:
            print(f"Test data not found for {domain}")
    
    # Initialize embedding model
    print("\nInitializing embedding model...")
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-4B', padding_side='left')
    model = AutoModel.from_pretrained(
        'Qwen/Qwen3-Embedding-4B', 
        attn_implementation="flash_attention_2", 
        torch_dtype=torch.bfloat16
    ).cuda()
    
    print("Model loaded successfully!")
    
    # Process each domain
    for domain in domains:
        if domain in train_data:
            process_domain(domain, train_data[domain], test_data.get(domain), model, tokenizer, output_dir)
            create_rag_data(domain, train_data[domain], test_data.get(domain), output_dir)
        else:
            print(f"Skipping {domain} - no train data available")
    
    print("\n🎉 All processing completed!")


if __name__ == "__main__":
    main() 