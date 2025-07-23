#!/usr/bin/env python3
"""
Script for creating reverse dataset for TaskA-Text2Onto.
Adds an OL field to each document with a list of terms found in that document.

Author: AI Assistant
Date: 2024
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict


def load_documents(documents_path: Path) -> Dict[str, Dict]:
    """Loads documents from JSONL file."""
    documents = {}
    with open(documents_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line.strip())
            documents[doc['id']] = doc
    return documents


def load_terms2docs(terms2docs_path: Path) -> Dict[str, List[str]]:
    """Loads mapping of terms to documents."""
    with open(terms2docs_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_reverse_dataset(documents: Dict[str, Dict], 
                          terms2docs: Dict[str, List[str]]) -> Dict[str, Dict]:
    """
    Creates reverse dataset by adding an OL field to each document 
    with a list of terms found in that document.
    
    Args:
        documents: Dictionary of documents
        terms2docs: Mapping of terms to documents
    """
    # Create reverse index: document -> list of terms
    docs2terms = defaultdict(list)
    for term, doc_ids in terms2docs.items():
        for doc_id in doc_ids:
            docs2terms[doc_id].append(term)
    
    # Create new dataset
    reverse_dataset = {}
    
    for doc_id, doc in documents.items():
        # Copy original document
        new_doc = doc.copy()
        
        # Create OL field with list of terms
        ol_terms = []
        
        if doc_id in docs2terms:
            ol_terms = docs2terms[doc_id]
        
        new_doc['OL'] = ol_terms
        reverse_dataset[doc_id] = new_doc
    
    return reverse_dataset


def save_reverse_dataset(reverse_dataset: Dict[str, Dict], output_path: Path):
    """Saves reverse dataset in JSONL format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc_id in sorted(reverse_dataset.keys()):
            doc = reverse_dataset[doc_id]
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')


def print_statistics(reverse_dataset: Dict[str, Dict], terms2docs: Dict[str, List[str]]):
    """Prints statistics for the created dataset."""
    total_docs = len(reverse_dataset)
    docs_with_terms = sum(1 for doc in reverse_dataset.values() if doc['OL'])
    total_terms = sum(len(doc['OL']) for doc in reverse_dataset.values())
    unique_terms = len(terms2docs)
    
    print(f"\n=== Data Statistics ===")
    print(f"Unique terms in terms2docs: {unique_terms}")
    
    print(f"\n=== Reverse Dataset Statistics ===")
    print(f"Total number of documents: {total_docs}")
    print(f"Documents with terms: {docs_with_terms}")
    print(f"Documents without terms: {total_docs - docs_with_terms}")
    print(f"Total number of terms: {total_terms}")
    print(f"Average number of terms per document: {total_terms / total_docs:.2f}")
    
    if docs_with_terms > 0:
        avg_terms_with_ol = total_terms / docs_with_terms
        print(f"Average number of terms per document with terms: {avg_terms_with_ol:.2f}")
    
    # Show examples of terms
    print(f"\nExamples of terms:")
    for i, term in enumerate(sorted(terms2docs.keys())):
        if i >= 5:  # Show only first 5
            break
        doc_count = len(terms2docs[term])
        print(f"  - {term}: {doc_count} documents")


def create_sample_output(reverse_dataset: Dict[str, Dict], output_path: Path, sample_size: int = 5):
    """Creates a file with examples for demonstration."""
    sample_path = output_path.parent / f"sample_{output_path.name}"
    
    # Find documents with terms for example
    docs_with_terms = [(doc_id, doc) for doc_id, doc in reverse_dataset.items() if doc['OL']]
    
    if docs_with_terms:
        print(f"\nCreating example file: {sample_path}")
        with open(sample_path, 'w', encoding='utf-8') as f:
            for i, (doc_id, doc) in enumerate(docs_with_terms[:sample_size]):
                f.write(json.dumps(doc, ensure_ascii=False, indent=2) + '\n')
                if i < len(docs_with_terms[:sample_size]) - 1:
                    f.write('\n')
    else:
        print("\nNo documents with terms to create examples.")


def main():
    domains = ["ecology", "engineering", "scholarly"]
    
    for domain in domains:
        input_dir = Path(f"../../2025/TaskA-Text2Onto-Processed/{domain}/train")
        output_dir = Path(f"../../2025/TaskA-Text2Onto-Processed/{domain}/train")
        
        documents_file = input_dir / "documents.jsonl"
        terms2docs_file = input_dir / "terms2docs.json"
        
        if documents_file.exists() and terms2docs_file.exists():
            
            print("Loading data...")
            
            # Load data
            documents = load_documents(documents_file)
            terms2docs = load_terms2docs(terms2docs_file)
            
            print(f"Loaded documents: {len(documents)}")
            print(f"Loaded terms in terms2docs: {len(terms2docs)}")
            
            print("Creating reverse dataset...")
            
            # Create reverse dataset
            reverse_dataset = create_reverse_dataset(documents, terms2docs)
            
            print(f"Saving to {output_dir}...")
            
            # Save result
            save_reverse_dataset(reverse_dataset, output_dir / "docs2terms.jsonl")
            
            # Print statistics
            print_statistics(reverse_dataset, terms2docs)
            
            # Create examples if requested
            create_sample_output(reverse_dataset, output_dir)
            
            print(f"\nReverse dataset successfully created: {output_dir}")
        else:
            raise ValueError(f"Error: File {documents_file} or {terms2docs_file} not found!")


if __name__ == "__main__":
    exit(main()) 