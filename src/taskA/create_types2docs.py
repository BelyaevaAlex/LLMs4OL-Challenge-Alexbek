import json
import os
import argparse
from collections import defaultdict
from typing import Dict, List, Set

def normalize_term(term: str) -> str:
    """Normalize term by converting to lowercase and removing extra spaces"""
    return ' '.join(term.lower().split())

def load_json_file(filepath: str) -> dict:
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(data: dict, filepath: str):
    """Save data to JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def create_types2docs(terms2docs: Dict[str, List[str]], terms2types: List[Dict[str, List[str]]]) -> Dict[str, List[str]]:
    """Create types2docs mapping from terms2docs and terms2types"""
    # Create term to types mapping for faster lookup with normalized terms
    term2types = {}
    for item in terms2types:
        normalized_term = normalize_term(item['term'])
        term2types[normalized_term] = item['types']
    
    # Initialize defaultdict to collect docs for each type
    types2docs = defaultdict(set)
    
    # Process each term and its documents
    for term, docs in terms2docs.items():
        # Normalize the term
        normalized_term = normalize_term(term)
        # Get types for the term if they exist
        if normalized_term in term2types:
            types = term2types[normalized_term]
            # Add documents to each type
            for type_ in types:
                types2docs[type_].update(docs)
    
    # Convert sets to sorted lists for JSON serialization
    return {type_: sorted(list(docs)) for type_, docs in types2docs.items()}

def process_directory(directory: str, output_dir: str):
    """Process directory to create types2docs mapping"""
    train_dir = os.path.join(directory, 'train')
    output_dir = os.path.join(output_dir, 'train')
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if required files exist
    terms2docs_path = os.path.join(train_dir, 'terms2docs.json')
    terms2types_path = os.path.join(train_dir, 'terms2types.json')
    
    if not (os.path.exists(terms2docs_path) and os.path.exists(terms2types_path)):
        print(f"Required files not found in {directory}")
        return
    
    # Load input files
    terms2docs = load_json_file(terms2docs_path)
    terms2types = load_json_file(terms2types_path)
    
    # Generate types2docs mapping
    types2docs = create_types2docs(terms2docs, terms2types)
    
    # Print some statistics
    print(f"\nStatistics for {directory}:")
    print(f"Number of terms in terms2docs: {len(terms2docs)}")
    print(f"Number of terms in terms2types: {len(terms2types)}")
    print(f"Number of types in types2docs: {len(types2docs)}")
    
    # Save result
    output_path = os.path.join(output_dir, 'types2docs.json')
    save_json_file(types2docs, output_path)
    print(f"Created types2docs.json in {directory}")

def main():
    # Batch mode - process all domains (legacy)
    base_dir = '../../2025/TaskA-Text2Onto'
    base_output_dir = '../../2025/TaskA-Text2Onto-Processed'
    subdirs = ['scholarly', 'engineering', 'ecology']
    
    for subdir in subdirs:
        directory = os.path.join(base_dir, subdir)
        output_dir = os.path.join(base_output_dir, subdir)
        print(f"\nProcessing {subdir}...")
        process_directory(directory, output_dir)


if __name__ == '__main__':
    main() 