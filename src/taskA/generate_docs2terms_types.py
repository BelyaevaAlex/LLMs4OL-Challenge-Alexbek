#!/usr/bin/env python3
"""
Script for generating docs2terms_types for all domains TaskA-Text2Onto
Creates a dataset where types and terms are mapped for each document
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

def load_domain_data(domain_path):
    """Loads data for one domain"""
    print(f"Loading data for domain: {domain_path}")
    
    # Load terms.txt
    terms_file = domain_path / "terms.txt"
    if terms_file.exists():
        with open(terms_file, 'r', encoding='utf-8') as f:
            terms_from_file = set(line.strip() for line in f if line.strip())
    else:
        terms_from_file = set()
    
    # Load types.txt
    types_file = domain_path / "types.txt"
    if types_file.exists():
        with open(types_file, 'r', encoding='utf-8') as f:
            types_from_file = set(line.strip() for line in f if line.strip())
    else:
        types_from_file = set()
    
    # Load terms2docs.json (treated as types2docs)
    terms2docs_file = domain_path / "terms2docs.json"
    if terms2docs_file.exists():
        with open(terms2docs_file, 'r', encoding='utf-8') as f:
            types2docs = json.load(f)
    else:
        types2docs = {}
    
    # Load terms2types.json
    terms2types_file = domain_path / "terms2types.json"
    if terms2types_file.exists():
        with open(terms2types_file, 'r', encoding='utf-8') as f:
            terms2types_data = json.load(f)
    else:
        terms2types_data = []
    
    # Convert terms2types to convenient format
    terms2types = {}
    types2terms = defaultdict(set)
    
    for entry in terms2types_data:
        term = entry['term']
        types = entry['types']
        terms2types[term] = types
        for type_name in types:
            types2terms[type_name].add(term)
    
    # Load documents
    documents_file = domain_path / "documents.jsonl"
    documents = []
    if documents_file.exists():
        with open(documents_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    documents.append(json.loads(line))
    
    return {
        'terms_from_file': terms_from_file,
        'types_from_file': types_from_file,
        'types2docs': types2docs,
        'terms2types': terms2types,
        'types2terms': types2terms,
        'documents': documents
    }

def save_dataset(dataset, output_path, domain_name):
    """Saves dataset in various formats"""
    
    # Create directory if it doesn't exist
    output_dir = output_path / domain_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON
    json_file = output_dir / "train" / "docs2terms_types.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    # Save to JSONL for convenience
    jsonl_file = output_dir / "train" / "docs2terms_types.jsonl"
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Create summary statistics
    stats = {
        'domain': domain_name,
        'total_documents': len(dataset),
        'total_unique_types': len(set(type_name for entry in dataset for type_name in entry['types'])),
        'total_unique_terms': len(set(term for entry in dataset for term in entry['terms'])),
        'avg_types_per_doc': sum(entry['types_count'] for entry in dataset) / len(dataset) if dataset else 0,
        'avg_terms_per_doc': sum(entry['terms_count'] for entry in dataset) / len(dataset) if dataset else 0,
        'max_types_per_doc': max(entry['types_count'] for entry in dataset) if dataset else 0,
        'max_terms_per_doc': max(entry['terms_count'] for entry in dataset) if dataset else 0,
        'min_types_per_doc': min(entry['types_count'] for entry in dataset) if dataset else 0,
        'min_terms_per_doc': min(entry['terms_count'] for entry in dataset) if dataset else 0
    }
    
    # Save statistics
    stats_file = output_dir / "dataset_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # Create CSV for analysis
    csv_data = []
    for entry in dataset:
        csv_data.append({
            'id': entry['id'],
            'title': entry['title'][:100] + '...' if len(entry['title']) > 100 else entry['title'],
            'text_length': len(entry['text']),
            'types_count': entry['types_count'],
            'terms_count': entry['terms_count'],
            'types': '; '.join(entry['types'][:5]) + ('...' if len(entry['types']) > 5 else ''),
            'terms': '; '.join(entry['terms'][:5]) + ('...' if len(entry['terms']) > 5 else '')
        })
    
    df = pd.DataFrame(csv_data)
    csv_file = output_dir / "docs2terms_types_summary.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8')
    
    return json_file, jsonl_file, stats_file, csv_file, stats

def create_docs2terms_types(data):
    """Creates docs2terms_types dataset without filtering"""
    
    # Create document index
    doc_index = {doc['id']: doc for doc in data['documents']}
    
    # 1. Create mapping docs -> types from types2docs
    docs2types = defaultdict(set)
    for type_name, doc_list in data['types2docs'].items():
        for doc_id in doc_list:
            docs2types[doc_id].add(type_name)
    
    # 2. Create mapping docs -> terms through types
    docs2terms = defaultdict(set)
    for doc_id, types_in_doc in docs2types.items():
        for type_name in types_in_doc:
            # Add all terms of this type (if they exist)
            if type_name in data['types2terms']:
                docs2terms[doc_id].update(data['types2terms'][type_name])
    
    # 3. Create final dataset
    dataset = []
    
    for doc in data['documents']:
        doc_id = doc['id']
        doc_types = list(docs2types[doc_id])
        doc_terms = list(docs2terms[doc_id])
        
        doc['types'] = doc_types
        doc['terms'] = doc_terms
        doc['types_count'] = len(doc_types)
        doc['terms_count'] = len(doc_terms)
        
        dataset.append(doc)

    return dataset

def main():
    """Main function"""
    
    # Domains to process
    domains = ["engineering", "ecology", "scholarly"]
    
    base_data_path = Path("../../2025/TaskA-Text2Onto-Processed")
    
    # Output directory
    output_path = base_data_path
    output_path.mkdir(exist_ok=True)
    
    # Overall statistics
    overall_stats = []
    
    print("="*80)
    print("GENERATING DOCS2TERMS_TYPES FOR ALL DOMAINS")
    print("="*80)
    
    for domain in domains:
        print(f"\n{'='*60}")
        print(f"PROCESSING DOMAIN: {domain.upper()}")
        print(f"{'='*60}")
        
        # Domain data path
        domain_path = base_data_path / domain / "train"
        
        if not domain_path.exists():
            print(f"⚠️  Directory not found: {domain_path}")
            continue
        
        try:
            # Load data
            data = load_domain_data(domain_path)
            
            print(f"📊 Loaded data statistics:")
            print(f"   Documents: {len(data['documents'])}")
            print(f"   Types in types2docs: {len(data['types2docs'])}")
            print(f"   Terms in terms2types: {len(data['terms2types'])}")
            print(f"   Types in terms2types: {len(data['types2terms'])}")
            
            # Create dataset
            print("🔄 Creating dataset...")
            dataset = create_docs2terms_types(data)
            
            # Additional debug information
            print(f"📋 Additional verification:")
            print(f"   Documents in dataset: {len(dataset)}")
            
            # Check type coverage
            all_types_in_dataset = set()
            for entry in dataset:
                all_types_in_dataset.update(entry['types'])
            print(f"   Unique types in dataset: {len(all_types_in_dataset)}")
            print(f"   Total types in types2docs: {len(data['types2docs'])}")
            
            # Check if there are documents without terms
            docs_without_terms = sum(1 for entry in dataset if entry['terms_count'] == 0)
            print(f"   Documents without terms: {docs_without_terms}")
            
            if len(all_types_in_dataset) != len(data['types2docs']):
                missing_types = set(data['types2docs'].keys()) - all_types_in_dataset
                print(f"   ⚠️  Missing types: {len(missing_types)}")
                if missing_types:
                    print(f"   First 5 missing types: {list(missing_types)[:5]}")
                    
            # Check documents
            docs_in_types2docs = set()
            for doc_list in data['types2docs'].values():
                docs_in_types2docs.update(doc_list)
            docs_in_dataset = set(entry['id'] for entry in dataset)
            print(f"   Total documents in types2docs: {len(docs_in_types2docs)}")
            print(f"   Documents in dataset: {len(docs_in_dataset)}")
            
            if len(docs_in_dataset) != len(docs_in_types2docs):
                missing_docs = docs_in_types2docs - docs_in_dataset
                print(f"   ⚠️  Missing documents: {len(missing_docs)}")
                if missing_docs:
                    print(f"   First 5 missing documents: {list(missing_docs)[:5]}")
            
            # Save
            print("💾 Saving results...")
            json_file, jsonl_file, stats_file, csv_file, stats = save_dataset(dataset, output_path, domain)
            
            # Output results
            print(f"✅ Dataset for domain '{domain}' created successfully!")
            print(f"   📄 JSON file: {json_file}")
            print(f"   📄 JSONL file: {jsonl_file}")
            print(f"   📊 Statistics: {stats_file}")
            print(f"   📈 CSV summary: {csv_file}")
            
            print(f"\n📈 Dataset statistics:")
            print(f"   Documents: {stats['total_documents']}")
            print(f"   Unique types: {stats['total_unique_types']}")
            print(f"   Unique terms: {stats['total_unique_terms']}")
            print(f"   Average types per document: {stats['avg_types_per_doc']:.2f}")
            print(f"   Average terms per document: {stats['avg_terms_per_doc']:.2f}")
            
            overall_stats.append(stats)
            
        except Exception as e:
            print(f"❌ Error processing domain '{domain}': {str(e)}")
            continue
    
    # Save overall statistics
    if overall_stats:
        overall_stats_file = output_path / "overall_stats.json"
        with open(overall_stats_file, 'w', encoding='utf-8') as f:
            json.dump(overall_stats, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*80}")
        print("OVERALL STATISTICS")
        print("="*80)
        
        total_docs = sum(stat['total_documents'] for stat in overall_stats)
        total_unique_types = sum(stat['total_unique_types'] for stat in overall_stats)
        total_unique_terms = sum(stat['total_unique_terms'] for stat in overall_stats)
        
        print(f"📊 Processed domains: {len(overall_stats)}")
        print(f"📊 Total documents: {total_docs}")
        print(f"📊 Total unique types: {total_unique_types}")
        print(f"📊 Total unique terms: {total_unique_terms}")
        print(f"📊 Overall statistics saved to: {overall_stats_file}")
        
        print(f"\n📋 Detailed statistics by domains:")
        for stat in overall_stats:
            print(f"   {stat['domain']}: {stat['total_documents']} documents, "
                  f"{stat['total_unique_types']} types, {stat['total_unique_terms']} terms")
    
    print(f"\n🎉 Generation completed! Results saved to: {output_path}")

if __name__ == "__main__":
    main() 