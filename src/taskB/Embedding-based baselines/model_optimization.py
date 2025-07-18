import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
import networkx as nx
from typing import Dict, List, Tuple
import pandas as pd
import os

def create_term_graph(train_data: List[Dict]) -> nx.Graph:
    """Create term graph based on RAG data"""
    G = nx.Graph()
    
    # Add nodes (terms) and their types
    for item in train_data:
        G.add_node(item['term'], types=item['types'])
        
        # Add edges based on RAG
        if 'RAG' in item:
            for rag_item in item['RAG']:
                # Edge weight - number of common types
                common_types = set(item['types']).intersection(set(rag_item['types']))
                weight = len(common_types) if common_types else 0.1
                G.add_edge(item['term'], rag_item['term'], weight=weight)
    
    return G

def extract_graph_features(G: nx.Graph, terms: List[str]) -> np.ndarray:
    """Extract graph features for each term"""
    features = []
    for term in terms:
        if term not in G:
            # If term not found in graph, use zero features
            features.append([0.0] * 4)
            continue
            
        # Calculate graph features
        degree = G.degree(term)
        clustering = nx.clustering(G, term)
        centrality = nx.degree_centrality(G)[term]
        pagerank = nx.pagerank(G)[term]
        
        features.append([
            degree,
            clustering,
            centrality,
            pagerank
        ])
    
    return np.array(features)

def get_base_classifiers():
    """Get base classifiers"""
    return {
        'random_forest': OneVsRestClassifier(RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            class_weight='balanced',
            random_state=42
        )),
        'random_forest_with_graph': OneVsRestClassifier(RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            class_weight='balanced',
            random_state=42
        ))
    }

def save_comparison_metrics(metrics_dict: Dict, domain: str, save_dir: str):
    """Save comparison metrics to file"""
    output_file = os.path.join(save_dir, f"metrics_comparison_{domain}.txt")
    
    with open(output_file, 'w') as f:
        f.write(f"Method comparison for domain {domain}\n\n")
        
        # Results without graph features
        f.write("\n=== Without graph features ===\n")
        if 'random_forest' in metrics_dict:
            f.write("\nRANDOM_FOREST:")
            for metric_name, value in metrics_dict['random_forest'].items():
                f.write(f"\n{metric_name}: {value:.3f}")
            f.write("\n")
        
        # Results with graph features
        f.write("\n=== With graph features ===\n")
        if 'random_forest_with_graph' in metrics_dict:
            f.write("\nRANDOM_FOREST_WITH_GRAPH:")
            for metric_name, value in metrics_dict['random_forest_with_graph'].items():
                f.write(f"\n{metric_name}: {value:.3f}")
            f.write("\n") 