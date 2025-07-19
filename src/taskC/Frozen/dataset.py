"""
CrossAttentionDataset - dataset for training Cross-Attention model
Supports loading entities.json and relations.json, grouping by datasets,
Train/Test Split and various sampling strategies.
"""

import json
import random
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Set, Optional, Any
from sklearn.model_selection import train_test_split
import numpy as np
from collections import defaultdict


class CrossAttentionDataset(Dataset):
    """
    Dataset for Cross-Attention model
    
    Args:
        entities_path: path to entities.json
        relations_path: path to relations.json  
        batch_size_1: size of first vector set (query)
        batch_size_2: size of second vector set (key)
        dataset_strategy: dataset selection strategy ("random", "sequential", "weighted", "single")
        mode: operation mode ("train", "test", "all")
        test_part: proportion of test data (for train/test mode)
        random_state: seed for reproducibility
    """
    
    def __init__(
        self,
        entities_path: str,
        relations_path: str,
        batch_size_1: int = 32,
        batch_size_2: int = 32,
        dataset_strategy: str = "single", # single, weighted (for dataset selection)
        sampling_strategy: str = "random", # random, balanced (for sampling within dataset)
        positive_ratio: float = 1.0, # proportion of positive pairs for balanced strategy (1.0 = maximum possible)
        mode: str = "all",
        test_part: float = 0.2,
        random_state: int = 42
    ):
        self.entities_path = entities_path
        self.relations_path = relations_path
        self.batch_size_1 = batch_size_1
        self.batch_size_2 = batch_size_2
        self.dataset_strategy = dataset_strategy
        self.sampling_strategy = sampling_strategy
        self.positive_ratio = positive_ratio
        self.mode = mode
        self.test_part = test_part
        self.random_state = random_state
        
        # Set seed
        random.seed(random_state)
        
        # Load data
        print("📁 Loading data...")
        self.entities = self._load_entities(entities_path)
        self.relations = self._load_relations(relations_path)
        
        # Group entities by datasets
        print("🔗 Grouping by datasets...")
        self.entities_by_dataset = self._group_entities_by_dataset()
        self.available_datasets = list(self.entities_by_dataset.keys())
        
        # Create indices for fast relation search by datasets
        print("📊 Indexing relations...")
        self.relations_by_dataset = self._group_relations_by_dataset()
        
        # Split into Train/Test if needed
        if mode in ["train", "test"]:
            print(f"✂️ Splitting into train/test ({mode} mode)...")
            self._split_train_test()
        else:
            # Build child -> parents index for fast search (for mode="all")
            self.child_to_parents = self._build_child_to_parents_index()
        
        # Dataset selection logic
        self.current_dataset_idx = 0
        
        # For weighted dataset selection
        if self.dataset_strategy == "weighted":
            self.dataset_weights = self._calculate_dataset_weights()
            
        # If only one dataset, ignore selection strategy
        if len(self.available_datasets) == 1:
            self.dataset_strategy = "single"
            
        # Print statistics
        self._print_statistics()
    
    def _load_entities(self, path: str) -> List[Dict[str, Any]]:
        """Load entities from entities.json"""
        with open(path, 'r', encoding='utf-8') as f:
            entities = json.load(f)
        
        # Validate data
        for i, entity in enumerate(entities):
            if 'id' not in entity:
                raise ValueError(f"Entity {i} does not contain 'id' field")
            if 'embedding' not in entity:
                raise ValueError(f"Entity {entity['id']} does not contain 'embedding' field")
            if 'text_description' not in entity:
                print(f"⚠️ Entity {entity['id']} does not contain 'text_description' field")
                
        print(f"✅ Loaded {len(entities)} entities")
        return entities
    
    def _load_relations(self, path: str) -> List[List[int]]:
        """Load relations from relations.json"""
        with open(path, 'r', encoding='utf-8') as f:
            relations_data = json.load(f)
            
        # Create mapping from text_description to ID
        text_to_id = {}
        for entity in self.entities:
            text_desc = entity.get('text_description', '')
            if text_desc:
                text_to_id[text_desc] = entity['id']
        
        relations = []
        skipped_relations = 0
        
        # Check data format
        if relations_data and isinstance(relations_data[0], dict):
            # New format: [{"ID": "...", "parent": "text", "child": "text"}, ...]
            for i, rel in enumerate(relations_data):
                if not isinstance(rel, dict):
                    raise ValueError(f"Relation {i} must be an object: {rel}")
                    
                parent_text = rel.get('parent', '')
                child_text = rel.get('child', '')
                
                if not parent_text or not child_text:
                    print(f"⚠️ Relation {i} does not contain parent or child")
                    skipped_relations += 1
                    continue
                    
                # Find ID by text_description
                parent_id = text_to_id.get(parent_text)
                child_id = text_to_id.get(child_text)
                
                if parent_id is None:
                    print(f"⚠️ ID not found for parent: '{parent_text}'")
                    skipped_relations += 1
                    continue
                    
                if child_id is None:
                    print(f"⚠️ ID not found for child: '{child_text}'")
                    skipped_relations += 1
                    continue
                    
                relations.append([parent_id, child_id])
        else:
            # Old format: [[id1, id2], ...]
            for i, rel in enumerate(relations_data):
                if not isinstance(rel, list) or len(rel) != 2:
                    raise ValueError(f"Relation {i} must be a list of 2 elements: {rel}")
                relations.append(rel)
                
        print(f"✅ Loaded {len(relations)} relations")
        if skipped_relations > 0:
            print(f"⚠️ Skipped {skipped_relations} relations (corresponding IDs not found)")
            
        return relations
    
    def _group_entities_by_dataset(self) -> Dict[str, List[Dict[str, Any]]]:
        """Group entities by datasets"""
        groups = {}
        for entity in self.entities:
            # If "dataset" field is missing, use "default"
            dataset = entity.get('dataset', 'default')
            if dataset not in groups:
                groups[dataset] = []
            groups[dataset].append(entity)
        
        print(f"📂 Found datasets: {list(groups.keys())}")
        for dataset, entities in groups.items():
            print(f"  {dataset}: {len(entities)} entities")
            
        return groups
    
    def _group_relations_by_dataset(self) -> Dict[str, Set[Tuple[int, int]]]:
        """Create dictionary of relations for each dataset"""
        relations_by_dataset = {}
        entity_to_dataset = {e['id']: e.get('dataset', 'default') for e in self.entities}
        
        for rel in self.relations:
            id1, id2 = rel
            dataset1 = entity_to_dataset.get(id1)
            dataset2 = entity_to_dataset.get(id2)
            
            # Relations only within one dataset
            if dataset1 == dataset2 and dataset1 is not None:
                if dataset1 not in relations_by_dataset:
                    relations_by_dataset[dataset1] = set()
                relations_by_dataset[dataset1].add((id1, id2))
        
        print(f"🔗 Distribution of relations by datasets:")
        for dataset, relations in relations_by_dataset.items():
            print(f"  {dataset}: {len(relations)} relations")
            
        return relations_by_dataset
    
    def _build_child_to_parents_index(self) -> Dict[str, Dict[int, Set[int]]]:
        """
        Create child -> {parent1, parent2, ...} index for fast relation search
        
        ARCHITECTURE:
        - vectors_1 = children (queries) - who is looking for parents
        - vectors_2 = parents (keys/values) - among whom we are looking for parents  
        - matrix[i, j] = 1 if child_i has parent_j
        
        Returns:
            Dict[dataset_name, Dict[child_id, Set[parent_ids]]]
        """
        child_to_parents = {}
        
        for dataset_name, relations_set in self.relations_by_dataset.items():
            child_to_parents[dataset_name] = defaultdict(set)
            
            for parent_id, child_id in relations_set:  # Relations: [parent_id, child_id]
                child_to_parents[dataset_name][child_id].add(parent_id)
        
        # Convert defaultdict to regular dict
        result = {}
        for dataset_name, child_dict in child_to_parents.items():
            result[dataset_name] = dict(child_dict)
        
        return result
    
    def _split_train_test(self):
        """Split data into train and test"""
        train_entities, test_entities = {}, {}
        train_relations, test_relations = [], []
        
        for dataset_name, dataset_entities in self.entities_by_dataset.items():
            entity_ids = [e['id'] for e in dataset_entities]
            
            if len(entity_ids) < 2:
                print(f"⚠️ Dataset {dataset_name} contains less than 2 entities, skipping split")
                if self.mode == "train":
                    train_entities[dataset_name] = dataset_entities
                continue
                
            # Split entities
            train_ids, test_ids = train_test_split(
                entity_ids, 
                test_size=self.test_part, 
                random_state=self.random_state
            )
            
            train_entities[dataset_name] = [e for e in dataset_entities if e['id'] in train_ids]
            test_entities[dataset_name] = [e for e in dataset_entities if e['id'] in test_ids]
        
        # Split relations
        all_train_ids = set([e['id'] for entities_list in train_entities.values() for e in entities_list])
        all_test_ids = set([e['id'] for entities_list in test_entities.values() for e in entities_list])
        
        for rel in self.relations:
            id1, id2 = rel
            if id1 in all_train_ids and id2 in all_train_ids:
                train_relations.append(rel)
            elif id1 in all_test_ids and id2 in all_test_ids:
                test_relations.append(rel)
            # Skip mixed train-test relations
        
        # Update data based on mode
        if self.mode == "train":
            self.entities_by_dataset = train_entities
            self.relations = train_relations
        elif self.mode == "test":
            self.entities_by_dataset = test_entities
            self.relations = test_relations
            
        # Reindex relations after split
        self.relations_by_dataset = self._group_relations_by_dataset()
        
        # Build child -> parents index for fast search
        self.child_to_parents = self._build_child_to_parents_index()
        
        print(f"✂️ Split completed:")
        print(f"  Train: {len(train_relations)} relations")
        print(f"  Test: {len(test_relations)} relations")
        print(f"  Excluded mixed: {len(self.relations) - len(train_relations) - len(test_relations)}")
    
    def _calculate_dataset_weights(self) -> Dict[str, float]:
        """Calculate dataset weights based on number of relations"""
        weights = {}
        total_relations = sum(len(rels) for rels in self.relations_by_dataset.values())
        
        if total_relations == 0:
            # If no relations, use uniform weights
            uniform_weight = 1.0 / len(self.available_datasets) if self.available_datasets else 0.0
            return {dataset: uniform_weight for dataset in self.available_datasets}
        
        for dataset_name in self.available_datasets:
            relations_count = len(self.relations_by_dataset.get(dataset_name, set()))
            weights[dataset_name] = relations_count / total_relations
        
        print(f"⚖️ Dataset weights:")
        for dataset, weight in weights.items():
            print(f"  {dataset}: {weight:.3f}")
            
        return weights
    
    def _select_dataset(self, idx: int) -> str:
        """Select dataset for current iteration"""
        if self.dataset_strategy == "single":
            # Only one dataset (or if only one dataset exists)
            return self.available_datasets[0]
        elif self.dataset_strategy == "weighted":
            # Weighted selection based on number of relations
            return self._weighted_dataset_selection()
        else:
            raise ValueError(f"Unknown dataset selection strategy: {self.dataset_strategy}. Available: 'single', 'weighted'")
    
    def _weighted_dataset_selection(self) -> str:
        """Weighted selection of dataset based on number of relations"""
        datasets = list(self.dataset_weights.keys())
        weights = list(self.dataset_weights.values())
        return random.choices(datasets, weights=weights)[0]
    
    def _sample_entities_random(self, dataset_name: str, batch_size: int) -> Tuple[torch.Tensor, List[int]]:
        """Random sampling of entities from a specific dataset"""
        entities = self.entities_by_dataset[dataset_name]
        
        if len(entities) == 0:
            raise ValueError(f"Dataset {dataset_name} is empty")
            
        sampled = random.sample(entities, min(batch_size, len(entities)))
        
        # Extract embeddings and IDs
        vectors = torch.tensor([e['embedding'] for e in sampled], dtype=torch.float32)
        ids = [e['id'] for e in sampled]
        
        return vectors, ids
    

    
    def _sample_entities_balanced(self, dataset_name: str, batch_size_1: int, batch_size_2: int) -> Tuple[torch.Tensor, List[int], torch.Tensor, List[int]]:
        """
        SMART strategy for maximizing positive pairs:
        1) Randomly select children (ids_1)
        2) Weighted sample parents (ids_2) based on the number of relations with selected children
        """
        entities = self.entities_by_dataset[dataset_name]
        child_to_parents_dict = self.child_to_parents.get(dataset_name, {})
        
        if len(entities) == 0:
            raise ValueError(f"Dataset {dataset_name} is empty")
        
        # If no relations in the dataset, use random sampling
        if len(child_to_parents_dict) == 0:
            vectors_1, ids_1 = self._sample_entities_random(dataset_name, batch_size_1)
            vectors_2, ids_2 = self._sample_entities_random(dataset_name, batch_size_2)
            return vectors_1, ids_1, vectors_2, ids_2
        
        entity_by_id = {e['id']: e for e in entities}
        
        # 1) Randomly select children (ids_1)
        sampled_children = random.sample(entities, min(batch_size_1, len(entities)))
        ids_1 = [e['id'] for e in sampled_children]
        
        # 2) Create a probability vector for all entities
        all_entity_ids = [e['id'] for e in entities]
        probs = torch.zeros(len(all_entity_ids), dtype=torch.float32)
        entity_id_to_index = {entity_id: i for i, entity_id in enumerate(all_entity_ids)}
        
        # 3) Increment probs for parents of selected children
        for child_id in ids_1:
            child_parents = child_to_parents_dict.get(child_id, set())
            for parent_id in child_parents:
                if parent_id in entity_id_to_index:
                    idx = entity_id_to_index[parent_id]
                    probs[idx] += 1.0
        
        # 4) Sample ids_2 according to probs
        if probs.sum() > 0:
            # If there are related parents - sample weighted
            probs = probs / probs.sum()  # Normalize
            
            # Sample without repetition
            selected_indices = torch.multinomial(probs, min(batch_size_2, len(entities)), replacement=False)
            ids_2 = [all_entity_ids[idx] for idx in selected_indices.tolist()]
            sampled_parents = [entity_by_id[entity_id] for entity_id in ids_2]
        else:
            # If no related parents - random sampling
            sampled_parents = random.sample(entities, min(batch_size_2, len(entities)))
            ids_2 = [e['id'] for e in sampled_parents]
        
        # Pad to the desired size if not enough
        while len(sampled_children) < batch_size_1:
            available = [e for e in entities if e['id'] not in ids_1]
            if not available:
                break
            selected = random.choice(available)
            sampled_children.append(selected)
            ids_1.append(selected['id'])
        
        while len(sampled_parents) < batch_size_2:
            available = [e for e in entities if e['id'] not in ids_2]
            if not available:
                break
            selected = random.choice(available)
            sampled_parents.append(selected)
            ids_2.append(selected['id'])
        
        # Truncate to desired size
        sampled_children = sampled_children[:batch_size_1]
        sampled_parents = sampled_parents[:batch_size_2]
        ids_1 = ids_1[:batch_size_1]
        ids_2 = ids_2[:batch_size_2]
        
        # Extract embeddings
        vectors_1 = torch.tensor([e['embedding'] for e in sampled_children], dtype=torch.float32)
        vectors_2 = torch.tensor([e['embedding'] for e in sampled_parents], dtype=torch.float32)
        
        return vectors_1, ids_1, vectors_2, ids_2
    
    def _sample_entities_from_dataset(self, dataset_name: str) -> Tuple[torch.Tensor, List[int], torch.Tensor, List[int]]:
        """Main sampling method with strategy selection"""
        if self.sampling_strategy == "random":
            vectors_1, ids_1 = self._sample_entities_random(dataset_name, self.batch_size_1)
            vectors_2, ids_2 = self._sample_entities_random(dataset_name, self.batch_size_2)
            return vectors_1, ids_1, vectors_2, ids_2
        elif self.sampling_strategy == "balanced":
            return self._sample_entities_balanced(dataset_name, self.batch_size_1, self.batch_size_2)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}. Available: 'random', 'balanced'")
    
    def _build_relation_matrix(self, ids_1: List[int], ids_2: List[int], dataset_name: str) -> torch.Tensor:
        """
        OPTIMIZED creation of n×m relation matrix for a specific dataset
        
        ARCHITECTURE:
        - ids_1 = children (rows) - who is looking for parents
        - ids_2 = parents (cols) - among whom we are looking for parents
        - matrix[i, j] = 1 if child_i has parent_j
        
        OPTIMIZATION: O(n*k) instead of O(n*m), where k is the average number of parents per child
        """
        matrix = torch.zeros(len(ids_1), len(ids_2), dtype=torch.float32)
        child_to_parents_dict = self.child_to_parents.get(dataset_name, {})
        
        # Create reverse index for fast search of parent_id positions in ids_2
        parent_id_to_col = {parent_id: j for j, parent_id in enumerate(ids_2)}
        
        # For each child, find its parents and set 1 in the corresponding positions
        for i, child_id in enumerate(ids_1):
            child_parents = child_to_parents_dict.get(child_id, set())
            
            for parent_id in child_parents:
                # If parent is present in the current batch of ids_2
                if parent_id in parent_id_to_col:
                    j = parent_id_to_col[parent_id]
                    matrix[i, j] = 1.0
        
        return matrix
    
    def __len__(self) -> int:
        """Dataset size (conditional)"""
        # Return conditional size for DataLoader
        total_entities = sum(len(entities) for entities in self.entities_by_dataset.values())
        # Approximate number of possible batches
        return max(1, total_entities // max(self.batch_size_1, self.batch_size_2))
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        Getting one dataset element
        
        Returns:
            vectors_1: (batch_size_1, hidden_size) - first vector set
            vectors_2: (batch_size_2, hidden_size) - second vector set  
            gt_matrix: (batch_size_1, batch_size_2) - relation matrix
            dataset_name: dataset name
        """
        # Select dataset for current iteration
        dataset_name = self._select_dataset(idx)
        
        # Sampling with selected strategy
        vectors_1, ids_1, vectors_2, ids_2 = self._sample_entities_from_dataset(dataset_name)
        
        # Build relation matrix for this dataset
        gt_matrix = self._build_relation_matrix(ids_1, ids_2, dataset_name)
        
        return vectors_1, vectors_2, gt_matrix, dataset_name
    
    def _print_statistics(self):
        """Print dataset statistics"""
        print(f"\n�� Dataset Statistics:")
        print(f"  Mode: {self.mode}")
        print(f"  Dataset selection strategy: {self.dataset_strategy}")
        print(f"  Sampling strategy: {self.sampling_strategy}")
        print(f"  Batch sizes: {self.batch_size_1}×{self.batch_size_2}")
        
        total_entities = sum(len(entities) for entities in self.entities_by_dataset.values())
        total_relations = len(self.relations)
        
        print(f"  Total entities: {total_entities}")
        print(f"  Total relations: {total_relations}")
        print(f"  Datasets: {len(self.available_datasets)}")
        
        # Validate embedding dimension
        if total_entities > 0:
            first_entity = next(iter(self.entities_by_dataset.values()))[0]
            embedding_dim = len(first_entity['embedding'])
            print(f"  Embedding dimension: {embedding_dim}")
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get detailed dataset information"""
        info = {
            'mode': self.mode,
            'dataset_strategy': self.dataset_strategy,
            'sampling_strategy': self.sampling_strategy,
            'batch_sizes': (self.batch_size_1, self.batch_size_2),
            'available_datasets': self.available_datasets,
            'total_entities': sum(len(entities) for entities in self.entities_by_dataset.values()),
            'total_relations': len(self.relations),
            'entities_by_dataset': {k: len(v) for k, v in self.entities_by_dataset.items()},
            'relations_by_dataset': {k: len(v) for k, v in self.relations_by_dataset.items()}
        }
        
        if hasattr(self, 'dataset_weights'):
            info['dataset_weights'] = self.dataset_weights
            
        return info


def create_train_test_datasets(
    entities_path: str,
    relations_path: str,
    batch_size_1: int = 32,
    batch_size_2: int = 32,
    dataset_strategy: str = "single",
    sampling_strategy: str = "balanced",
    positive_ratio: float = 1.0,
    test_part: float = 0.2,
    random_state: int = 42
) -> Tuple[CrossAttentionDataset, CrossAttentionDataset]:
    """
    Creating train and test datasets
    
    Returns:
        train_dataset, test_dataset
    """
    train_dataset = CrossAttentionDataset(
        entities_path=entities_path,
        relations_path=relations_path,
        batch_size_1=batch_size_1,
        batch_size_2=batch_size_2,
        dataset_strategy=dataset_strategy,
        sampling_strategy=sampling_strategy,
        positive_ratio=positive_ratio,
        mode="train",
        test_part=test_part,
        random_state=random_state
    )
    
    test_dataset = CrossAttentionDataset(
        entities_path=entities_path,
        relations_path=relations_path,
        batch_size_1=batch_size_1,
        batch_size_2=batch_size_2,
        dataset_strategy=dataset_strategy,
        sampling_strategy=sampling_strategy,
        positive_ratio=positive_ratio,
        mode="test",
        test_part=test_part,
        random_state=random_state
    )
    
    return train_dataset, test_dataset


if __name__ == "__main__":
    # Test example with dummy data
    print("🚀 Testing CrossAttentionDataset...")
    
    # Create test data
    import tempfile
    import os
    
    # Create temporary files
    temp_dir = tempfile.mkdtemp()
    entities_file = os.path.join(temp_dir, 'entities.json')
    relations_file = os.path.join(temp_dir, 'relations.json')
    
    # Test entities
    test_entities = [
        {"id": 0, "text_description": "entity 0", "embedding": [0.1] * 512, "dataset": "ecology"},
        {"id": 1, "text_description": "entity 1", "embedding": [0.2] * 512, "dataset": "ecology"},
        {"id": 2, "text_description": "entity 2", "embedding": [0.3] * 512, "dataset": "engineering"},
        {"id": 3, "text_description": "entity 3", "embedding": [0.4] * 512, "dataset": "engineering"},
        {"id": 4, "text_description": "entity 4", "embedding": [0.5] * 512},  # without 'dataset' field
    ]
    
    # Test relations
    test_relations = [
        [0, 1],  # ecology
        [2, 3],  # engineering  
        [0, 2],  # between datasets - will be excluded
    ]
    
    # Save test data
    with open(entities_file, 'w') as f:
        json.dump(test_entities, f)
    with open(relations_file, 'w') as f:
        json.dump(test_relations, f)
    
    try:
        # Test dataset creation
        dataset = CrossAttentionDataset(
            entities_path=entities_file,
            relations_path=relations_file,
            batch_size_1=2,
            batch_size_2=3,
            dataset_strategy="single",
            sampling_strategy="balanced",
            positive_ratio=0.3
        )
        
        print(f"\nDataset information:")
        info = dataset.get_dataset_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Test getting a batch
        print(f"\n🎯 Testing batch:")
        vectors_1, vectors_2, gt_matrix, dataset_name = dataset[0]
        
        print(f"  Selected dataset: {dataset_name}")
        print(f"  vectors_1: {vectors_1.shape}")
        print(f"  vectors_2: {vectors_2.shape}")
        print(f"  gt_matrix: {gt_matrix.shape}")
        print(f"  Relation matrix:")
        print(f"    {gt_matrix}")
        
        # Test train/test split
        print(f"\n✂️ Testing train/test split:")
        train_dataset, test_dataset = create_train_test_datasets(
            entities_file, relations_file,
            batch_size_1=2, batch_size_2=2,
            sampling_strategy="balanced"
        )
        
        print("✅ All tests passed successfully!")
        
    finally:
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir) 