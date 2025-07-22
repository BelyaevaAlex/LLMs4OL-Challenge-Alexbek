"""
Utilities for Cross-Attention model
Includes functions for data processing, metrics, validation and additional operations.
"""

import torch
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from datetime import datetime

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    sns = None

VISUALIZATION_AVAILABLE = MATPLOTLIB_AVAILABLE

if not MATPLOTLIB_AVAILABLE:
    print("⚠️ Matplotlib is not available. Visualization disabled.")
elif not SEABORN_AVAILABLE:
    print("⚠️ Seaborn is not available. Using only matplotlib for visualization.")


def validate_entities_file(entities_path: str) -> Dict[str, Any]:
    """
    Validation of entities.json file
    
    Args:
        entities_path: path to entities.json file
        
    Returns:
        validation_report: validation report
    """
    print(f"🔍 Validating file: {entities_path}")
    
    if not os.path.exists(entities_path):
        return {"valid": False, "error": f"File not found: {entities_path}"}
    
    try:
        with open(entities_path, 'r', encoding='utf-8') as f:
            entities = json.load(f)
    except Exception as e:
        return {"valid": False, "error": f"JSON loading error: {e}"}
    
    if not isinstance(entities, list):
        return {"valid": False, "error": "File must contain a list"}
    
    # Statistics
    total_entities = len(entities)
    valid_entities = 0
    issues = []
    
    # Check each entity
    embedding_dims = set()
    datasets = set()
    
    for i, entity in enumerate(entities):
        if not isinstance(entity, dict):
            issues.append(f"Entity {i} is not a dictionary")
            continue
            
        # Check required fields
        required_fields = ['id', 'embedding']
        missing_fields = [field for field in required_fields if field not in entity]
        
        if missing_fields:
            issues.append(f"Entity {i}: missing fields {missing_fields}")
            continue
        
        # Check types
        if not isinstance(entity['id'], int):
            issues.append(f"Entity {i}: id must be int")
            continue
            
        if not isinstance(entity['embedding'], list):
            issues.append(f"Entity {i}: embedding must be a list")
            continue
        
        # Check embedding dimensions
        embedding_dims.add(len(entity['embedding']))
        
        # Check dataset
        if 'dataset' in entity:
            datasets.add(entity['dataset'])
        else:
            datasets.add('default')
        
        # Check text_description
        if 'text_description' not in entity:
            issues.append(f"Entity {i}: recommended to add text_description")
        
        valid_entities += 1
    
    # Validation result
    is_valid = len(issues) == 0 and valid_entities == total_entities
    
    report = {
        "valid": is_valid,
        "total_entities": total_entities,
        "valid_entities": valid_entities,
        "issues": issues,
        "embedding_dimensions": list(embedding_dims),
        "datasets": list(datasets),
        "consistent_embedding_dim": len(embedding_dims) == 1,
        "embedding_dim": list(embedding_dims)[0] if len(embedding_dims) == 1 else None
    }
    
    print(f"✅ Validation completed: {valid_entities}/{total_entities} valid entities")
    if issues:
        print(f"⚠️ Issues found: {len(issues)}")
    
    return report


def validate_relations_file(relations_path: str, entities_ids: set = None) -> Dict[str, Any]:
    """
    Validation of relations.json file
    
    Args:
        relations_path: path to relations.json file
        entities_ids: set of valid entity IDs
        
    Returns:
        validation_report: validation report
    """
    print(f"🔍 Validating file: {relations_path}")
    
    if not os.path.exists(relations_path):
        return {"valid": False, "error": f"File not found: {relations_path}"}
    
    try:
        with open(relations_path, 'r', encoding='utf-8') as f:
            relations = json.load(f)
    except Exception as e:
        return {"valid": False, "error": f"JSON loading error: {e}"}
    
    if not isinstance(relations, list):
        return {"valid": False, "error": "File must contain a list"}
    
    # Statistics
    total_relations = len(relations)
    valid_relations = 0
    issues = []
    
    for i, relation in enumerate(relations):
        if not isinstance(relation, list):
            issues.append(f"Relation {i} is not a list")
            continue
            
        if len(relation) != 2:
            issues.append(f"Relation {i} must contain 2 elements")
            continue
        
        id1, id2 = relation
        
        if not isinstance(id1, int) or not isinstance(id2, int):
            issues.append(f"Relation {i}: IDs must be int")
            continue
        
        # Check entity existence
        if entities_ids is not None:
            if id1 not in entities_ids:
                issues.append(f"Relation {i}: entity {id1} not found")
                continue
            if id2 not in entities_ids:
                issues.append(f"Relation {i}: entity {id2} not found")
                continue
        
        valid_relations += 1
    
    # Validation result
    is_valid = len(issues) == 0 and valid_relations == total_relations
    
    report = {
        "valid": is_valid,
        "total_relations": total_relations,
        "valid_relations": valid_relations,
        "issues": issues,
        "unique_relations": len(set(tuple(rel) for rel in relations if isinstance(rel, list) and len(rel) == 2))
    }
    
    print(f"✅ Validation completed: {valid_relations}/{total_relations} valid relations")
    if issues:
        print(f"⚠️ Issues found: {len(issues)}")
    
    return report


def validate_dataset_files(entities_path: str, relations_path: str) -> Dict[str, Any]:
    """
    Comprehensive validation of dataset files
    
    Args:
        entities_path: path to entities.json
        relations_path: path to relations.json
        
    Returns:
        validation_report: complete validation report
    """
    print(f"🔍 Comprehensive dataset validation...")
    
    # Validate entities
    entities_report = validate_entities_file(entities_path)
    
    if not entities_report["valid"]:
        return {
            "valid": False,
            "entities_report": entities_report,
            "relations_report": None,
            "error": "entities.json validation error"
        }
    
    # Get set of entity IDs
    with open(entities_path, 'r', encoding='utf-8') as f:
        entities = json.load(f)
    entities_ids = set(entity['id'] for entity in entities)
    
    # Validate relations
    relations_report = validate_relations_file(relations_path, entities_ids)
    
    # Overall result
    overall_valid = entities_report["valid"] and relations_report["valid"]
    
    report = {
        "valid": overall_valid,
        "entities_report": entities_report,
        "relations_report": relations_report,
        "summary": {
            "total_entities": entities_report["total_entities"],
            "total_relations": relations_report["total_relations"],
            "embedding_dim": entities_report.get("embedding_dim"),
            "datasets": entities_report["datasets"],
            "consistent_embeddings": entities_report["consistent_embedding_dim"]
        }
    }
    
    print(f"📊 Overall validation result: {'✅ Success' if overall_valid else '❌ Errors'}")
    
    return report


def calculate_detailed_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Calculate detailed classification metrics
    
    Args:
        predictions: array of predictions (0-1)
        targets: array of target values (0-1)
        
    Returns:
        metrics: dictionary with metrics
    """
    metrics = {}
    
    # ROC AUC
    if len(np.unique(targets)) > 1:
        metrics['roc_auc'] = roc_auc_score(targets, predictions)
    else:
        metrics['roc_auc'] = 0.0
    
    # Metrics for different thresholds
    thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]
    
    for threshold in thresholds:
        pred_binary = (predictions > threshold).astype(int)
        
        metrics[f'acc_{threshold}'] = accuracy_score(targets, pred_binary)
        metrics[f'f1_{threshold}'] = f1_score(targets, pred_binary, zero_division=0)
        metrics[f'precision_{threshold}'] = precision_score(targets, pred_binary, zero_division=0)
        metrics[f'recall_{threshold}'] = recall_score(targets, pred_binary, zero_division=0)
    
    # Additional metrics
    metrics['mean_prediction'] = np.mean(predictions)
    metrics['std_prediction'] = np.std(predictions)
    metrics['positive_ratio'] = np.mean(targets)
    
    return metrics


def plot_training_history(training_log: List[Dict[str, Any]], save_path: str = None):
    """
    Plot training history graphs
    
    Args:
        training_log: training log
        save_path: path to save the plot
    """
    if not VISUALIZATION_AVAILABLE:
        print("❌ Visualization not available. Install matplotlib.")
        return
        
    if not training_log:
        print("❌ Training log is empty")
        return
    
    # Extract data
    steps = [entry['step'] for entry in training_log]
    train_losses = [entry['train_loss'] for entry in training_log]
    test_losses = [entry['test_loss'] for entry in training_log]
    learning_rates = [entry['lr'] for entry in training_log]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(steps, train_losses, label='Train Loss', color='blue')
    axes[0, 0].plot(steps, test_losses, label='Test Loss', color='red')
    axes[0, 0].set_xlabel('Steps')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Test Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Learning rate plot
    axes[0, 1].plot(steps, learning_rates, color='green')
    axes[0, 1].set_xlabel('Steps')
    axes[0, 1].set_ylabel('Learning Rate')
    axes[0, 1].set_title('Learning Rate Schedule')
    axes[0, 1].grid(True)
    
    # Loss histogram
    axes[1, 0].hist(train_losses, bins=30, alpha=0.7, label='Train Loss')
    axes[1, 0].hist(test_losses, bins=30, alpha=0.7, label='Test Loss')
    axes[1, 0].set_xlabel('Loss')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Loss Distribution')
    axes[1, 0].legend()
    
    # Difference between train and test loss
    loss_diff = [train - test for train, test in zip(train_losses, test_losses)]
    axes[1, 1].plot(steps, loss_diff, color='purple')
    axes[1, 1].set_xlabel('Steps')
    axes[1, 1].set_ylabel('Train Loss - Test Loss')
    axes[1, 1].set_title('Overfitting Indicator')
    axes[1, 1].grid(True)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved: {save_path}")
    
    plt.show()


def plot_attention_matrix(attention_matrix: np.ndarray, save_path: str = None, 
                         entity_names: List[str] = None):
    """
    Visualize attention matrix
    
    Args:
        attention_matrix: attention matrix (n×m)
        save_path: path to save
        entity_names: entity names for labels
    """
    if not VISUALIZATION_AVAILABLE:
        print("❌ Visualization not available. Install matplotlib.")
        return
        
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    if SEABORN_AVAILABLE:
        # Use seaborn if available
        sns.heatmap(
            attention_matrix,
            annot=True,
            fmt='.3f',
            cmap='Blues',
            cbar_kws={'label': 'Attention Weight'}
        )
    else:
        # Use only matplotlib
        im = plt.imshow(attention_matrix, cmap='Blues', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Attention Weight')
        
        # Add annotations
        for i in range(attention_matrix.shape[0]):
            for j in range(attention_matrix.shape[1]):
                plt.text(j, i, f'{attention_matrix[i, j]:.3f}',
                        ha="center", va="center", color="black" if attention_matrix[i, j] < 0.5 else "white")
    
    plt.title('Cross-Attention Matrix')
    plt.xlabel('Key Entities')
    plt.ylabel('Query Entities')
    
    # Add labels if available
    if entity_names:
        n_entities = len(entity_names)
        if attention_matrix.shape[0] <= n_entities:
            plt.yticks(range(attention_matrix.shape[0]), 
                      entity_names[:attention_matrix.shape[0]], rotation=0)
        if attention_matrix.shape[1] <= n_entities:
            plt.xticks(range(attention_matrix.shape[1]), 
                      entity_names[:attention_matrix.shape[1]], rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Attention matrix saved: {save_path}")
    
    plt.show()


def create_summary_report(
    entities_path: str,
    relations_path: str,
    model_info: Dict[str, Any],
    training_log: List[Dict[str, Any]],
    final_metrics: Dict[str, float],
    save_path: str
):
    """
    Create final report about model and training
    
    Args:
        entities_path: path to entities.json
        relations_path: path to relations.json
        model_info: model information
        training_log: training log
        final_metrics: final metrics
        save_path: path to save the report
    """
    
    # Validate data
    validation_report = validate_dataset_files(entities_path, relations_path)
    
    # Training statistics
    if training_log:
        training_stats = {
            'total_steps': len(training_log),
            'final_train_loss': training_log[-1]['train_loss'],
            'final_test_loss': training_log[-1]['test_loss'],
            'min_train_loss': min(entry['train_loss'] for entry in training_log),
            'min_test_loss': min(entry['test_loss'] for entry in training_log),
            'training_time': training_log[-1]['timestamp']
        }
    else:
        training_stats = {}
    
    # Create report
    report = {
        'model_info': model_info,
        'dataset_validation': validation_report,
        'training_statistics': training_stats,
        'final_metrics': final_metrics,
        'generation_time': datetime.now().isoformat(),
        'files': {
            'entities': entities_path,
            'relations': relations_path
        }
    }
    
    # Save report
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📋 Final report saved: {save_path}")
    
    # Print brief summary
    print("\n📊 FINAL SUMMARY")
    print("=" * 50)
    print(f"Model: {model_info.get('model_type', 'N/A')}")
    print(f"Parameters: {model_info.get('num_trainable_params', 'N/A'):,}")
    print(f"Dimension: {model_info.get('hidden_size', 'N/A')}")
    print(f"Entities: {validation_report['summary']['total_entities']}")
    print(f"Relations: {validation_report['summary']['total_relations']}")
    print(f"Datasets: {len(validation_report['summary']['datasets'])}")
    print(f"Final ROC AUC: {final_metrics.get('roc_auc', 'N/A'):.4f}")
    print(f"Final accuracy (0.5): {final_metrics.get('acc_0.5', 'N/A'):.4f}")
    print("=" * 50)


def compare_models(model_paths: List[str], test_dataset_path: str = None):
    """
    Compare multiple models
    
    Args:
        model_paths: paths to models for comparison
        test_dataset_path: path to test dataset
    """
    print(f"🔍 Comparing {len(model_paths)} models...")
    
    comparison_results = []
    
    for model_path in model_paths:
        try:
            # Load model information
            checkpoint = torch.load(model_path, map_location='cpu')
            
            model_info = {
                'path': model_path,
                'filename': os.path.basename(model_path),
                'step': checkpoint.get('step', 'N/A'),
                'roc_auc': checkpoint.get('roc_auc', 'N/A'),
                'metrics': checkpoint.get('metrics', {}),
                'timestamp': checkpoint.get('timestamp', 'N/A')
            }
            
            comparison_results.append(model_info)
            
        except Exception as e:
            print(f"❌ Error loading {model_path}: {e}")
    
    # Sort by ROC AUC
    comparison_results.sort(key=lambda x: x['roc_auc'], reverse=True)
    
    # Print results
    print("\n📊 MODEL COMPARISON")
    print("=" * 80)
    print(f"{'Model':<30} {'Step':<10} {'ROC AUC':<10} {'Acc(0.5)':<10} {'F1(0.5)':<10}")
    print("-" * 80)
    
    for result in comparison_results:
        metrics = result['metrics']
        acc_05 = metrics.get('acc_0.5', 'N/A')
        f1_05 = metrics.get('f1_0.5', 'N/A')
        
        print(f"{result['filename']:<30} {result['step']:<10} {result['roc_auc']:<10.4f} "
              f"{acc_05:<10.4f} {f1_05:<10.4f}")
    
    print("=" * 80)
    
    return comparison_results


if __name__ == "__main__":
    # Test utilities
    print("🧪 Testing utilities...")
    
    # Create test data
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    # Test entities
    test_entities = [
        {"id": 0, "text_description": "entity 0", "embedding": [0.1] * 512, "dataset": "test"},
        {"id": 1, "text_description": "entity 1", "embedding": [0.2] * 512, "dataset": "test"},
    ]
    
    # Test relations
    test_relations = [[0, 1]]
    
    entities_file = os.path.join(temp_dir, 'entities.json')
    relations_file = os.path.join(temp_dir, 'relations.json')
    
    with open(entities_file, 'w') as f:
        json.dump(test_entities, f)
    with open(relations_file, 'w') as f:
        json.dump(test_relations, f)
    
    # Test validation
    print("\n🔍 Testing validation...")
    report = validate_dataset_files(entities_file, relations_file)
    print(f"Validation result: {report['valid']}")
    
    # Test metrics
    print("\n📊 Testing metrics...")
    predictions = np.array([0.8, 0.2, 0.9, 0.1])
    targets = np.array([1, 0, 1, 0])
    
    metrics = calculate_detailed_metrics(predictions, targets)
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Accuracy (0.5): {metrics['acc_0.5']:.4f}")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    
    print("\n✅ Utilities testing completed!") 