
"""
Term Classification using Embeddings instead of TfidfVectorizer
Adapted version for using semantic embeddings instead of TF-IDF
"""

import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import gc
from typing import List, Dict
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️  XGBoost not available, alternative classifiers will be used")
    XGBOOST_AVAILABLE = False

# Try to import additional libraries
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from sklearn.multioutput import MultiOutputRegressor
import joblib
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from model_optimization import (
    get_base_classifiers, 
    save_comparison_metrics,
    create_term_graph,
    extract_graph_features
)

# GPU settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def last_token_pool(last_hidden_states, attention_mask):
    """Last token pooling"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_embeddings_batch(texts: List[str], model, tokenizer, max_length=8192, batch_size=8):
    """Get embeddings for texts in batches"""
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing embeddings"):
        batch_texts = texts[i:i+batch_size]
        
        # Batch tokenization
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
        
        # Memory cleanup
        del batch_tokenized, outputs, batch_embeddings
        torch.cuda.empty_cache()
        gc.collect()
    
    return torch.cat(all_embeddings, dim=0)


def get_detailed_instruct(instruction, query):
    """Create detailed instruct for embeddings"""
    return f"Instruct: {instruction}\nQuery: {query}"


class EmbeddingTermClassifier:
    """Term classifier based on embeddings"""
    
    def __init__(self, model_name="Qwen/Qwen3-Embedding-4B", max_length=8192, batch_size=32):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        print(f"Loading embedding model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("Using device: cuda")
        else:
            print("Using device: cpu")
        self.mlb = MultiLabelBinarizer()
        
    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Getting embeddings"):
            batch_texts = texts[i:i + self.batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                  max_length=self.max_length, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
                embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 normalization
                all_embeddings.append(embeddings.cpu())
            
            # CUDA memory cleanup
            if torch.cuda.is_available():
                del inputs, outputs, embeddings
                torch.cuda.empty_cache()
                gc.collect()
        
        return torch.cat(all_embeddings, dim=0)

    def prepare_data(self, train_data: List[Dict], test_data: List[Dict] = None):
        # Save data for graph creation
        self.train_data = train_data
        self.train_terms = [item['term'] for item in train_data]
        train_types = [item['types'] if isinstance(item['types'], list) else [item['types']] 
                      for item in train_data]
        
        # Get embeddings
        print("Getting embeddings for training data...")
        self.X_train_embeddings = self.get_embeddings(self.train_terms)
        
        # Prepare labels
        self.mlb.fit(train_types)
        self.y_train = self.mlb.transform(train_types)
        
        # Prepare test data if provided
        if test_data:
            test_terms = [item['term'] for item in test_data]
            print("Getting embeddings for test data...")
            self.X_test_embeddings = self.get_embeddings(test_terms)
            return self.X_train_embeddings, self.y_train, self.X_test_embeddings, test_terms
        
        return self.X_train_embeddings, self.y_train

    def train_and_evaluate(self, domain: str, save_path: str):
        """Train classifiers and evaluate their quality"""
        classifiers = get_base_classifiers()
        metrics_dict = {}
        
        X_train = self.X_train_embeddings.numpy()
        
        # Create graph and get graph features
        G = create_term_graph(self.train_data)
        graph_features = extract_graph_features(G, self.train_terms)
        X_train_with_graph = np.hstack([X_train, graph_features])
        
        # Train and evaluate classifier without graph features
        clf = classifiers['random_forest']
        print("\nTraining Random Forest without graph features...")
        clf.fit(X_train, self.y_train)
        
        train_pred = clf.predict(X_train)
        metrics = self._calculate_metrics(self.y_train, train_pred)
        metrics_dict['random_forest'] = metrics
        
        print(f"Metrics for Random Forest without graph features:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.3f}")
            
        # Train and evaluate classifier with graph features
        clf_with_graph = classifiers['random_forest_with_graph']
        print("\nTraining Random Forest with graph features...")
        clf_with_graph.fit(X_train_with_graph, self.y_train)
        
        train_pred = clf_with_graph.predict(X_train_with_graph)
        metrics = self._calculate_metrics(self.y_train, train_pred)
        metrics_dict['random_forest_with_graph'] = metrics
        
        print(f"Metrics for Random Forest with graph features:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.3f}")
        
        # Save metrics
        save_comparison_metrics(metrics_dict, domain, save_path)
        
        return clf, clf_with_graph, G

    def predict(self, classifier, classifier_with_graph, G, X_test_embeddings: torch.Tensor, 
                test_terms: List[str], test_data: List[Dict], save_path: str):
        """Prediction for test data"""
        X_test = X_test_embeddings.numpy()
        
        # Get graph features for test data
        graph_features = extract_graph_features(G, test_terms)
        X_test_with_graph = np.hstack([X_test, graph_features])
        
        print("\nПредсказание с помощью Random Forest без графовых признаков...")
        y_pred_proba = classifier.predict_proba(X_test)
        predictions = self._format_predictions(test_terms, test_data, y_pred_proba)
        
        # Сохранение предсказаний
        output_file = os.path.join(save_path, "predictions_random_forest.json")
        with open(output_file, 'w') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
            
        print("\nПредсказание с помощью Random Forest с графовыми признаками...")
        y_pred_proba = classifier_with_graph.predict_proba(X_test_with_graph)
        predictions_with_graph = self._format_predictions(test_terms, test_data, y_pred_proba)
        
        # Сохранение предсказаний
        output_file = os.path.join(save_path, "predictions_random_forest_with_graph.json")
        with open(output_file, 'w') as f:
            json.dump(predictions_with_graph, f, indent=2, ensure_ascii=False)
        
        return predictions, predictions_with_graph

    def _calculate_metrics(self, y_true, y_pred):
        """Calculate metrics for multi-class classification"""
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        accuracy = accuracy_score(y_true, y_pred)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        }

    def _format_predictions(self, test_terms: List[str], test_data: List[Dict], 
                          y_pred_proba: np.ndarray, threshold: float = 0.3) -> List[Dict]:
        """Format predictions with confidence threshold"""
        predictions = []
        classes = self.mlb.classes_
        
        for i, (term, proba) in enumerate(zip(test_terms, y_pred_proba)):
            # Get indices of classes where probability is above threshold
            selected_indices = np.where(proba > threshold)[0]
            
            # If no classes above threshold, take class with maximum probability
            if len(selected_indices) == 0:
                selected_indices = [np.argmax(proba)]
            
            # Get predicted types
            predicted_types = [classes[idx] for idx in selected_indices]
            
            # Format prediction in required format
            prediction = {
                "id": test_data[i]["id"],
                "types": predicted_types
            }
            predictions.append(prediction)
        
        return predictions


def load_taskb_data():
    """Load TaskB data for all domains"""
    domains = ['MatOnto', 'OBI', 'SWEET']
    train_data = {}
    test_data = {}
    
    print("Loading train data...")
    for domain in domains:
        train_path = f'../../../2025/TaskB-TermTyping/{domain}/train/term_typing_train_data.json'
        if os.path.exists(train_path):
            with open(train_path, 'r', encoding='utf-8') as f:
                train_data[domain] = json.load(f)
            print(f"{domain} train: {len(train_data[domain])} examples")
        else:
            print(f"❌ Train file not found: {train_path}")

    print("\nLoading test data...")
    for domain in domains:
        domain_lower = domain.lower()
        test_path = f'../../../2025/TaskB-TermTyping/{domain}/test/{domain_lower}_term_typing_test_data.json'
        
        if os.path.exists(test_path):
            with open(test_path, 'r', encoding='utf-8') as f:
                test_json_data = json.load(f)
            # Convert to required format (add empty ID field for compatibility)
            test_data[domain] = [{"ID": i, "term": item["term"]} for i, item in enumerate(test_json_data)]
            print(f"{domain} test: {len(test_data[domain])} terms")
        else:
            print(f"❌ Test file not found: {test_path}")
    
    return train_data, test_data, domains


def process_single_domain(domain: str, train_data_domain: list, test_data_domain: list, 
                         classifier: EmbeddingTermClassifier):
    """Process single domain"""
    print(f"\n=== Processing domain: {domain} ===")
    
    # Prepare data
    train_texts, test_texts, y, test_terms, df_train = classifier.prepare_data(
        train_data_domain, test_data_domain
    )
    
    # Save terms for graph features
    classifier.train_terms = df_train['term'].tolist()
    
    # Get embeddings
    print("Getting train embeddings...")
    X_train_embeddings = classifier.get_embeddings(train_texts)
    
    print("Getting test embeddings...")
    X_test_embeddings = classifier.get_embeddings(test_texts)
    
    # Train classifier
    clf, clf_with_graph, G = classifier.train_and_evaluate(domain, f'.')
    
    # Evaluate on training data
    print("\n📊 Metrics on training data:")
    metrics_dict = {}
    
    X_train_combined = X_train_embeddings.numpy()
    
    metrics = classifier._calculate_metrics(y, clf.predict(X_train_combined))
    metrics_dict['random_forest'] = metrics
    
    print(f"\nRandom Forest:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.3f}")
    
    # Save comparison metrics
    save_dir = f'.'
    save_comparison_metrics(metrics_dict, domain, save_dir)
    
    # Prediction
    predictions, predictions_with_graph = classifier.predict(clf, clf_with_graph, G, X_test_embeddings, test_terms, test_data_domain, save_dir)
    
    # Save results
    save_dir = f'./saved_models_embedding_{domain}'
    predictions_dir = f'./predictions_embedding_{domain}'
    
    # classifier.save_models(save_dir) # This line was removed as per the edit hint
    # classifier.save_predictions(predictions, predictions_dir, domain) # This line was removed as per the edit hint
    
    print(f"✅ Domain {domain} processed successfully!")
    return predictions, predictions_with_graph


def main():
    """Main function for processing all TaskB domains."""
    
    # Load data for all domains
    train_data, test_data, domains = load_taskb_data()
    
    if not train_data or not test_data:
        print("❌ Failed to load data!")
        return
    
    print(f"\n✅ TaskB data loaded successfully!")
    print(f"Train domains: {list(train_data.keys())}")
    print(f"Test domains: {list(test_data.keys())}")
    
    # Show data examples
    print("\n📊 Data examples:")
    for domain in domains:
        if domain in train_data and train_data[domain]:
            print(f"\n{domain} train example:")
            print(json.dumps(train_data[domain][0], indent=2, ensure_ascii=False))
            break

    for domain in domains:
        if domain in test_data and test_data[domain]:
            print(f"\n{domain} test example:")
            print(json.dumps(test_data[domain][0], indent=2, ensure_ascii=False))
            break
    
    # Process each domain
    all_results = {}
    
    for domain in domains:
        if domain in train_data and domain in test_data:
            # Create new classifier for each domain
            # (to avoid problems with different classes)
            classifier = EmbeddingTermClassifier(
                model_name="Qwen/Qwen3-Embedding-4B",  # Use same model as in dev.ipynb
                max_length=8192
            )
            
            try:
                predictions, predictions_with_graph = process_single_domain(
                    domain, 
                    train_data[domain], 
                    test_data[domain], 
                    classifier
                )
                all_results[domain] = (predictions, predictions_with_graph)
                
                # Memory cleanup after each domain
                del classifier
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"❌ Error processing domain {domain}: {e}")
                continue
        else:
            print(f"⚠️  Skipping domain {domain} - no data available")
    
    print(f"\n🎉 Processing completed!")
    print(f"Successfully processed domains: {len(all_results)}")


if __name__ == "__main__":
    main() 
