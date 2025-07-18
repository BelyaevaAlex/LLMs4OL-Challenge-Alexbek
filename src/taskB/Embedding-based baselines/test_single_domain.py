"""
Test script to verify correct data formatting for a single domain
"""

import sys
import os
import json

# Add the project root to the path
project_root = '../../../'
sys.path.append(project_root)

from src.taskB.method_v1_1.term_classification_with_embeddings import EmbeddingTermClassifier, load_taskb_data

def test_single_domain(domain_name="MatOnto"):
    """Test a single domain and check data formatting"""
    print(f"🧪 Testing domain: {domain_name}")
    
    # Load the data
    train_data, test_data, domains = load_taskb_data()
    
    if domain_name not in train_data or domain_name not in test_data:
        print(f"❌ Domain {domain_name} not found")
        return
    
    # Show data examples
    print(f"\n📊 Examples from {domain_name}:")
    print("Train example:")
    print(json.dumps(train_data[domain_name][0], indent=2, ensure_ascii=False))
    print("\nTest example:")
    print(json.dumps(test_data[domain_name][0], indent=2, ensure_ascii=False))
    
    # Create the classifier (only using fast algorithms for testing)
    classifier = EmbeddingTermClassifier(
        model_name="Qwen/Qwen3-Embedding-4B",
        max_length=8192
    )
    
    # Clear any existing classifiers
    classifier.classifiers = {}
    
    # Prepare data (use only a small subset for testing)
    train_texts, test_texts, y, test_terms, df_train = classifier.prepare_data(
        train_data[domain_name][:10],  # First 10 samples only
        test_data[domain_name][:5]     # First 5 samples only
    )
    
    print(f"\n🔢 Data sizes:")
    print(f"Train: {len(train_texts)} texts")
    print(f"Test: {len(test_texts)} texts")
    print(f"Classes: {y.shape[1]} unique types")
    
    # Generate embeddings (small batch size for test speed)
    print("\n🧮 Generating embeddings...")
    X_train = classifier.get_embeddings(train_texts, batch_size=4)
    X_test = classifier.get_embeddings(test_texts, batch_size=4)
    
    print(f"Train embeddings: {X_train.shape}")
    print(f"Test embeddings: {X_test.shape}")
    
    # Quick training with logistic regression only
    print("\n🤖 Training classifier...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    
    lr_model = OneVsRestClassifier(LogisticRegression(max_iter=100))
    lr_model.fit(X_train.numpy(), y)
    classifier.classifiers['logistic_regression'] = lr_model
    
    # Make predictions
    print("\n🔮 Making predictions...")
    predictions = classifier.predict(X_test, test_terms, test_data[domain_name][:5])
    
    # Show results
    print(f"\n✅ Results for {domain_name}:")
    for name, pred_data in predictions.items():
        print(f"\n{name.upper()}:")
        for i, pred in enumerate(pred_data):
            print(f"  {pred}")
            if i >= 2:  # Show only first 3
                print("  ...")
                break
    
    # Check prediction format
    print(f"\n🔍 Format check:")
    sample_pred = predictions['logistic_regression'][0]
    required_keys = ['id', 'types']
    
    for key in required_keys:
        if key in sample_pred:
            print(f"  ✅ {key}: {type(sample_pred[key])}")
        else:
            print(f"  ❌ {key}: missing!")
    
    # Check if ID matches
    if 'id' in sample_pred:
        original_id = test_data[domain_name][0]['id']
        predicted_id = sample_pred['id']
        if original_id == predicted_id:
            print(f"  ✅ ID matches: {original_id}")
        else:
            print(f"  ❌ ID mismatch: {original_id} != {predicted_id}")
    
    print(f"\n🎉 Test for {domain_name} completed!")

if __name__ == "__main__":
    test_single_domain("MatOnto")
