import os
import json
from term_classification_with_embeddings import EmbeddingTermClassifier

def load_json_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def get_domain_files(domain, base_path):
    domain_lower = domain.lower()
    train_file = os.path.join(base_path, domain, "train", "term_typing_train_data.json")
    test_file = os.path.join(base_path, domain, "test", f"{domain_lower}_term_typing_test_data.json")
    return train_file, test_file

def main():
    base_path = "../../../2025/TaskB-TermTyping"
    domains = ["MatOnto", "OBI", "SWEET"]
    
    for domain in domains:
        print(f"\n🔄 Processing domain {domain}")
        
        # Get file paths for the data
        train_file, test_file = get_domain_files(domain, base_path)
        
        # Load data
        train_data = load_json_data(train_file)
        test_data = load_json_data(test_file)
        
        print(f"📊 Training set size: {len(train_data)}")
        print(f"📊 Test set size: {len(test_data)}")
        
        # Create a directory to save results
        save_dir = f"predictions_embedding_{domain}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize the classifier
        classifier = EmbeddingTermClassifier()
        
        # Prepare data
        X_train_embeddings, y_train, X_test_embeddings, test_terms = classifier.prepare_data(train_data, test_data)
        
        # Train and evaluate classifiers
        clf, clf_with_graph, G = classifier.train_and_evaluate(domain, save_dir)
        
        # Predict on test data
        predictions, predictions_with_graph = classifier.predict(
            clf, clf_with_graph, G, X_test_embeddings, test_terms, test_data, save_dir
        )
        
        print(f"\n✅ Finished processing domain {domain}")
        print(f"📁 Results saved in directory: {save_dir}")

if __name__ == "__main__":
    main()
