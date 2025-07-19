import argparse
import sys
from meta_model import Qwen3CrossAttentionMetaModel
from inference_meta_model import run_meta_model_inference
from dataset import MetaModelDataset
import numpy as np
import pandas as pd
from inference_meta_model import threshold_analysis, plot_prediction_distribution
import os
from inference_meta_model import extract_relationships_from_matrix
from inference_meta_model import save_relationships_to_json
import zipfile

def main():
    parser = argparse.ArgumentParser(description="Run inference for specific domain")
    parser.add_argument("domain", type=str, help="Domain to process (e.g., DOID, MatOnto, OBI, etc.)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for computations (cuda/cpu)")
    
    args = parser.parse_args()
    domain = args.domain
    
    # Check that domain is valid
    valid_domains = ['DOID', 'MatOnto', 'OBI', 'PO', 'PROCO', 'SWEET', 'SchemaOrg', 'FoodOn']
    if domain not in valid_domains:
        print(f"Error: Domain '{domain}' is not supported. Available domains: {valid_domains}")
        sys.exit(1)
    
    print(f"Processing domain: {domain}")
    
    # Load experiment parameters
    roc_auc_best_params_df = pd.read_csv("roc_auc_best_params_df.csv")
    
    # Create dataset
    exp_df = roc_auc_best_params_df.loc[roc_auc_best_params_df['experiment_type'] == domain.lower()+"_try"]
    if exp_df.empty:
        print(f"Error: No experiment parameters found for domain {domain}")
        sys.exit(1)
    
    exp_folder = exp_df.experiment_run.values[0]

    pairs_path = f"/home/jovyan/rahmatullaev/rand_exps/LLMs4OL-Challenge/2025/TaskC-TaxonomyDiscovery/{domain}/train/{domain.lower()}_train_pairs.json"
    types_path = f"/home/jovyan/rahmatullaev/rand_exps/LLMs4OL-Challenge/2025/TaskC-TaxonomyDiscovery/{domain}/train/{domain.lower()}_train_types.txt"
    test_types_path = f"/home/jovyan/rahmatullaev/rand_exps/LLMs4OL-Challenge/2025/TaskC-TaxonomyDiscovery/{domain}/test/{domain.lower()}_test_types.txt"

    # Check file existence
    for path in [pairs_path, types_path, test_types_path]:
        if not os.path.exists(path):
            print(f"Error: File not found: {path}")
            sys.exit(1)

    dataset_train= MetaModelDataset(
        terms_path=types_path,
        relations_path=pairs_path,
        batch_size_1=128,
        batch_size_2=128,
        mode="all"
    )

    with open(test_types_path, "r") as f:
        test_types = f.read().splitlines()

    print(f"Terms in test dataset: {len(test_types)}")

    # Load model
    model_path = f"/home/jovyan/rahmatullaev/rand_exps/LLMs4OL-Challenge/src/taskC/method_v6_hm/experiments/{domain.lower()}_try/{exp_folder}/"
    
    if not os.path.exists(model_path):
        print(f"Error: Model folder not found: {model_path}")
        sys.exit(1)
    
    # Find model file
    model_file = None
    for file in os.listdir(model_path):
        if file.startswith("best_model_step_"):
            model_file = os.path.join(model_path, file)
            break
    
    if model_file is None:
        print(f"Error: Model file not found in folder: {model_path}")
        sys.exit(1)
    
    model_path = model_file
    
    # Check if result already exists
    if os.path.exists(os.path.join(model_path, f"relationships_base_{domain}.zip")):
        print(f"Relationships base already exists for {domain}")
        return
    
    print(f"Loading model from: {model_path}")
    model = Qwen3CrossAttentionMetaModel.from_pretrained(model_path)

    print(f"Model loaded: {model}")
    print(f"F1 threshold: {getattr(model, 'f1_threshold', 'not set')}")
    print(f"F1 threshold from file: {exp_df.best_f1_trs.values[0]:.5f}; value {exp_df.best_f1.values[0]:.5f}")

    print("Starting inference...")
    test_results = run_meta_model_inference(
        terms=test_types,
        model=model,
        batch_size=1024
    )

    print("Saving results...")
    from inference_meta_model import save_results
    save_results(test_results, save_path=model_path)

    print("Analyzing prediction distribution...")
    threshold_results = plot_prediction_distribution(
        pred_matrix=test_results['prediction_matrix'],
        save_path=os.path.join(model_path, "prediction_distribution.png")
    )

    print("Extracting relationships...")
    relationships = extract_relationships_from_matrix(
        pred_matrix=test_results['prediction_matrix'],
        terms=test_results['terms'],
        threshold=model.f1_threshold
    )

    print("Number of relationships: ", len(relationships))

    save_relationships_to_json(
        relationships=relationships,
        save_path=os.path.join(model_path, f"relationships_base_{domain}.json")
    )

    print("Creating zip archive...")
    # Create new zip archive
    zip_path = os.path.join(model_path, f'relationships_base_{domain}.zip')
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(os.path.join(model_path, f'relationships_base_{domain}.json'), f"relationships_base_{domain}.json")

    expected_ratio = len(dataset_train.relations) / len(dataset_train.terms)**2 * 100
    percentile = np.percentile(test_results['prediction_matrix'], 100-expected_ratio)
    percentile = round(percentile*1000)/1000

    print(f"Current square ratio: {len(relationships) / len(test_results['terms'])**2 * 100:.5f}%") 
    print(f"Expected square ratio: {expected_ratio:.5f}%") 

    print(f"Percentile for expected square ratio {expected_ratio:.5f} of pred_matrix is: {percentile:.5f}")

    print("Current ratio: ", len(relationships) / len(test_results["terms"])) 
    print("Expected ratio: ", len(dataset_train.relations) / len(dataset_train.terms)) 

    # Save statistics
    ratios_path = os.path.join(model_path, "ratios.txt")
    with open(ratios_path, "w") as f:
        f.write(f"Current square ratio: {len(relationships) / len(test_results['terms'])**2 * 100:.2f}%\n")
        f.write(f"Expected square ratio: {len(dataset_train.relations) / len(dataset_train.terms)**2 * 100:.2f}%\n")
        f.write(f"Percentile for expected square ratio {expected_ratio} of pred_matrix is: {percentile}\n")
        f.write(f"Current ratio: {len(relationships) / len(test_results['terms'])}\n")
        f.write(f"Expected ratio: {len(dataset_train.relations) / len(dataset_train.terms)}\n")

    print("Extracting relationships by percentile...")
    relationships_by_ratio = extract_relationships_from_matrix(
        pred_matrix=test_results['prediction_matrix'],
        terms=test_results['terms'],
        threshold=percentile
    )

    print("Number of relationships by percentile: ", len(relationships_by_ratio))

    save_relationships_to_json(
        relationships=relationships_by_ratio,
        save_path=os.path.join(model_path, f"relationships_ratio_{domain}.json")
    )

    print("Creating zip archive for relationships_ratio...")
    zip_ratio_path = os.path.join(model_path, f'relationships_ratio_{domain}.zip')
    with zipfile.ZipFile(zip_ratio_path, 'w') as zipf:
        zipf.write(os.path.join(model_path, f'relationships_ratio_{domain}.json'), f"relationships_ratio_{domain}.json")

    print(f"Domain {domain} processing completed successfully!")
    print(f"Results saved to: {model_path}")

if __name__ == "__main__":
    main()