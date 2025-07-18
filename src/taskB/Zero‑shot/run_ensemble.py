import os
from pathlib import Path
from ensemble_similarity_zs import run_ensemble_pipeline

def main():
    # Base paths
    base_dir = Path("../../../2025/TaskB-TermTyping")
    domains = ["B4_blind", "B5_blind", "B6_blind"]
    
    # Create output directories for each domain
    for domain in domains:
        domain_dir = base_dir / domain
        output_dir = domain_dir / "predictions"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*50}")
        print(f"Running Ensemble method for domain: {domain}")
        print(f"{'='*50}\n")
        
        run_ensemble_pipeline(
            data_dir=str(domain_dir),
            output_dir=str(output_dir)
        )

if __name__ == "__main__":
    main() 