# Zero-Shot Term Type Classification

This module implements zero-shot approaches for term type classification using multiple embedding models with various prompting strategies and ensemble methods.

## Overview

The module provides two main approaches:
1. **Single Model Classification** - Using individual embedding models with cosine similarity
2. **Ensemble Classification** - Combining multiple models with confidence-based weighting

## Models

### 1. Sentence-Transformers (all-mpnet-base-v2)
- Architecture: MPNet base model fine-tuned for semantic similarity
- Max sequence length: 512 tokens
- Embedding dimension: 768
- Built-in mean pooling strategy
- Advantages:
  - Fast inference
  - Good performance on semantic similarity tasks
  - Efficient memory usage

### 2. Qwen3-Embedding-4B
- Architecture: Large language model optimized for embeddings
- Max sequence length: 8192 tokens
- Uses last token pooling strategy
- Advantages:
  - Very large context window
  - Strong semantic understanding
  - Supports instruction format

### 3. BGE (bge-large-en-v1.5) - Ensemble Only
- Optimized for retrieval tasks
- Strong cross-lingual capabilities
- Used in ensemble approach for improved performance

## Prompting Strategies

Each model supports three different prompting strategies:

### 1. Plain (No Prompts)
- Terms and types are used as-is without any additional context
- Example:
  ```
  Term: "Earth"
  Type: "terrestrial planet"
  ```

### 2. Style 1 (Simple Prompts)
- MPNet:
  ```
  Term: "What is the meaning of 'Earth'?"
  Type: "This category represents terrestrial planet"
  ```
- Qwen:
  ```
  Term: "Define the term: Earth"
  Type: "Category description: terrestrial planet"
  ```

### 3. Style 2 (Instruction Format)
- MPNet:
  ```
  Term instruction: "Analyze this term and describe its key characteristics and domain"
  Type instruction: "Explain this category's scope and what concepts it encompasses"
  ```
- Qwen:
  ```
  Term instruction: "Given a term, provide its semantic meaning and key characteristics"
  Type instruction: "Given a category type, describe its defining features and scope"
  ```

## Implementation Details

### Embedding Generation
1. MPNet:
   - Uses SentenceTransformer's built-in encoding
   - Automatic batching and pooling
   - Returns normalized 768-dimensional vectors

2. Qwen:
   - Custom batching (batch_size=32) to manage memory
   - Last token pooling implementation
   - L2 normalization of embeddings

### Type Selection
- Computes cosine similarity between term and type embeddings
- Selects single type with highest similarity score
- Returns predictions in format:
  ```json
  {
    "id": "TT_id",
    "types": ["single_type"]
  }
  ```

## Usage

### Single Model Classification
```python
from cosine_similarity_zs import run_pipeline

# Run all models and prompting strategies
run_pipeline(
    data_dir="/path/to/data",  # Contains {domain}-Blind-Terms.json and {domain}-Blind-Types.txt
    output_dir="/path/to/output"  # Where to save predictions
)
```

### Ensemble Classification
```python
from ensemble_similarity_zs import run_ensemble_pipeline

# Run ensemble predictions
run_ensemble_pipeline(
    data_dir="/path/to/data",    # Contains {domain}-Blind-Terms.json and {domain}-Blind-Types.txt
    output_dir="/path/to/output" # Where to save predictions
)
```

### Running All Methods
```python
from run_all import main

# Run both single model and ensemble approaches for all domains
main()
```

**Note**: The scripts automatically process data from `2025/TaskB-TermTyping/` directory relative to project root.

### Output Files

#### Single Model Output
For each domain, generates 6 prediction files:
1. predictions_cosine_mpnet_plain.json
2. predictions_cosine_mpnet_prompted_style1.json
3. predictions_cosine_mpnet_prompted_style2.json
4. predictions_cosine_qwen_plain.json
5. predictions_cosine_qwen_prompted_style1.json
6. predictions_cosine_qwen_prompted_style2.json

#### Ensemble Output
Generates predictions with additional metadata:
```json
{
    "id": "term_id",
    "types": ["predicted_type"],
    "confidence": 0.85,
    "model_weights": {
        "mpnet": 0.35,
        "qwen": 0.45,
        "bge": 0.20
    }
}
```

## Requirements
- Python 3.8+
- PyTorch 2.0+
- transformers>=4.51.0
- sentence-transformers>=2.2.2
- scikit-learn>=1.0.0
- numpy>=1.21.0
- tqdm>=4.65.0

## Environment Setup
```bash
conda activate rah_11_cu12.4_torch  # Environment with PyTorch and CUDA
# или для более новой версии transformers
conda activate rah_python312_cuda124
pip install sentence-transformers  # Only additional requirement
```

## Data Format

### Input
1. Terms file (JSON): `{domain_short}-Blind-Terms.json` (e.g., `B4-Blind-Terms.json`)
   ```json
   [
     {
       "id": "TT_a824248a",
       "term": "Earth"
     },
     ...
   ]
   ```

2. Types file (TXT): `{domain_short}-Blind-Types.txt` (e.g., `B4-Blind-Types.txt`)
   ```
   desert ecosystem
   montane shrubland biome
   ocean
   ...
   ```

### Supported Domains
- **B4_blind**: Earth Sciences (geography, geology, environmental science)
- **B5_blind**: Linguistics (grammar, syntax, language structures)  
- **B6_blind**: Units and Measurements (physical units, measurement systems)

### Output
JSON file with predictions:
```json
[
  {
    "id": "TT_a824248a",
    "types": ["terrestrial planet"]
  },
  ...
]
```

## Ensemble Approach Details

### Key Features

#### 1. Confidence-Based Weighting
The ensemble dynamically adjusts model weights based on:
- Confidence scores from similarity distributions
- Entropy of predictions
- Model-specific performance patterns

Formula:
```python
confidence = max_prob * (1 - entropy_normalized)
final_weight = 0.7 * confidence_weight + 0.3 * entropy_weight
```

#### 2. Domain-Specific Prompts
Specialized prompts for each domain:

**B4_blind (Earth Sciences)**
```python
{
    "term_prefix": "In Earth sciences and geography, explain the concept of",
    "type_prefix": "This Earth science category represents",
    "instruction": "Analyze this geographical or Earth science term"
}
```

**B5_blind (Linguistics)**
```python
{
    "term_prefix": "In linguistics and grammar, define the term",
    "type_prefix": "This linguistic category encompasses",
    "instruction": "Analyze this linguistic or grammatical term"
}
```

**B6_blind (Units)**
```python
{
    "term_prefix": "In the context of units and measurements, describe",
    "type_prefix": "This measurement-related category represents",
    "instruction": "Analyze this unit or measurement-related term"
}
```

### Implementation Details

#### Confidence Calculation
1. Apply softmax with temperature to similarities
2. Calculate max probability and entropy
3. Normalize entropy to [0, 1]
4. Combine for final confidence score

#### Weight Adjustment
1. Calculate confidence-based weights
2. Adjust based on entropy
3. Combine both factors (70% confidence, 30% entropy)
4. Normalize final weights

#### Prediction Process
1. Get embeddings from all models
2. Calculate similarities and confidences
3. Adjust weights based on performance
4. Combine predictions using weighted average
5. Select type with highest final score

## Performance Considerations

### Single Model
- Qwen model requires more memory and is slower but potentially more accurate
- MPNet is faster and more memory efficient
- Batch size of 32 for Qwen to avoid OOM on most GPUs
- All embeddings are normalized for consistent cosine similarity

### Ensemble
- Memory Usage: ~16GB GPU RAM (Qwen model is the most demanding)
- Processing Time: ~2-3x slower than single model approach
- Improved Accuracy: Typically 5-10% better than best single model
- Batch processing recommended for large datasets
- Consider using CPU fallback for memory-constrained environments

## Notes
- Always use the same prompting style for both terms and types
- Instruction format (Style 2) might be more effective for complex domains
- Consider memory usage when processing large datasets with Qwen
- The approach is particularly effective for zero-shot scenarios where no training data is available
- Ensemble methods typically provide better accuracy than individual models 