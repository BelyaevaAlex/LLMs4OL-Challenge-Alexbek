# RAG-based Term Type Classification for TaskB

This module implements an approach to term type classification using Retrieval-Augmented Generation (RAG) for the TaskB-TermTyping task.

## Key Features

### 1. RAG Approach
- **Semantic Search**: Finding most relevant examples for each term
- **Dynamic Examples**: Using semantically similar terms as few-shot examples
- **Improved Accuracy**: More accurate classification through contextually relevant examples

### 2. Batch Processing
- **Batch Processing**: Simultaneous processing of multiple terms for faster inference
- **Memory Optimization**: Efficient GPU memory usage
- **Flash Attention 2**: Improved memory efficiency and speed

### 3. Structured Output
- **JSON Format**: Guaranteed structured output using outlines
- **Retry Logic**: Automatic retry attempts for failed generation (up to 3 times by default)
- **Fallback**: Graceful transition to regular generation when structured output fails

## Data Structure

### Input Data
The method expects different file formats for train and test data:

**Train Data**: `{domain}/train/term_typing_train_data.json`
```json
[
  {
    "term": "Earth",
    "types": ["terrestrial planet"]
  }
]
```

**Test Data**: `{domain}/test/terms2types.json`
```json
[
  {
    "id": "TT_a824248a",
    "term": "Earth"
  }
]
```

**RAG-Enhanced Data** (with similarity scores):
```json
[
  {
    "id": "TT_a824248a",
    "term": "Earth",
    "RAG": [
      {
        "term": "Mars",
        "types": ["terrestrial planet"],
        "similarity": 0.95
      },
      {
        "term": "Venus", 
        "types": ["terrestrial planet"],
        "similarity": 0.92
      }
    ]
  }
]
```

### Output Data
```json
[
  {
    "id": "TT_a824248a",
    "types": ["terrestrial planet"]
  }
]
```

## Installation and Setup

### 1. Activate Environment
```bash
conda activate rah_11_cu12.4_torch
```

### 2. Install Dependencies
```bash
pip install transformers torch tqdm pydantic
pip install outlines  # for structured output
```

### 3. Check GPU
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
```

## Usage

### Basic Inference
```bash
cd ../../../  # Go to project root
python src/taskB/method_v1_rag/inference.py \
    --model-path /path/to/model \
    --input /path/to/terms.json \
    --output /path/to/output.json
```

### With RAG and Batch Processing
```bash
python src/taskB/method_v1_rag/inference.py \
    --model-path /path/to/model \
    --input /path/to/terms.json \
    --output /path/to/output.json \
    --use-rag \
    --batch-size 8 \
    --max-retries 5
```

### Programmatic Usage
```python
from src.taskB.method_v1_rag.inference import process_terms
from src.taskB.method_v1_rag.data import load_few_shot_examples

# Load model
model, tokenizer = load_model_and_tokenizer("/path/to/model")

# Process terms
predictions, metrics = process_terms(
    model=model,
    tokenizer=tokenizer,
    terms_data=terms_data,
    use_rag=True,
    batch_size=8,
    max_retries=3
)
```

## Supported Domains

- **MatOnto**: Material Ontology - materials science terms
- **OBI**: Ontology for Biomedical Investigations - biomedical terms  
- **SWEET**: Semantic Web for Earth and Environmental Terminology - environmental terms

## System Prompt

The module uses a specialized system prompt for TaskB:

```
You are an expert in ontologies and semantic term classification. Your task is to determine semantic types for given terms based on the domain-specific ontology.

CRITICAL INSTRUCTIONS:
1. Analyze the provided term carefully
2. Determine the most appropriate semantic types from the domain-specific ontology
3. A semantic type is a GENERALIZING CONCEPT or CATEGORY that the term belongs to
4. Use the provided examples to understand the patterns and relationships between terms and their types
5. Respond ONLY in JSON format using " quotes
6. The reasoning should be concise and to the point, not too long. Write not more than 100 words in reasoning.

RESPONSE FORMAT:
{
  "term": "term name",
  "reasoning": "term name is similar to provided examples, but is not like the first example because ..., so it is not type1, but type2",
  "types": ["type2"]
}
```

## RAG Functionality

### Semantic Search
- Using embeddings to find semantically similar terms
- Sorting examples by similarity degree
- Dynamic selection of example quantity

### Prompt Integration
- Embedding RAG examples in prompts
- Contextually relevant few-shot examples
- Adaptation to term domain

## Evaluation Metrics

### Main Metrics
- **Jaccard Similarity**: Jaccard similarity between predicted and true types
- **F1 Score**: F1-score for multi-label classification
- **Exact Match Accuracy**: Accuracy of complete match of all types

### Metrics Format
```python
{
    'jaccard_similarity': 0.85,
    'f1_score': 0.82,
    'exact_match_accuracy': 0.78
}
```

## Data Files

### CSV Files with Scores
- `{Domain}_test_train_scores.csv`: Similarity scores between test and training terms
- `{Domain}_train_train_scores.csv`: Similarity scores within training set

### CSV Structure
```csv
term1,term2,similarity_score
"Earth","Mars",0.95
"Earth","Venus",0.92
```

## Performance

### Optimizations
- **Flash Attention 2**: Improved attention efficiency
- **Batch Processing**: Batch processing for acceleration
- **Memory Management**: Automatic GPU memory cleanup

### Recommendations
- **Batch Size**: 4-8 for large models, 16+ for smaller ones
- **Max Retries**: 3-5 for structured output
- **GPU Memory**: Minimum 16GB for large models

## Troubleshooting

### CUDA out of memory error
```bash
# Reduce batch_size
python inference.py --batch-size 2
```

### Structured output issues
```bash
# Disable structured output
python inference.py --no-structured-output
```

### Slow performance
```bash
# Increase batch_size if memory allows
python inference.py --batch-size 16
```

## Logging

The module creates detailed logs in the `inference_taskB.log` file:
- Processing progress
- Errors and warnings
- Performance metrics
- RAG examples information

## Next Steps

1. Experiment with different embedding models for RAG
2. Optimize the number of RAG examples
3. Adaptive strategies for example selection
4. Integration with other TaskB methods 