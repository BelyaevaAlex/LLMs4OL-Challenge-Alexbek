# Method v5: Universal Terms and Types Extraction with Enhanced TF-IDF Support

## Description

Method v5 is an enhanced version of a universal method for extracting ontological terms and types from documents. The system works with our new combined docs2terms_types datasets and supports both semantic and statistical hints.

## Main Features

- **Dual extraction**: Prediction of both terms and types
- **TF-IDF support**: Use of statistical hints from TF-IDF analysis
- **Semantic hints**: Support for ontological hints from OL
- **RAG support**: Automatic addition of few-shot examples based on cosine similarity
- **Structured output**: Mandatory structured output through outlines
- **Enhanced datasets**: Work with combined docs2terms_types.jsonl files
- **Comprehensive metrics**: Separate metrics for terms and types
- **Backward compatibility**: Compatibility with original data formats

## Architecture

### Main Files

- `data.py` - Functions for data preparation with terms and types support
- `inference.py` - Inference with terms and types prediction
- `train.py` - Model training with new data format support
- `requirements.txt` - Project dependencies

### Key Functions

#### data.py
- `load_docs2terms_types_dataset()` - Loading combined datasets
- `extract_data_from_document()` - Extracting all data from document
- `build_conversation_for_training()` - Creating conversation for training
- `build_conversation_for_inference()` - Creating conversation for inference
- `build_hf_dataset()` - Creating HuggingFace Dataset

#### inference.py
- `extract_terms_and_types_from_document_structured()` - Extracting terms and types
- `process_documents()` - Processing multiple documents
- `calculate_metrics()` - Calculating metrics for terms and types separately

## Usage

### Inference with Base Model

```bash
python -m src.taskA.method_v5.inference \
    --model-path microsoft/DialoGPT-small \
    --input ../../2025/TaskA-Text2Onto/engineering/train/docs2terms_types.jsonl \
    --use-tfidf \
    --use-semantic \
    --seed 42
```

### Inference with TF-IDF Combined Data

```bash
python -m src.taskA.method_v5.inference \
    --model-path models/trained_model \
    --input 2025/TaskA-Text2Onto-TFIDF/engineering/train/docs2terms_types.jsonl \
    --use-tfidf \
    --random-few-shot 3 \
    --seed 42
```

### Inference with RAG Data (recommended for test data)

```bash
python -m src.taskA.method_v5.inference \
    --model-path qwen/Qwen2.5-14B-Instruct \
    --input 2025/TaskA-Text2Onto-TFIDF-with-RAG/text2onto_ecology_test_documents_with_rag.jsonl \
    --use-tfidf \
    --seed 42
```

### Model Training

```bash
python -m src.taskA.method_v5.train \
    --train-data ../../2025/TaskA-Text2Onto-TFIDF/engineering/train/docs2terms_types.jsonl \
    --val-data ../../2025/TaskA-Text2Onto-TFIDF/ecology/train/docs2terms_types.jsonl \
    --model-name microsoft/DialoGPT-medium \
    --output-dir ./trained_models/method_v5_engineering \
    --epochs 3 \
    --batch-size 4 \
    --use-tfidf \
    --use-semantic \
    --use-wandb
```

### Command Line Arguments

#### Inference
- `--model-path` - Path to model (required)
- `--input` - Input JSONL file with documents (required)
- `--output` - Output file (optional, auto-generated)
- `--few-shot` - File with few-shot examples (optional)
- `--random-few-shot N` - Number of random examples (optional)
- `--use-tfidf` - Use TF-IDF hints
- `--seed` - Seed for reproducibility (default 42)

#### Training
- `--train-data` - Path to training data (required)
- `--val-data` - Path to validation data (required)
- `--model-name` - Base model for fine-tuning
- `--output-dir` - Directory for saving model (required)
- `--epochs` - Number of training epochs
- `--batch-size` - Batch size
- `--learning-rate` - Learning rate
- `--use-tfidf` - Use TF-IDF hints during training
- `--use-wandb` - Use Weights & Biases for logging

## Data Format

### Input Documents (docs2terms_types.jsonl)

```jsonl
{
  "document_id": "0_0",
  "title": "Document Title",
  "text": "Document content...",
  "types": ["type1", "type2"],
  "terms": ["term1", "term2"],
  "types_count": 2,
  "terms_count": 2,
  "tfidf_terms": ["tfidf1", "tfidf2"],
  "ol_terms": ["ol1", "ol2"],
  "tfidf_terms_count": 2,
  "ol_terms_count": 2,
  "has_tfidf_data": true
}
```

### Output Format

```json
{
  "results": [
    {
      "document_id": "0_0",
      "title": "Document Title",
      "text": "Document content...",
      "generated_text": "{\"terms\": [\"extracted_term1\"], \"types\": [\"extracted_type1\"]}",
      "extracted_terms": ["extracted_term1"],
      "extracted_types": ["extracted_type1"],
      "true_terms": ["term1", "term2"],
      "true_types": ["type1", "type2"],
      "metrics": {
        "terms_exact_precision": 0.8,
        "terms_exact_recall": 0.5,
        "terms_exact_f1": 0.62,
        "types_exact_precision": 0.9,
        "types_exact_recall": 0.45,
        "types_exact_f1": 0.6,
        "terms_soft_precision": 0.9,
        "terms_soft_recall": 0.55,
        "terms_soft_f1": 0.68,
        "types_soft_precision": 0.95,
        "types_soft_recall": 0.5,
        "types_soft_f1": 0.65
      },
      "tfidf_suggestions_count": 2,
      "semantic_suggestions_count": 2
    }
  ],
  "summary_metrics": {
    "terms_exact_precision": 0.82,
    "terms_exact_recall": 0.48,
    "terms_exact_f1": 0.61,
    "types_exact_precision": 0.87,
    "types_exact_recall": 0.42,
    "types_exact_f1": 0.57,
    "terms_soft_precision": 0.89,
    "terms_soft_recall": 0.52,
    "terms_soft_f1": 0.66,
    "types_soft_precision": 0.92,
    "types_soft_recall": 0.47,
    "types_soft_f1": 0.62
  },
  "config": {
    "model_path": "models/trained_model",
    "use_tfidf": true,
    "use_semantic": true,
    "use_structured_output": true,
    "seed": 42,
    "total_documents": 1
  }
}
```

## Main Improvements over method_v3_t1

### 1. Dual Prediction
- Now the model predicts both terms and types
- Separate metrics for each prediction type
- Structured output supports both fields

### 2. Enhanced Hint Support
- **TF-IDF hints**: Statistical terms from TF-IDF analysis
- **Semantic hints**: Ontological terms from OL
- Ability to use both types of hints simultaneously

### 3. Improved Metrics
- Separate exact and soft metrics for terms and types
- More detailed reporting
- Support for wandb visualization

### 4. New Data Format
- Work with combined docs2terms_types.jsonl files
- Support for TF-IDF data from combined datasets
- Backward compatibility with original formats

### 5. Enhanced Training
- Honest inference test during training for terms and types
- Support for few-shot example masking
- Integration with wandb for monitoring

## RAG Data for Few-shot Learning

### Creating RAG Data

We created an enhanced version of datasets with RAG data (few-shot examples) to improve inference quality. The `add_rag_data.py` script automatically adds 3 most similar examples from the training set to each test document based on cosine similarity.

### RAG Data Structure

```jsonl
{
  "id": "1012_0",
  "title": "Document Title",
  "text": "Document content...",
  "RAG": [
    {
      "id": "0_0",
      "title": "Similar document 1",
      "text": "Similar content...",
      "terms": ["term1", "term2"],
      "types": ["type1", "type2"],
      "TF-IDF": ["tfidf1", "tfidf2"],
      "OL": ["ol1", "ol2"]
    },
    {
      "id": "0_1",
      "title": "Similar document 2",
      "text": "Similar content...",
      "terms": ["term3", "term4"],
      "types": ["type3", "type4"],
      "TF-IDF": ["tfidf3", "tfidf4"],
      "OL": ["ol3", "ol4"]
    },
    {
      "id": "0_2",
      "title": "Similar document 3",
      "text": "Similar content...",
      "terms": ["term5", "term6"],
      "types": ["type5", "type6"],
      "TF-IDF": ["tfidf5", "tfidf6"],
      "OL": ["ol5", "ol6"]
    }
  ]
}
```

### Created RAG Files

**Location**: `2025/TaskA-Text2Onto-TFIDF-with-RAG/`

- `text2onto_ecology_test_documents_with_rag.jsonl` - Ecology test documents with RAG
- `text2onto_engineering_test_documents_with_rag.jsonl` - Engineering test documents with RAG
- `text2onto_scholarly_test_documents_with_rag.jsonl` - Scholarly test documents with RAG
- `ecology_train_docs2terms_types.jsonl` - Ecology training data
- `engineering_train_docs2terms_types.jsonl` - Engineering training data
- `scholarly_train_docs2terms_types.jsonl` - Scholarly training data

### Using RAG Data

```bash
# Inference with RAG data for ecology
python -m src.taskA.method_v5.inference \
    --model-path qwen/Qwen2.5-14B-Instruct \
    --input 2025/TaskA-Text2Onto-TFIDF-with-RAG/text2onto_ecology_test_documents_with_rag.jsonl \
    --use-tfidf \
    --use-semantic \
    --seed 42

# Inference with RAG data for engineering
python -m src.taskA.method_v5.inference \
    --model-path qwen/Qwen2.5-14B-Instruct \
    --input 2025/TaskA-Text2Onto-TFIDF-with-RAG/text2onto_engineering_test_documents_with_rag.jsonl \
    --use-tfidf \
    --use-semantic \
    --seed 42
```

### Creating RAG Data for Other Domains

```bash
# Add RAG data for any domain
python src/taskA/add_rag_data.py \
    --domain DOMAIN_NAME \
    --data-dir 2025/TaskA-Text2Onto-TFIDF \
    --scores-dir 2025/TaskA-Text2Onto-TFIDF-with_scores \
    --output-dir 2025/TaskA-Text2Onto-TFIDF-with-RAG \
    --k 3
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test Inference with Base Model (Qwen recommended)
```bash
# For ecology with RAG data
python -m src.taskA.method_v5.inference \
    --model-path qwen/Qwen2.5-14B-Instruct \
    --input 2025/TaskA-Text2Onto-TFIDF-with-RAG/text2onto_ecology_test_documents_with_rag.jsonl \
    --use-tfidf \
    --seed 42

# For engineering with RAG data
python -m src.taskA.method_v5.inference \
    --model-path qwen/Qwen2.5-14B-Instruct \
    --input 2025/TaskA-Text2Onto-TFIDF-with-RAG/text2onto_engineering_test_documents_with_rag.jsonl \
    --use-tfidf \
    --seed 42
```

### 3. Train Model
```bash
python -m src.taskA.method_v5.train \
    --train-data 2025/TaskA-Text2Onto-TFIDF-with-RAG/engineering_train_docs2terms_types.jsonl \
    --val-data 2025/TaskA-Text2Onto-TFIDF-with-RAG/ecology_train_docs2terms_types.jsonl \
    --model-name qwen/Qwen2.5-14B-Instruct \
    --output-dir ./models/method_v5_trained \
    --epochs 2 \
    --batch-size 2 \
    --use-tfidf
```

### 4. Inference with Trained Model
```bash
python -m src.taskA.method_v5.inference \
    --model-path ./models/method_v5_trained/final_model \
    --input 2025/TaskA-Text2Onto-TFIDF-with-RAG/text2onto_scholarly_test_documents_with_rag.jsonl \
    --use-tfidf \
    --seed 42
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- **Outlines** (required for structured output)
- CUDA (recommended for training)

### Install Dependencies
```bash
pip install -r requirements.txt
pip install outlines
```

## Compatibility

- Supports all models with conversation format (DialoGPT, Llama, Qwen, Mistral, etc.)
- Backward compatibility with method_v3_t1
- Integration with existing TF-IDF data
- **Mandatory** structured output through outlines (guarantees correct JSON)

## Hint Logic

1. **TF-IDF mode** (`--use-tfidf`):
   - Uses "tfidf_terms" field from combined data
   - Fallback to "TF-IDF" field for backward compatibility

2. **Semantic mode** (`--use-semantic`):
   - Uses "ol_terms" field from combined data
   - Fallback to "OL" field for backward compatibility

3. **Combined mode** (`--use-tfidf --use-semantic`):
   - Uses both types of hints simultaneously
   - Maximum informativeness for the model

## Output Files

- **JSON results**: Detailed inference results with metrics
- **Trained models**: Complete models ready for inference
- **Training logs**: Detailed training process logs
- **Wandb integration**: Real-time metrics visualization 