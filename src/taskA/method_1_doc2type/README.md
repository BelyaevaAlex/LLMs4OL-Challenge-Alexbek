# Method v3_t1: Universal Term Extraction with RAG Support

## Description

Method v3_t1 is a universal method for extracting ontological terms from documents with RAG (Retrieval-Augmented Generation) support. The system can use both traditional few-shot examples and contextually relevant RAG examples embedded directly in input documents.

## Main Features

- **RAG support**: Use of examples from the "RAG" field in each document
- **Few-shot learning**: Traditional few-shot examples from separate files
- **TF-IDF integration**: Support for TF-IDF hints in examples and documents
- **Structured output**: Structured output through outlines (optional)
- **Universality**: Work with any models supporting conversation format
- **Determinism**: Reproducible generation with fixed seed

## Architecture

### Main Files

- `data.py` - Functions for data preparation and conversation format
- `inference.py` - Main inference logic with RAG support
- `train.py` - Model training (inherits from method_v2_t1)
- `run_inference.sh` - Script for running inference with RAG

### Key Functions

#### data.py
- `extract_rag_examples()` - Extracting RAG examples from document
- `build_conversation_for_training()` - Creating conversation for training
- `build_conversation_for_inference()` - Creating conversation for inference
- `build_hf_dataset()` - Creating HuggingFace Dataset

#### inference.py
- `extract_terms_from_document_structured()` - Extracting terms with structured output
- `process_documents()` - Processing multiple documents
- `generate_output_filename()` - Auto-generation of filenames

## Usage

### Basic RAG Inference

```bash
python -m src.taskA.method_v3_t1.inference \
    --model-path models/trained_model \
    --input documents_with_rag.jsonl \
    --use-rag \
    --use-tfidf \
    --seed 42
```

### RAG with Random Selection

```bash
python -m src.taskA.method_v3_t1.inference \
    --model-path models/trained_model \
    --input documents_with_rag.jsonl \
    --use-rag \
    --random-few-shot 3 \
    --use-structured-output \
    --seed 42
```

### Traditional Few-shot

```bash
python -m src.taskA.method_v3_t1.inference \
    --model-path models/trained_model \
    --input documents.jsonl \
    --few-shot examples.jsonl \
    --random-few-shot 5 \
    --use-tfidf \
    --seed 42
```

### Command Line Arguments

- `--model-path` - Path to trained model (required)
- `--input` - Input JSONL file with documents (required)
- `--output` - Output file (optional, auto-generated)
- `--few-shot` - File with few-shot examples (optional)
- `--random-few-shot N` - Number of random examples (optional)
- `--use-rag` - Use RAG examples from input documents
- `--use-tfidf` - Use TF-IDF hints
- `--use-structured-output` - Use structured output through outlines
- `--seed` - Seed for reproducibility (default 42)

## Data Format

### Input Documents with RAG

```jsonl
{
  "id": "0_0",
  "title": "Document Title",
  "text": "Document content...",
  "OL": ["term1", "term2"],
  "TF-IDF": ["hint1", "hint2"],
  "RAG": [
    {
      "id": "related_0_1",
      "title": "Related Document 1",
      "text": "Related content...",
      "OL": ["related_term1", "related_term2"],
      "TF-IDF": ["related_hint1", "related_hint2"]
    }
  ]
}
```

### Few-shot Examples

```jsonl
{"id": "example_1", "title": "Title", "text": "Content...", "OL": ["term1", "term2"], "TF-IDF": ["hint1", "hint2"]}
```

### Output Format

```json
{
  "results": [
    {
      "id": "0_0",
      "title": "Document Title",
      "text": "Document content...",
      "generated_text": "{\"terms\": [\"extracted_term1\", \"extracted_term2\"]}",
      "extracted_terms": ["extracted_term1", "extracted_term2"],
      "true_terms": ["term1", "term2"],
      "metrics": {
        "exact_precision": 0.8,
        "exact_recall": 0.9,
        "exact_f1": 0.85,
        "soft_precision": 0.9,
        "soft_recall": 0.95,
        "soft_f1": 0.92
      },
      "rag_examples_count": 3
    }
  ],
  "summary_metrics": {
    "exact_precision": 0.82,
    "exact_recall": 0.88,
    "exact_f1": 0.85,
    "soft_precision": 0.89,
    "soft_recall": 0.92,
    "soft_f1": 0.90
  },
  "config": {
    "model_path": "models/trained_model",
    "use_rag": true,
    "use_tfidf": true,
    "use_structured_output": true,
    "seed": 42,
    "deterministic": true
  }
}
```

## Example Selection Logic

1. **RAG mode** (`--use-rag`):
   - Examples are extracted from the "RAG" field of each document
   - If `--random-few-shot N` is specified, N random RAG examples are selected
   - Few-shot examples from `--few-shot` are ignored

2. **Few-shot mode** (without `--use-rag`):
   - Examples from the `--few-shot` file are used
   - If `--random-few-shot N` is specified, N random examples are selected
   - The "RAG" field in documents is ignored

3. **Fallback logic**:
   - If RAG examples are unavailable, the system can use few-shot examples
   - If examples are completely unavailable, only the system prompt is used

## Advantages of RAG Approach

1. **Contextual relevance** - Examples are selected individually for each document
2. **Automatic adaptation** - No need to create separate few-shot files
3. **Improved quality** - Use of most relevant examples
4. **Flexibility** - Different number of examples for different documents
5. **TF-IDF integration** - RAG examples support TF-IDF hints

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- tqdm
- fuzzywuzzy
- pydantic
- outlines (optional, for structured output)

## Compatibility

- Supports all models with conversation format (Llama, Qwen, Mistral, etc.)
- Backward compatibility with method_v2_t1
- Integration with existing TF-IDF data
- Support for structured output through outlines
