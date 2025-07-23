# Method v3_t2: Term-to-Type Classification with RAG

## Description

Method v3_t2 implements term-to-type classification task using RAG (Retrieval-Augmented Generation). The system predicts semantic types for given terms based on contextually relevant examples.

## Project Structure

```
src/taskA/method_v3_t2/
├── data.py                    # Data processing module
├── inference.py               # Inference script
├── run_inference_base.sh      # Bash script for running
├── requirements.txt           # Dependencies
├── dev.ipynb                  # Notebook for development and data preparation
└── README.md                  # This file
```

## Term-to-Type Task

The task is to predict semantic types for terms:

**Input:** Term (e.g., "personal pronoun")  
**Output:** List of types (e.g., ["pronoun", "part of speech"])

## RAG Approach

The system uses RAG to improve prediction quality:

1. **Building embeddings:** Uses Qwen3-Embedding-4B to obtain term embeddings
2. **Finding similar examples:** Finds 10 most semantically similar terms from train data
3. **Contextual learning:** Uses found examples as few-shot examples for LLM

## Data

The system works with three domains:
- **engineering** - engineering terms
- **scholarly** - scholarly terms  
- **ecology** - ecological terms

Data format:
```json
[
  {
    "term": "personal pronoun",
    "types": ["pronoun", "part of speech"],
    "RAG": [
      {
        "term": "possessive pronoun", 
        "types": ["pronoun", "part of speech"]
      },
      // ... 9 more similar examples
    ]
  }
]
```

## Usage

### 1. Data Preparation

Run the `dev.ipynb` notebook for:
- Loading source data
- Building similarity matrices using embeddings
- Creating files with RAG examples

### 2. Inference

```bash
# Basic run for engineering test data
./run_inference_base.sh engineering test microsoft/Qwen2.5-14B-Instruct

# For scholarly train data
./run_inference_base.sh scholarly train microsoft/Qwen2.5-14B-Instruct

# For ecology train data (no test data)
./run_inference_base.sh ecology train microsoft/Qwen2.5-14B-Instruct
```

### 3. Manual Run

```bash
python -m src.taskA.method_v3_t2.inference \
    --model-path microsoft/Qwen2.5-14B-Instruct \
    --input 2025/TaskA-Text2Onto-TFIDF-with_scores/engineering/test/terms2types.json \
    --use-rag \
    --random-few-shot 5 \
    --use-structured-output \
    --seed 42
```

## Parameters

- `--model-path`: Path to model (HuggingFace or local)
- `--input`: Input JSON file with terms
- `--use-rag`: Use RAG examples from "RAG" field
- `--random-few-shot N`: Randomly select N examples from available RAG examples
- `--use-structured-output`: Use structured output through outlines
- `--seed`: Seed for reproducibility

## Metrics

The system calculates the following metrics:

1. **Jaccard Similarity**: Intersection of types / union of types
2. **F1 Score**: Harmonic mean of precision and recall for types
3. **Exact Match Accuracy**: Percentage of terms with exactly matching types

## Output Files

Results are saved in JSON format:
```json
{
  "results": [
    {
      "term": "personal pronoun",
      "predicted_types": ["pronoun", "part of speech"],
      "true_types": ["pronoun", "part of speech"],
      "rag_examples_count": 5,
      "generated_text": "..."
    }
  ],
  "summary_metrics": {
    "jaccard_similarity": 0.85,
    "f1_score": 0.87,
    "exact_match_accuracy": 0.73
  },
  "config": {
    "model_path": "microsoft/Qwen2.5-14B-Instruct",
    "use_rag": true,
    "random_few_shot_count": 5,
    "seed": 42
  }
}
```

## Dependencies

See `requirements.txt` for complete list of dependencies. Main ones:
- torch>=2.0.0
- transformers>=4.40.0
- outlines>=0.0.34 (for structured output)
- pandas, numpy, tqdm

## Features

1. **Deterministic generation**: Uses greedy decoding for reproducibility
2. **Structured output support**: Through outlines library for reliable parsing
3. **Automatic filenames**: Generation of output filenames based on parameters
4. **Memory handling**: GPU memory cleanup between batches
5. **Logging**: Detailed inference process logs 