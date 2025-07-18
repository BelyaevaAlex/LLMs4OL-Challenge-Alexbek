# Term Classification using Embeddings

This module provides an alternative to TfidfVectorizer for term classification tasks using semantic embeddings.

## Advantages of Embeddings over TF-IDF

- **Semantic Understanding**: Better capture semantic relationships between terms
- **Contextual Information**: Consider the context of term usage
- **Dense Representations**: More efficient representation compared to sparse TF-IDF vectors
- **Pre-trained Models**: Utilize knowledge from large text corpora

## Data Structure

### Train Data
```json
[
  {
    "term": "protein folding",
    "types": ["biological process", "molecular function"]
  }
]
```

### Test Data
```json
[
  {
    "id": "TT_5a5763f5",
    "term": "newton"
  }
]
```

### Output Data (CodaLab format)
```json
[
  {"id": "TT_5a5763f5", "types": ["force unit"]},
  {"id": "TT_c4f339b8", "types": ["torque unit", "energy unit"]},
  {"id": "TT_ce5417fc", "types": ["pressure unit"]}
]
```

## Installation and Setup

### 1. Activate conda environment
```bash
conda activate rah_11_cu12.4_torch
# or for newer transformers version
conda activate rah_python312_cuda124
```

### 2. Install dependencies
```bash
pip install -r requirements_embedding.txt
```

### 3. Check GPU availability
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
```

## Usage

### Run for all TaskB domains
```bash
cd ../../../  # Go to project root
python src/taskB/method_v1_1/run_embedding_classification.py
```

**Note**: The script automatically processes data from `2025/TaskB-TermTyping/` directory relative to project root.

### Programmatic usage
```python
from src.taskB.method_v1_1.term_classification_with_embeddings import EmbeddingTermClassifier

# Create classifier
classifier = EmbeddingTermClassifier(
    model_name="Qwen/Qwen3-Embedding-4B",
    max_length=8192
)

# Prepare data
train_texts, test_texts, y, test_terms, df_train = classifier.prepare_data(train_data, test_data)

# Get embeddings
X_train = classifier.get_embeddings(train_texts, batch_size=8)
X_test = classifier.get_embeddings(test_texts, batch_size=8)

# Train classifiers
classifier.train_classifiers(X_train, y)

# Prediction
predictions = classifier.predict(X_test, test_terms, test_data)
```

## Supported TaskB Domains

- **MatOnto**: Material Ontology - materials science terms
- **OBI**: Ontology for Biomedical Investigations - biomedical terms
- **SWEET**: Semantic Web for Earth and Environmental Terminology - environmental terms

## Output Files

### Models
```
saved_models_embedding_{domain}/
├── embedding_logistic_regression_model.pkl
├── embedding_random_forest_model.pkl
├── embedding_xgboost_model.pkl
└── mlb.pkl
```

### Predictions
```
predictions_embedding_{domain}/
├── predictions_embedding_logistic_regression_{domain}.json
├── predictions_embedding_random_forest_{domain}.json
└── predictions_embedding_xgboost_{domain}.json
```

## Classifiers

### Always available (from sklearn):
1. **Logistic Regression**: OneVsRestClassifier with LogisticRegression
2. **Random Forest**: OneVsRestClassifier with RandomForestClassifier  
3. **Extra Trees**: OneVsRestClassifier with ExtraTreesClassifier (faster than Random Forest)
4. **Gradient Boosting**: OneVsRestClassifier with GradientBoostingClassifier (XGBoost alternative)
5. **SVM**: OneVsRestClassifier with SVC (only for small datasets < 10K)
6. **MLP**: OneVsRestClassifier with MLPClassifier (neural network)

### Additional (require installation):
7. **XGBoost**: XGBClassifier for multi-label classification
8. **LightGBM**: LGBMClassifier from Microsoft (fast gradient boosting)
9. **CatBoost**: CatBoostClassifier from Yandex (works well with categorical data)

## Technical Details

### Embedding Model
- **Model**: Qwen/Qwen3-Embedding-4B
- **Maximum length**: 8192 tokens
- **Embedding dimension**: 4096
- **Normalization**: L2 normalization

### Data Processing
- **Instruction**: "Given a term, find similar terms that have related semantic types or categories."
- **Train format**: "Term: {term}, Types: {types}"  
- **Test format**: "Term: {term}"
- **Pooling**: Last token pooling
- **Confidence threshold**: 0.3 for predictions
- **Empty prediction handling**: 
  - If all class probabilities are below threshold, select class with maximum probability
  - If no predictions at all, use "unknown" label

### Evaluation Metrics
- **Precision**: Prediction accuracy (weighted average)
- **Recall**: Prediction completeness (weighted average)
- **F1**: F1-score (weighted average)
- **Accuracy**: Overall accuracy

### Memory and Performance
- **Batch size**: 8 for embeddings
- **Automatic memory cleanup**: torch.cuda.empty_cache()
- **Domain processing**: separate classifier for each domain

## Comparison with TF-IDF

| Metric | TF-IDF | Embeddings |
|---------|---------|------------|
| Dimension | 10000+ (sparse) | 4096 (dense) |
| Semantics | No | Yes |
| Pre-training | No | Yes |
| Generalization | Limited | Good |
| Speed | Fast | Slower |

## Process Monitoring

The script outputs detailed progress information:
- Loading data for each domain
- Preparing embeddings (with progress bar)
- Training classifiers
- Saving results

## Troubleshooting

### CUDA out of memory error
```python
# Reduce batch_size
classifier = EmbeddingTermClassifier(batch_size=4)
```

### Slow performance
```python
# Use CPU if GPU is not available
device = torch.device('cpu')
```

### Compatibility errors
```bash
# Make sure the correct environment is activated
conda activate rah_11_cu12.4_torch
which python
```

## Next Steps

1. Compare quality with TF-IDF approach
2. Experiment with other embedding models (BGE, E5, etc.)
3. Fine-tune hyperparameters
4. Add new classifiers
5. Evaluate on additional metrics
6. Integrate with RAG approach for quality improvement
7. Experiment with ensemble methods 