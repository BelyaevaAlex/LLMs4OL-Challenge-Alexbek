#!/bin/bash
set -e

echo "================================================================================"
echo "UNIVERSAL DATASET PROCESSING SCRIPT FOR TASK A"
echo "================================================================================"
echo ""

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "Step 1: Adding TF-IDF to all original documents (train/documents.jsonl)..."
cp -r ../../2025/TaskA-Text2Onto/ ../../2025/TaskA-Text2Onto-Processed/

echo ""
echo "Step 2: Adding TF-IDF to all documents"
python tfidf_ngrams_processor.py 

echo ""
echo "Step 3: Creating docs2terms.jsonl using TF-IDF enhanced documents..."
python create_docs2terms.py

echo ""
echo "Step 4: Creating types2docs.json from processed files..."
python create_types2docs.py

echo ""
echo "Step 5: Creating docs2terms_types.jsonl using all processed data..."
python generate_docs2terms_types.py

echo ""
echo "================================================================================"
echo "ALL PROCESSING COMPLETED SUCCESSFULLY!"
echo "Results saved to: ../../2025/TaskA-Text2Onto-Processed"
echo ""
echo "Final files created:"
echo "  - docs2terms.jsonl (with TF-IDF)"
echo "  - documents.jsonl (test files with TF-IDF)" 
echo "  - docs2terms_types.jsonl (with TF-IDF)"
echo "  - types2docs.json"
echo "================================================================================" 