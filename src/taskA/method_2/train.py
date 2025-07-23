#!/usr/bin/env python3
"""Training pipeline for Task A1 terms and types extraction using conversation models."""

import json
import argparse
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.model_selection import KFold

from datasets import DatasetDict
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)

from .data import (
    load_docs2terms_types_dataset, 
    build_hf_dataset, 
    load_few_shot_examples,
    build_conversation_for_inference,
    extract_data_from_document,
    SYSTEM_PROMPT
)


def tokenize_function(examples, tokenizer):
    """Tokenize the texts for causal language modeling."""
    # Handle both old format (text field) and new format (input_ids/labels fields)
    if "text" in examples:
        # Old format - simple tokenization
        tokenized = tokenizer(
            examples["text"], 
            truncation=True, 
            padding=False, 
            max_length=2048,
            return_tensors=None
        )
        # For causal LM labels = input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
    else:
        # New format - data already tokenized with proper masking
        tokenized = {
            "input_ids": examples["input_ids"],
            "labels": examples["labels"]
        }
    
    return tokenized


def extract_terms_and_types_from_generated_text(text: str) -> Tuple[List[str], List[str]]:
    """Extract terms and types from generated text by finding JSON in the response."""
    try:
        # Look for JSON in text
        json_match = re.search(r'\{[^}]*"terms"[^}]*"types"[^}]*\}', text)
        if not json_match:
            # Alternative search
            json_match = re.search(r'\{[^}]*"types"[^}]*"terms"[^}]*\}', text)
        
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
            return parsed.get("terms", []), parsed.get("types", [])
    except:
        pass
    return [], []


def run_inference_test(model, tokenizer, val_dataset, few_shot_examples=None, use_tfidf=False, use_semantic=False, random_few_shot_count=None, max_samples=100):
    """Honest test with full inference using conversation format."""
    print(f"Running honest inference test on {min(max_samples, len(val_dataset))} samples...")
    
    model.eval()
    
    # Separate metrics for terms and types
    terms_precisions = []
    terms_recalls = []
    terms_f1s = []
    types_precisions = []
    types_recalls = []
    types_f1s = []
    
    # Limit number of examples for test and take random sample
    if len(val_dataset) > max_samples:
        indices = np.random.choice(len(val_dataset), max_samples, replace=False)
        test_samples = val_dataset.select(indices.tolist())
    else:
        test_samples = val_dataset
    
    for i, doc in tqdm(enumerate(test_samples)):
        # Extract data from document
        title, text, true_terms, true_types, tfidf_suggestions, semantic_suggestions = extract_data_from_document(doc)
        
        # Build conversation with random few-shot support
        conversation = build_conversation_for_inference(
            title, text, few_shot_examples, use_tfidf, use_semantic, 
            tfidf_suggestions if use_tfidf else None,
            semantic_suggestions if use_semantic else None,
            random_few_shot_count
        )
        
        # Apply chat template
        inputs = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            truncation=True,
            max_length=1024  # Reduced for speed
        )
        
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        
        # Honest generation (fast settings)
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=256,  # Reduced for speed
                do_sample=False,     # Greedy decoding for speed
                pad_token_id=tokenizer.eos_token_id,
                num_beams=1,         # No beam search for speed
            )
        
        # Decode only new tokens
        new_tokens = outputs[0][inputs.shape[1]:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Extract terms and types
        pred_terms, pred_types = extract_terms_and_types_from_generated_text(generated_text)
        
        true_terms_set = set(true_terms)
        true_types_set = set(true_types)
        pred_terms_set = set(pred_terms)
        pred_types_set = set(pred_types)
        
        # Calculate metrics for terms
        if not pred_terms_set and not true_terms_set:
            terms_precisions.append(1.0)
            terms_recalls.append(1.0)
            terms_f1s.append(1.0)
        else:
            tp_terms = len(pred_terms_set & true_terms_set)
            precision_terms = tp_terms / len(pred_terms_set) if pred_terms_set else 0.0
            recall_terms = tp_terms / len(true_terms_set) if true_terms_set else 0.0
            
            if precision_terms + recall_terms > 0:
                f1_terms = 2 * precision_terms * recall_terms / (precision_terms + recall_terms)
            else:
                f1_terms = 0.0
                
            terms_precisions.append(precision_terms)
            terms_recalls.append(recall_terms)
            terms_f1s.append(f1_terms)
        
        # Calculate metrics for types
        if not pred_types_set and not true_types_set:
            types_precisions.append(1.0)
            types_recalls.append(1.0)
            types_f1s.append(1.0)
        else:
            tp_types = len(pred_types_set & true_types_set)
            precision_types = tp_types / len(pred_types_set) if pred_types_set else 0.0
            recall_types = tp_types / len(true_types_set) if true_types_set else 0.0
            
            if precision_types + recall_types > 0:
                f1_types = 2 * precision_types * recall_types / (precision_types + recall_types)
            else:
                f1_types = 0.0
                
            types_precisions.append(precision_types)
            types_recalls.append(recall_types)
            types_f1s.append(f1_types)
    
    test_metrics = {
        "test_terms_precision": np.mean(terms_precisions),
        "test_terms_recall": np.mean(terms_recalls),
        "test_terms_f1": np.mean(terms_f1s),
        "test_types_precision": np.mean(types_precisions),
        "test_types_recall": np.mean(types_recalls),
        "test_types_f1": np.mean(types_f1s),
        "test_samples": len(test_samples),
    }
    
    print(f"  Terms  - P={test_metrics['test_terms_precision']:.3f}, R={test_metrics['test_terms_recall']:.3f}, F1={test_metrics['test_terms_f1']:.3f}")
    print(f"  Types  - P={test_metrics['test_types_precision']:.3f}, R={test_metrics['test_types_recall']:.3f}, F1={test_metrics['test_types_f1']:.3f}")
    return test_metrics


class HonestInferenceCallback(TrainerCallback):
    """Callback for running honest inference test every eval_steps."""
    
    def __init__(self, val_dataset, tokenizer, few_shot_examples=None, use_tfidf=False, use_semantic=False, random_few_shot_count=None, max_samples=100):
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer  
        self.few_shot_examples = few_shot_examples
        self.use_tfidf = use_tfidf
        self.use_semantic = use_semantic
        self.random_few_shot_count = random_few_shot_count
        self.max_samples = max_samples
        
    def on_evaluate(self, args, state, control, model=None, logs=None, **kwargs):
        """Runs after each evaluation."""
        if model is not None:
            print(f"\n--- Running inference test (step {state.global_step}) ---")
            test_metrics = run_inference_test(
                model, self.tokenizer, self.val_dataset, self.few_shot_examples, 
                self.use_tfidf, self.use_semantic, self.random_few_shot_count, self.max_samples
            )
            
            # First try to log directly to wandb
            logged_to_wandb = False
            print(f"  Report to: {args.report_to}")
            if "wandb" in str(args.report_to):
                try:
                    import wandb
                    print(f"  Wandb imported, run status: {wandb.run is not None}")
                    if wandb.run is not None:
                        print(f"  Wandb run id: {wandb.run.id}")
                        wandb.log({
                            f"test/{k}": v for k, v in test_metrics.items()
                        }, step=state.global_step)
                        print(f"  Successfully logged test metrics to wandb (step {state.global_step})")
                        logged_to_wandb = True
                    else:
                        print("  Wandb run not active")
                except ImportError:
                    print("  Error: wandb not installed")
                except Exception as e:
                    print(f"  Error logging to wandb: {e}")
            
            # Additionally log to logs for file recording
            if logs is not None:
                logs.update(test_metrics)
                if not logged_to_wandb:
                    print(f"  Added test metrics to training logs")


def train_fold(train_dataset, val_dataset, model_name, output_dir, args):
    """Train a single fold."""
    print(f"\n=== Training fold ===")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    
    # Enable gradient checkpointing to save memory
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # Tokenize datasets
    tokenized_train = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=False,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train"
    )
    
    tokenized_val = val_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=False,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing val"
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
        pad_to_multiple_of=8,
    )
    
    # Load few-shot examples for callback
    few_shot_examples = None
    if args.few_shot_examples:
        few_shot_examples = load_few_shot_examples(Path(args.few_shot_examples))
        print(f"Loaded {len(few_shot_examples)} few-shot examples for evaluation")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        
        # Training parameters
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        
        # Evaluation and saving
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Optimization
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        
        # Mixed precision and efficiency
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        
        # Logging
        logging_strategy="steps",
        logging_steps=10,
        report_to=["wandb"] if args.use_wandb else [],
        run_name=f"method_v5_{Path(model_name).name}_{args.epochs}ep",
        
        # Other
        seed=42,
        data_seed=42,
        disable_tqdm=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        callbacks=[
            HonestInferenceCallback(
                val_dataset, tokenizer, few_shot_examples, 
                args.use_tfidf, args.use_semantic, args.random_few_shot_count, 
                args.max_inference_samples
            )
        ]
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    final_model_path = output_dir / "final_model"
    trainer.save_model(final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    return trainer


def train_and_evaluate_model(train_dataset, val_dataset, model_name, args):
    """Train and evaluate model with single fold."""
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = train_fold(train_dataset, val_dataset, model_name, output_dir, args)
    
    print(f"\nTraining completed!")
    print(f"Model saved to: {output_dir}")
    
    return trainer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train model for terms and types extraction")
    
    # Data arguments
    parser.add_argument("--train-data", type=str, required=True,
                        help="Path to training JSONL file with docs2terms_types format")
    parser.add_argument("--val-data", type=str, required=True,
                        help="Path to validation JSONL file with docs2terms_types format")
    parser.add_argument("--few-shot-examples", type=str, default=None,
                        help="Path to few-shot examples JSONL file")
    
    # Model arguments
    parser.add_argument("--model-name", type=str, default="microsoft/DialoGPT-small",
                        help="Base model to finetune")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for trained model")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size per device")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")
    
    # Evaluation arguments
    parser.add_argument("--eval-steps", type=int, default=100,
                        help="Evaluation steps")
    parser.add_argument("--save-steps", type=int, default=100,
                        help="Save steps")
    parser.add_argument("--max-inference-samples", type=int, default=100,
                        help="Maximum samples for inference test during training")
    
    # Data processing arguments
    parser.add_argument("--use-tfidf", action="store_true",
                        help="Use TF-IDF suggestions from documents")
    parser.add_argument("--use-semantic", action="store_true",
                        help="Use semantic suggestions from documents")
    parser.add_argument("--random-few-shot-count", type=int, default=None,
                        help="Number of random few-shot examples to use")
    parser.add_argument("--mask-few-shot", action="store_true", default=True,
                        help="Mask few-shot examples during training")
    
    # Other arguments
    parser.add_argument("--use-wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    
    args = parser.parse_args()
    
    print("="*80)
    print("METHOD V5 TRAINING")
    print("="*80)
    print(f"Train data: {args.train_data}")
    print(f"Val data: {args.val_data}")
    print(f"Model: {args.model_name}")
    print(f"Output dir: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Use TF-IDF: {args.use_tfidf}")
    print(f"Use semantic: {args.use_semantic}")
    print(f"Random few-shot: {args.random_few_shot_count}")
    print(f"Use wandb: {args.use_wandb}")
    
    # Initialize wandb if requested
    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project="method_v5_training",
                name=f"method_v5_{Path(args.model_name).name}_{args.epochs}ep",
                config=vars(args)
            )
            print("Wandb initialized successfully")
        except ImportError:
            print("Warning: wandb not available, falling back to local logging")
            args.use_wandb = False
    
    # Load datasets
    print(f"\nLoading training data from: {args.train_data}")
    train_data = load_docs2terms_types_dataset(Path(args.train_data))
    print(f"Loaded {len(train_data)} training examples")
    
    print(f"Loading validation data from: {args.val_data}")
    val_data = load_docs2terms_types_dataset(Path(args.val_data))
    print(f"Loaded {len(val_data)} validation examples")
    
    # Load few-shot examples for training
    few_shot_examples = None
    if args.few_shot_examples:
        print(f"Loading few-shot examples from: {args.few_shot_examples}")
        few_shot_examples = load_few_shot_examples(Path(args.few_shot_examples))
        print(f"Loaded {len(few_shot_examples)} few-shot examples")
    
    # Initialize tokenizer for dataset building
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Build HuggingFace datasets
    print("Building training dataset...")
    train_dataset = build_hf_dataset(
        train_data, tokenizer, few_shot_examples, 
        args.use_tfidf, args.use_semantic, args.mask_few_shot,
        args.random_few_shot_count
    )
    
    print("Building validation dataset...")
    val_dataset = build_hf_dataset(
        val_data, tokenizer, few_shot_examples, 
        args.use_tfidf, args.use_semantic, args.mask_few_shot,
        args.random_few_shot_count
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Train model
    trainer = train_and_evaluate_model(train_dataset, val_dataset, args.model_name, args)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    main() 