#!/usr/bin/env python3
"""
Meta-Model Training Script - Simplified version for training only Meta-Model

This script provides:
- Training Meta-Model from scratch
- Automatic determination of optimal F1 threshold
- Flexible Qwen3 freezing
- Complete metrics and monitoring system

Usage example:
    python train_meta_model.py \
        --terms_path data/terms.txt \
        --relations_path data/relations.json \
        --output_dir experiments/new_approach \
        --dataset_name DATASET_NAME \
        --freeze_strategy except_last
"""

import sys
import os
import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Matplotlib setup for server environment
import matplotlib
matplotlib.use('Agg')  # Use backend without GUI
import matplotlib.pyplot as plt

# Wandb for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ wandb not found. Install: pip install wandb")

from meta_model import Qwen3CrossAttentionMetaModel, create_meta_model_from_scratch
from dataset import MetaModelDataset


def setup_logging(log_dir: str, log_level: str = "INFO") -> logging.Logger:
    """Setup logging to file and console"""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger("meta_model_training")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler for file
    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Handler for console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging configured. Log file: {log_file}")
    return logger


def apply_freeze_strategy(model: Qwen3CrossAttentionMetaModel, freeze_strategy: str, lora_rank: int = 8, lora_alpha: int = 16):
    """
    Apply weight freezing strategy
    
    Args:
        model: meta-model
        freeze_strategy: freezing strategy ("full", "except_last", "lora", "none")
        lora_rank: rank for LORA adapter
        lora_alpha: alpha for LORA adapter
    """
    logger = logging.getLogger("meta_model_training")
    
    if freeze_strategy == "full":
        # Full Qwen3 freezing
        logger.info("🧊 Applying full Qwen3 freezing")
        for param in model.qwen3_model.parameters():
            param.requires_grad = False
        
    elif freeze_strategy == "except_last":
        # Freeze all layers except the last one
        logger.info("🧊 Applying Qwen3 freezing except last layer")
        for param in model.qwen3_model.parameters():
            param.requires_grad = False
        
        # Unfreeze the last layer
        if hasattr(model.qwen3_model, 'layers') and model.qwen3_model.layers:
            last_layer = model.qwen3_model.layers[-1]
            for param in last_layer.parameters():
                param.requires_grad = True
            logger.info(f"✅ Last layer ({len(model.qwen3_model.layers)-1}) unfrozen")
        
        # Also unfreeze layer norm if exists
        if hasattr(model.qwen3_model, 'norm'):
            for param in model.qwen3_model.norm.parameters():
                param.requires_grad = True
            logger.info("✅ Final layer norm unfrozen")
    
    elif freeze_strategy == "lora":
        # Freeze base weights + add LORA adapters
        logger.info(f"🧊 Applying LORA adapter (rank={lora_rank}, alpha={lora_alpha})")
        
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            # Freeze all base weights
            for param in model.qwen3_model.parameters():
                param.requires_grad = False
            
            # LORA configuration
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.1,
            )
            
            # Apply LORA to model
            model.qwen3_model = get_peft_model(model.qwen3_model, lora_config)
            logger.info(f"✅ LORA adapter applied successfully")
            
        except ImportError:
            logger.warning("⚠️ PEFT library not found. Install: pip install peft")
            logger.info("🔄 Switching to 'except_last' strategy")
            apply_freeze_strategy(model, "except_last", lora_rank, lora_alpha)
            return
    
    elif freeze_strategy == "none":
        # No freezing - train all weights
        logger.info("🔥 No freezing - training all Qwen3 weights")
        for param in model.qwen3_model.parameters():
            param.requires_grad = True
    
    else:
        raise ValueError(f"Unknown freezing strategy: {freeze_strategy}")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    # Save strategy information in model for proper restoration
    model.freeze_strategy = freeze_strategy
    if freeze_strategy == "lora":
        model.lora_rank = lora_rank
        model.lora_alpha = lora_alpha
    
    logger.info(f"📊 Parameters:")
    logger.info(f"   Total: {total_params:,}")
    logger.info(f"   Trainable: {trainable_params:,}")
    logger.info(f"   Trainable ratio: {trainable_params/total_params*100:.2f}%")


def create_training_plots(training_log: List[Dict], save_dir: str):
    """Create training plots (loss, ROC AUC and learning rate)"""
    if len(training_log) == 0:
        return
    
    # Extract data for plots
    steps = []
    train_losses = []
    test_losses = []
    train_roc_aucs = []
    test_roc_aucs = []
    learning_rates = []
    
    for entry in training_log:
        if 'step' in entry:
            steps.append(entry['step'])
            train_losses.append(entry.get('train_loss', None))
            test_losses.append(entry.get('test_loss', None))
            train_roc_aucs.append(entry.get('train_roc_auc', None))
            test_roc_aucs.append(entry.get('test_roc_auc', None))
            learning_rates.append(entry.get('lr', None))
    
    if len(steps) == 0:
        return
    
    # Создаем графики (3 графика вместо 2)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    
    # График Loss
    ax1.set_title('Training and Test Loss')
    if any(l is not None for l in train_losses):
        valid_steps = [s for s, l in zip(steps, train_losses) if l is not None]
        valid_losses = [l for l in train_losses if l is not None]
        if valid_steps:
            ax1.plot(valid_steps, valid_losses, 'b-', label='Train Loss', linewidth=2)
    
    if any(l is not None for l in test_losses):
        valid_steps = [s for s, l in zip(steps, test_losses) if l is not None]
        valid_losses = [l for l in test_losses if l is not None]
        if valid_steps:
            ax1.plot(valid_steps, valid_losses, 'r-', label='Test Loss', linewidth=2)
    
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График ROC AUC
    ax2.set_title('Training and Test ROC AUC')
    if any(auc is not None for auc in train_roc_aucs):
        valid_steps = [s for s, auc in zip(steps, train_roc_aucs) if auc is not None]
        valid_aucs = [auc for auc in train_roc_aucs if auc is not None]
        if valid_steps:
            ax2.plot(valid_steps, valid_aucs, 'b-', label='Train ROC AUC', linewidth=2)
    
    if any(auc is not None for auc in test_roc_aucs):
        valid_steps = [s for s, auc in zip(steps, test_roc_aucs) if auc is not None]
        valid_aucs = [auc for auc in test_roc_aucs if auc is not None]
        if valid_steps:
            ax2.plot(valid_steps, valid_aucs, 'r-', label='Test ROC AUC', linewidth=2)
    
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('ROC AUC')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # График Learning Rate
    ax3.set_title('Learning Rate Schedule')
    if any(lr is not None for lr in learning_rates):
        valid_steps = [s for s, lr in zip(steps, learning_rates) if lr is not None]
        valid_lrs = [lr for lr in learning_rates if lr is not None]
        if valid_steps:
            ax3.plot(valid_steps, valid_lrs, 'g-', label='Learning Rate', linewidth=2)
    
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Learning Rate')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')  # Логарифмическая шкала для LR
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_progress.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path


def evaluate_train_roc_auc(model: Qwen3CrossAttentionMetaModel, train_loader: DataLoader, device: str, max_batches: int = 10) -> float:
    """Быстрое вычисление ROC AUC на небольшой выборке тренировочных данных"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (texts_1, texts_2, gt_matrix, dataset_name) in enumerate(train_loader):
            if batch_idx >= max_batches:
                break
                
            # Правильно извлекаем данные из DataLoader (batch_size=1)
            texts_1 = [item[0] for item in texts_1]  # Извлекаем строки из списков
            texts_2 = [item[0] for item in texts_2]  # Извлекаем строки из списков
            gt_matrix = gt_matrix.squeeze(0).to(device)  # Убираем размерность batch: [1, 32, 32] -> [32, 32]
            
            pred_matrix = model(texts_1, texts_2)
            
            all_predictions.extend(pred_matrix.flatten().cpu().numpy())
            all_targets.extend(gt_matrix.flatten().cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # ROC AUC
    if len(np.unique(all_targets)) > 1 and len(all_predictions) > 0:
        return roc_auc_score(all_targets, all_predictions)
    else:
        return 0.0


def weighted_bce_loss(pred_matrix: torch.Tensor, gt_matrix: torch.Tensor) -> torch.Tensor:
    """Взвешенный Binary Cross Entropy Loss"""
    ones_mask = (gt_matrix == 1)
    zeros_mask = (gt_matrix == 0)
    
    ones_count = ones_mask.sum().item()
    zeros_count = zeros_mask.sum().item()
    
    if ones_count == 0:
        return F.binary_cross_entropy(pred_matrix, gt_matrix)
    
    loss_ones = 0.0
    if ones_mask.sum() > 0:
        loss_ones = F.binary_cross_entropy(pred_matrix[ones_mask], gt_matrix[ones_mask])
    
    loss_zeros = 0.0
    if zeros_mask.sum() > 0:
        loss_zeros = F.binary_cross_entropy(pred_matrix[zeros_mask], gt_matrix[zeros_mask])
    
    total_loss = loss_ones / ones_count + loss_zeros / zeros_count
    return total_loss


def evaluate_meta_model(model: Qwen3CrossAttentionMetaModel, dataloader: DataLoader, device: str) -> Dict[str, float]:
    """Оценка мета-модели на тестовых данных"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for texts_1, texts_2, gt_matrix, dataset_name in dataloader:
            # Правильно извлекаем данные из DataLoader (batch_size=1)
            texts_1 = [item[0] for item in texts_1]  # Извлекаем строки из списков
            texts_2 = [item[0] for item in texts_2]  # Извлекаем строки из списков
            gt_matrix = gt_matrix.squeeze(0).to(device)  # Убираем размерность batch: [1, 32, 32] -> [32, 32]
            
            pred_matrix = model(texts_1, texts_2)
            
            all_predictions.extend(pred_matrix.flatten().cpu().numpy())
            all_targets.extend(gt_matrix.flatten().cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    metrics = {}
    
    # ROC AUC
    if len(np.unique(all_targets)) > 1:
        metrics['roc_auc'] = roc_auc_score(all_targets, all_predictions)
    else:
        metrics['roc_auc'] = 0.0
    
    # Метрики для разных порогов
    for threshold in [0.05, 0.15, 0.25, 0.5]:
        pred_binary = (all_predictions > threshold).astype(int)
        
        metrics[f'acc_{threshold}'] = accuracy_score(all_targets, pred_binary)
        metrics[f'f1_{threshold}'] = f1_score(all_targets, pred_binary, zero_division=0)
        metrics[f'precision_{threshold}'] = precision_score(all_targets, pred_binary, zero_division=0)
        metrics[f'recall_{threshold}'] = recall_score(all_targets, pred_binary, zero_division=0)
    
    return metrics


def analyze_thresholds(model: Qwen3CrossAttentionMetaModel, dataloader: DataLoader, device: str, save_dir: str = None) -> Dict[str, Any]:
    """Анализ мета-модели с различными порогами"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for texts_1, texts_2, gt_matrix, dataset_name in dataloader:
            # Правильно извлекаем данные из DataLoader (batch_size=1)
            texts_1 = [item[0] for item in texts_1]  # Извлекаем строки из списков
            texts_2 = [item[0] for item in texts_2]  # Извлекаем строки из списков
            gt_matrix = gt_matrix.squeeze(0).to(device)  # Убираем размерность batch: [1, 32, 32] -> [32, 32]
            
            pred_matrix = model(texts_1, texts_2)
            
            all_predictions.extend(pred_matrix.flatten().cpu().numpy())
            all_targets.extend(gt_matrix.flatten().cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Анализируем пороги от 0.01 до 0.95
    thresholds = np.arange(0.01, 0.95, 0.01)
    
    results = {
        'thresholds': thresholds.tolist(),
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    for threshold in thresholds:
        pred_binary = (all_predictions > threshold).astype(int)
        
        acc = accuracy_score(all_targets, pred_binary)
        prec = precision_score(all_targets, pred_binary, zero_division=0)
        rec = recall_score(all_targets, pred_binary, zero_division=0)
        f1 = f1_score(all_targets, pred_binary, zero_division=0)
        
        results['accuracy'].append(acc)
        results['precision'].append(prec)
        results['recall'].append(rec)
        results['f1'].append(f1)
    
    # Находим лучшие пороги
    best_thresholds = {
        'accuracy': thresholds[np.argmax(results['accuracy'])],
        'precision': thresholds[np.argmax(results['precision'])],
        'recall': thresholds[np.argmax(results['recall'])],
        'f1': thresholds[np.argmax(results['f1'])]
    }
    
    best_values = {
        'accuracy': np.max(results['accuracy']),
        'precision': np.max(results['precision']),
        'recall': np.max(results['recall']),
        'f1': np.max(results['f1'])
    }
    
    # Строим график
    if save_dir:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(thresholds, results['accuracy'], 'b-', linewidth=2)
        plt.axvline(best_thresholds['accuracy'], color='red', linestyle='--', alpha=0.7)
        plt.title(f'Accuracy (лучший порог: {best_thresholds["accuracy"]:.2f}, значение: {best_values["accuracy"]:.3f})')
        plt.xlabel('Порог')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(thresholds, results['precision'], 'g-', linewidth=2)
        plt.axvline(best_thresholds['precision'], color='red', linestyle='--', alpha=0.7)
        plt.title(f'Precision (лучший порог: {best_thresholds["precision"]:.2f}, значение: {best_values["precision"]:.3f})')
        plt.xlabel('Порог')
        plt.ylabel('Precision')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.plot(thresholds, results['recall'], 'orange', linewidth=2)
        plt.axvline(best_thresholds['recall'], color='red', linestyle='--', alpha=0.7)
        plt.title(f'Recall (лучший порог: {best_thresholds["recall"]:.2f}, значение: {best_values["recall"]:.3f})')
        plt.xlabel('Порог')
        plt.ylabel('Recall')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        plt.plot(thresholds, results['f1'], 'purple', linewidth=2)
        plt.axvline(best_thresholds['f1'], color='red', linestyle='--', alpha=0.7)
        plt.title(f'F1-Score (лучший порог: {best_thresholds["f1"]:.2f}, значение: {best_values["f1"]:.3f})')
        plt.xlabel('Порог')
        plt.ylabel('F1-Score')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, 'threshold_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 График анализа порогов сохранен: {plot_path}")
    
    analysis = {
        'threshold_analysis': results,
        'best_thresholds': best_thresholds,
        'best_values': best_values
    }
    
    return analysis


def train_meta_model(
    model: Qwen3CrossAttentionMetaModel,
    train_dataset: MetaModelDataset,
    test_dataset: MetaModelDataset,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    device: str = "auto",
    save_dir: str = "./checkpoints",
    eval_every_steps: int = 100,
    save_metrics_every_steps: int = 50,
    batch_size: int = 1,
    num_workers: int = 0,
    max_steps: int = None,
    wandb_config: dict = None
):
    """Обучение Meta-Model"""
    
    logger = setup_logging(os.path.join(save_dir, "logs"))
    
    # Инициализация wandb
    use_wandb = wandb_config is not None and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=wandb_config.get('project', 'llms4ol-meta-model'),
            name=wandb_config.get('name'),
            config=wandb_config.get('config', {}),
            tags=wandb_config.get('tags', []),
            notes=wandb_config.get('notes', ''),
            dir=save_dir
        )
        logger.info(f"🔗 Wandb инициализирован: {wandb.run.name}")
    elif wandb_config is not None and not WANDB_AVAILABLE:
        logger.warning("⚠️ Wandb запрошен, но не доступен")
    
    # Определяем устройство
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.train()
    
    # Создаем DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False
    )
    
    # Оптимизируем только обучаемые параметры
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
    
    # Планировщик с warmup
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * 0.1)  # 10% от общего количества шагов
    
    try:
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        logger.info(f"📈 Планировщик с warmup: {warmup_steps} шагов warmup из {total_steps} общих")
    except ImportError:
        # Fallback к стандартному scheduler если transformers не доступен
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        logger.info("⚠️ transformers не найден, используется стандартный CosineAnnealingLR")
        warmup_steps = 0
    
    # Создаем структуру папок
    results_dir = save_dir
    metrics_dir = os.path.join(save_dir, "metrics")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    logger.info(f"🚀 Начало обучения Meta-Model:")
    logger.info(f"   Эпохи: {num_epochs}")
    logger.info(f"   Шагов на эпоху: {len(train_loader)}")
    logger.info(f"   Всего шагов: {total_steps}")
    logger.info(f"   Устройство: {device}")
    logger.info(f"   Learning rate: {learning_rate}")
    logger.info(f"   Обучаемые параметры: {len(trainable_params):,}")
    
    # Логирование
    training_log = []
    step_counter = 0
    best_roc_auc = 0.0
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (texts_1, texts_2, gt_matrix, dataset_name) in enumerate(train_loader):
            # Правильно извлекаем данные из DataLoader (batch_size=1)
            # DataLoader возвращает: список из 32 списков, каждый содержит 1 строку
            # Нужно извлечь все строки из внутренних списков
            texts_1 = [item[0] for item in texts_1]  # Извлекаем строки из списков
            texts_2 = [item[0] for item in texts_2]  # Извлекаем строки из списков
            gt_matrix = gt_matrix.squeeze(0).to(device)  # Убираем размерность batch: [1, 32, 32] -> [32, 32]
            
            # Forward pass
            pred_matrix = model(texts_1, texts_2)
            
            # Loss
            loss = weighted_bce_loss(pred_matrix, gt_matrix)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            step_counter += 1
            
            # Записываем метрики
            basic_log_entry = {
                'step': step_counter,
                'epoch': epoch,
                'batch': batch_idx,
                'train_loss': loss.item(),
                'lr': scheduler.get_last_lr()[0],
                'dataset': dataset_name,
                'timestamp': datetime.now().isoformat()
            }
            training_log.append(basic_log_entry)
            
            # Логгирование в wandb
            if use_wandb:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/learning_rate': scheduler.get_last_lr()[0],
                    'train/epoch': epoch,
                    'train/step': step_counter
                }, step=step_counter)
            
            # Проверка max_steps
            if max_steps is not None and step_counter >= max_steps:
                logger.info(f"🛑 Достигнуто максимальное количество шагов: {max_steps}")
                break
            
            # Логирование
            if step_counter % save_metrics_every_steps == 0:
                log_message = (f"Шаг {step_counter:5d} | Эпоха {epoch+1:2d}/{num_epochs} | "
                              f"Train Loss: {loss.item():.4f} | "
                              f"LR: {scheduler.get_last_lr()[0]:.6f}")
                logger.info(log_message)
            
            # Полная оценка
            if step_counter % eval_every_steps == 0:
                logger.info(f"\n📊 Полная оценка на шаге {step_counter}...")
                
                # Быстрый test_loss
                model.eval()
                with torch.no_grad():
                    test_texts_1, test_texts_2, test_gt_matrix, test_dataset_name = next(iter(test_loader))
                    # Правильно извлекаем данные из DataLoader (batch_size=1)
                    test_texts_1 = [item[0] for item in test_texts_1]  # Извлекаем строки из списков
                    test_texts_2 = [item[0] for item in test_texts_2]  # Извлекаем строки из списков
                    test_gt_matrix = test_gt_matrix.squeeze(0).to(device)  # Убираем размерность batch: [1, 32, 32] -> [32, 32]
                    
                    test_pred_matrix = model(test_texts_1, test_texts_2)
                    test_loss_for_curves = weighted_bce_loss(test_pred_matrix, test_gt_matrix)
                
                # Вычисляем train ROC AUC на небольшой выборке
                train_roc_auc = evaluate_train_roc_auc(model, train_loader, device, max_batches=10)
                
                # Обновляем training_log с дополнительными метриками
                training_log[-1]['test_loss'] = test_loss_for_curves.item()
                training_log[-1]['train_roc_auc'] = train_roc_auc
                
                # Полная оценка метрик НА ТЕСТОВЫХ ДАННЫХ
                logger.info(f"🔍 Вычисление метрик на тестовых данных...")
                metrics = evaluate_meta_model(model, test_loader, device)
                
                # Добавляем test ROC AUC в лог
                training_log[-1]['test_roc_auc'] = metrics['roc_auc']
                
                # Логгирование в wandb
                if use_wandb:
                    wandb_metrics = {
                        'test/loss': test_loss_for_curves.item(),
                        'test/roc_auc': metrics['roc_auc'],
                        'train/roc_auc': train_roc_auc,
                    }
                    # Добавляем метрики для разных порогов
                    for threshold in [0.05, 0.15, 0.25, 0.5]:
                        wandb_metrics[f'test/acc_{threshold}'] = metrics[f'acc_{threshold}']
                        wandb_metrics[f'test/f1_{threshold}'] = metrics[f'f1_{threshold}']
                        wandb_metrics[f'test/precision_{threshold}'] = metrics[f'precision_{threshold}']
                        wandb_metrics[f'test/recall_{threshold}'] = metrics[f'recall_{threshold}']
                    
                    wandb.log(wandb_metrics, step=step_counter)
                
                logger.info(f"🎯 Метрики (ТЕСТ):")
                logger.info(f"   Test ROC AUC: {metrics['roc_auc']:.4f}")
                logger.info(f"   Train ROC AUC: {train_roc_auc:.4f}")
                for threshold in [0.05, 0.15, 0.25, 0.5]:
                    threshold_msg = (f"   Порог {threshold}: Acc={metrics[f'acc_{threshold}']:.3f}, "
                                   f"F1={metrics[f'f1_{threshold}']:.3f}")
                    logger.info(threshold_msg)
                
                # Сохранение метрик
                metrics['step'] = step_counter
                metrics['epoch'] = epoch
                metrics['train_roc_auc'] = train_roc_auc
                save_metrics(metrics, f"{metrics_dir}/metrics_step_{step_counter}.json")
                
                # Сохранение лучшей модели (удаляем предыдущую)
                current_roc_auc = metrics['roc_auc']
                if current_roc_auc > best_roc_auc:
                    # Удаляем предыдущую лучшую модель
                    old_best_dirs = [d for d in os.listdir(results_dir) if d.startswith("best_model_step_")]
                    for old_dir in old_best_dirs:
                        old_path = os.path.join(results_dir, old_dir)
                        if os.path.isdir(old_path):
                            import shutil
                            shutil.rmtree(old_path)
                            logger.info(f"🗑️ Удалена предыдущая модель: {old_dir}")
                    
                    best_roc_auc = current_roc_auc
                    best_model_dir = os.path.join(results_dir, f"best_model_step_{step_counter}_auc_{current_roc_auc:.4f}")
                    model.save_pretrained(best_model_dir)
                    
                    # Анализ порогов для лучшей модели
                    threshold_analysis = analyze_thresholds(model, test_loader, device, best_model_dir)
                    with open(os.path.join(best_model_dir, 'threshold_analysis.json'), 'w') as f:
                        json.dump(threshold_analysis, f, indent=2)
                    
                    # Устанавливаем оптимальный F1 threshold
                    model.f1_threshold = float(threshold_analysis['best_thresholds']['f1'])
                    
                    logger.info(f"💾 Новая лучшая модель сохранена (Test ROC AUC: {current_roc_auc:.4f})")
                    logger.info(f"🎯 Оптимальный F1 threshold: {model.f1_threshold:.4f}")
                
                # Создаем обновляемые графики
                plot_path = create_training_plots(training_log, results_dir)
                if plot_path:
                    logger.info(f"📈 Графики обучения обновлены: training_progress.png")
                
                model.train()
                logger.info("")
        
        # Проверка max_steps
        if max_steps is not None and step_counter >= max_steps:
            break
        
        # Статистика эпохи
        avg_epoch_loss = epoch_loss / len(train_loader)
        elapsed_time = time.time() - start_time
        
        logger.info(f"\n📈 Эпоха {epoch+1}/{num_epochs} завершена:")
        logger.info(f"   Средний loss: {avg_epoch_loss:.4f}")
        logger.info(f"   Время: {elapsed_time/60:.1f} мин")
        logger.info(f"   Лучший ROC AUC: {best_roc_auc:.4f}")
        logger.info("-" * 60)
    
    # Сохранение лога обучения
    with open(f"{metrics_dir}/training_log.json", 'w') as f:
        json.dump(training_log, f, indent=2)
    
    # Финальная оценка
    logger.info(f"\n🏁 Финальная оценка на тестовых данных...")
    final_metrics = evaluate_meta_model(model, test_loader, device)
    
    # Добавляем финальный train ROC AUC
    final_train_roc_auc = evaluate_train_roc_auc(model, train_loader, device, max_batches=20)
    final_metrics['train_roc_auc'] = final_train_roc_auc
    
    save_metrics(final_metrics, f"{metrics_dir}/final_metrics.json")
    
    # Финальный анализ порогов
    logger.info(f"\n🔍 Финальный анализ порогов на тестовых данных...")
    threshold_analysis = analyze_thresholds(model, test_loader, device, results_dir)
    save_metrics(threshold_analysis, f"{metrics_dir}/threshold_analysis.json")
    
    # Создаем финальные графики
    final_plot_path = create_training_plots(training_log, results_dir)
    
    logger.info(f"📊 Лучшие пороги (на тестовых данных):")
    for metric, threshold in threshold_analysis['best_thresholds'].items():
        value = threshold_analysis['best_values'][metric]
        logger.info(f"   {metric.capitalize()}: {threshold:.2f} (значение: {value:.4f})")
    
    # Финальное логгирование в wandb
    if use_wandb:
        final_wandb_metrics = {
            'final/test_roc_auc': final_metrics['roc_auc'],
            'final/train_roc_auc': final_train_roc_auc,
            'final/best_test_roc_auc': best_roc_auc,
            'final/training_time_minutes': (time.time() - start_time) / 60,
            'final/total_steps': step_counter
        }
        
        # Добавляем лучшие пороги
        for metric, threshold in threshold_analysis['best_thresholds'].items():
            final_wandb_metrics[f'best_threshold/{metric}'] = threshold
            final_wandb_metrics[f'best_value/{metric}'] = threshold_analysis['best_values'][metric]
        
        wandb.log(final_wandb_metrics, step=step_counter)
        
        # Сохраняем графики в wandb
        if final_plot_path and os.path.exists(final_plot_path):
            wandb.log({"charts/training_progress": wandb.Image(final_plot_path)})
        
        threshold_plot_path = os.path.join(results_dir, 'threshold_analysis.png')
        if os.path.exists(threshold_plot_path):
            wandb.log({"charts/threshold_analysis": wandb.Image(threshold_plot_path)})
        
        wandb.finish()
        logger.info(f"🔗 Wandb run завершен: {wandb.run.url if wandb.run else 'N/A'}")
    
    logger.info(f"✅ Обучение Meta-Model завершено!")
    logger.info(f"   Финальный Test ROC AUC: {final_metrics['roc_auc']:.4f}")
    logger.info(f"   Финальный Train ROC AUC: {final_train_roc_auc:.4f}")
    logger.info(f"   Лучший Test ROC AUC: {best_roc_auc:.4f}")
    logger.info(f"   Время обучения: {(time.time() - start_time)/60:.1f} мин")
    if final_plot_path:
        logger.info(f"   📈 Графики обучения: training_progress.png")
    
    return model, training_log, final_metrics, threshold_analysis


def save_metrics(metrics: Dict[str, Any], filepath: str):
    """Сохранение метрик в JSON файл"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)


def create_experiment_name(args) -> str:
    """Создание уникального имени эксперимента"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    parts = [
        args.dataset_name,
        f"ep{args.epochs}",
        f"lr{args.lr:.0e}",
        f"bs{args.batch_size}",
        f"eval{args.eval_every}",
        f"seed{args.seed}",
        f"freeze_{args.freeze_strategy}"
    ]
    
    if args.qwen3_model != "Qwen/Qwen3-Embedding-0.6B":
        model_short = args.qwen3_model.split('/')[-1].lower().replace('-', '_')
        parts.append(model_short)
    
    if args.embedding_pooling != "last_token_norm":
        parts.append(f"pool_{args.embedding_pooling}")
    
    if args.max_steps:
        parts.append(f"max{args.max_steps}")
    
    if args.freeze_strategy == "lora":
        parts.append(f"r{args.lora_rank}_a{args.lora_alpha}")
    
    experiment_name = f"{timestamp}_{'_'.join(parts)}"
    return experiment_name


def show_usage_examples():
    """Показ примеров использования"""
    examples = """
🚀 ОБУЧЕНИЕ META-MODEL - ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ:

1. Полная заморозка Qwen3 (быстрое обучение):
   python train_meta_model.py \\
       --terms_path data/doid_train_types.txt \\
       --relations_path data/doid_train_pairs.json \\
       --output_dir experiments/freeze_full \\
       --dataset_name DOID \\
       --freeze_strategy full \\
       --epochs 10 \\
       --lr 1e-4

2. Заморозка кроме последнего слоя (средняя адаптация):
   python train_meta_model.py \\
       --terms_path data/doid_train_types.txt \\
       --relations_path data/doid_train_pairs.json \\
       --output_dir experiments/freeze_except_last \\
       --dataset_name DOID \\
       --freeze_strategy except_last \\
       --epochs 15 \\
       --lr 5e-5

3. LORA адаптер (эффективная адаптация):
   python train_meta_model.py \\
       --terms_path data/doid_train_types.txt \\
       --relations_path data/doid_train_pairs.json \\
       --output_dir experiments/lora_training \\
       --dataset_name DOID \\
       --freeze_strategy lora \\
       --lora_rank 16 \\
       --lora_alpha 32 \\
       --epochs 20 \\
       --lr 1e-4

4. Без заморозки (полное дообучение):
   python train_meta_model.py \\
       --terms_path data/doid_train_types.txt \\
       --relations_path data/doid_train_pairs.json \\
       --output_dir experiments/full_training \\
       --dataset_name DOID \\
       --freeze_strategy none \\
       --epochs 5 \\
       --lr 1e-5

5. Быстрое тестирование:
   python train_meta_model.py \\
       --terms_path data/small_test.txt \\
       --relations_path data/small_test.json \\
       --output_dir experiments/quick_test \\
       --dataset_name TEST \\
       --freeze_strategy full \\
       --epochs 2 \\
       --max_steps 50 \\
       --eval_every 10

6. Обучение с wandb логгированием:
   python train_meta_model.py \\
       --terms_path data/doid_train_types.txt \\
       --relations_path data/doid_train_pairs.json \\
       --output_dir experiments/wandb_training \\
       --dataset_name DOID \\
       --freeze_strategy lora \\
       --epochs 15 \\
       --wandb \\
       --wandb_project "doid-meta-model" \\
       --wandb_name "lora_r16_a32_experiment" \\
       --wandb_tags "lora" "doid" "meta-model" \\
       --wandb_notes "Эксперимент с LORA адаптером"

📝 КЛЮЧЕВЫЕ ПАРАМЕТРЫ:
   --freeze_strategy: "full", "except_last", "lora", "none"
   --lora_rank: Rank для LORA адаптера (4-64, обычно 8-16)
   --lora_alpha: Alpha для LORA (обычно 16-32)
   --embedding_pooling: "mean", "last_token", "last_token_norm"
   
   WANDB ПАРАМЕТРЫ:
   --wandb: Включить wandb логгирование
   --wandb_project: Название проекта в wandb
   --wandb_name: Название эксперимента (авто если не указано)
   --wandb_tags: Теги для эксперимента
   --wandb_notes: Описание эксперимента

💡 РЕКОМЕНДАЦИИ:
   • full: Самый быстрый, для похожих доменов
   • except_last: Хороший баланс скорости и качества
   • lora: Эффективно для адаптации под новые домены
   • none: Самое медленное, но лучшее качество
   • Начинайте с except_last для большинства задач
   • Используйте wandb для отслеживания экспериментов
    """
    print(examples)


if __name__ == "__main__":
    # Показываем примеры если запустили без аргументов
    if len(sys.argv) == 1:
        print("🚀 Обучение Meta-Model с расширенными опциями заморозки")
        print("=" * 70)
        show_usage_examples()
        print("\n📖 Для получения полной справки:")
        print("   python train_meta_model.py --help")
        sys.exit(0)
    
    parser = argparse.ArgumentParser(
        description="Meta-Model Training with Advanced Freezing Options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="🔥 Упрощенный скрипт только для Meta-Model с txt файлами!"
    )
    
    # Основные параметры
    parser.add_argument("--terms_path", type=str, required=True, help="Path to terms TXT file")
    parser.add_argument("--relations_path", type=str, required=True, help="Path to relations JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    
    # Параметры обучения
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--eval_every", type=int, default=100, help="Evaluate every N steps")
    parser.add_argument("--save_every", type=int, default=50, help="Save every N steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split size")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum training steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Параметры модели
    parser.add_argument("--qwen3_model", type=str, default="Qwen/Qwen3-Embedding-0.6B", help="Qwen3 model name")
    parser.add_argument("--embedding_pooling", type=str, default="last_token_norm", 
                       choices=["mean", "last_token", "last_token_norm"], help="Embedding pooling method")
    
    # Параметры заморозки (НОВОЕ!)
    parser.add_argument("--freeze_strategy", type=str, default="full", 
                       choices=["full", "except_last", "lora", "none"],
                       help="Freezing strategy for Qwen3 weights")
    parser.add_argument("--lora_rank", type=int, default=8, help="LORA rank (for lora strategy)")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LORA alpha (for lora strategy)")
    
    # Параметры датасета
    parser.add_argument("--dataset_strategy", type=str, default="single", 
                       choices=["single", "weighted"], help="Dataset selection strategy")
    parser.add_argument("--sampling_strategy", type=str, default="balanced", 
                       choices=["random", "balanced"], help="Sampling strategy")
    parser.add_argument("--positive_ratio", type=float, default=1.0, help="Positive pairs ratio")
    
    # Прочие параметры
    parser.add_argument("--log_level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    
    # Wandb параметры
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="llms4ol-meta-model", help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name (auto if None)")
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=[], help="Wandb tags")
    parser.add_argument("--wandb_notes", type=str, default="", help="Wandb run notes")
    
    args = parser.parse_args()
    
    # Создаем уникальную папку для эксперимента
    experiment_name = create_experiment_name(args)
    experiment_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Настройка логирования
    logger = setup_logging(experiment_dir, args.log_level)
    
    logger.info(f"🚀 Начало эксперимента: {experiment_name}")
    logger.info(f"📁 Папка эксперимента: {experiment_dir}")
    logger.info(f"🎲 Seed: {args.seed}")
    logger.info(f"🧊 Стратегия заморозки: {args.freeze_strategy}")
    
    # Создаем датасеты
    logger.info(f"🔄 Создание датасетов...")
    logger.info(f"   Terms: {args.terms_path}")
    logger.info(f"   Relations: {args.relations_path}")
    
    train_dataset = MetaModelDataset(
        terms_path=args.terms_path,
        relations_path=args.relations_path,
        batch_size_1=args.batch_size,
        batch_size_2=args.batch_size,
        dataset_strategy=args.dataset_strategy,
        sampling_strategy=args.sampling_strategy,
        positive_ratio=args.positive_ratio,
        mode="train",
        test_part=args.test_size,
        random_state=args.seed
    )
    
    test_dataset = MetaModelDataset(
        terms_path=args.terms_path,
        relations_path=args.relations_path,
        batch_size_1=args.batch_size,
        batch_size_2=args.batch_size,
        dataset_strategy=args.dataset_strategy,
        sampling_strategy=args.sampling_strategy,
        positive_ratio=args.positive_ratio,
        mode="test",
        test_part=args.test_size,
        random_state=args.seed
    )
    
    logger.info(f"✅ Датасеты созданы: Train={len(train_dataset)}, Test={len(test_dataset)}")
    
    # Создаем мета-модель
    logger.info(f"🔄 Создание Meta-Model: {args.qwen3_model}")
    
    model = create_meta_model_from_scratch(
        qwen3_model_name=args.qwen3_model,
        embedding_pooling=args.embedding_pooling,
        init_from_qwen3=True,  # Всегда инициализируем из Qwen3
        device="auto"
    )
    
    logger.info(f"✅ Модель создана: {model}")
    
    # Применяем стратегию заморозки
    apply_freeze_strategy(model, args.freeze_strategy, args.lora_rank, args.lora_alpha)
    
    # Настройка wandb
    wandb_config = None
    if args.wandb and WANDB_AVAILABLE:
        wandb_config = {
            'project': args.wandb_project,
            'name': args.wandb_name or experiment_name,
            'tags': args.wandb_tags + [args.dataset_name, args.freeze_strategy],
            'notes': args.wandb_notes,
            'config': {
                'experiment_name': experiment_name,
                'dataset_name': args.dataset_name,
                'epochs': args.epochs,
                'learning_rate': args.lr,
                'batch_size': args.batch_size,
                'freeze_strategy': args.freeze_strategy,
                'lora_rank': args.lora_rank if args.freeze_strategy == "lora" else None,
                'lora_alpha': args.lora_alpha if args.freeze_strategy == "lora" else None,
                'qwen3_model': args.qwen3_model,
                'embedding_pooling': args.embedding_pooling,
                'test_size': args.test_size,
                'max_steps': args.max_steps,
                'seed': args.seed,
                'eval_every_steps': args.eval_every,
                'dataset_strategy': args.dataset_strategy,
                'sampling_strategy': args.sampling_strategy,
                'positive_ratio': args.positive_ratio
            }
        }
    
    # Запускаем обучение
    logger.info(f"\n🚀 Начинаем обучение...")
    if wandb_config:
        logger.info(f"🔗 Wandb проект: {wandb_config['project']}")
        logger.info(f"🔗 Wandb run: {wandb_config['name']}")
    
    trained_model, training_log, final_metrics, threshold_analysis = train_meta_model(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        save_dir=experiment_dir,
        eval_every_steps=args.eval_every,
        save_metrics_every_steps=args.save_every,
        batch_size=1,  # DataLoader batch size
        num_workers=0,
        max_steps=args.max_steps,
        wandb_config=wandb_config
    )
    
    logger.info(f"\n✅ Обучение завершено! Результаты сохранены в: {experiment_dir}")
    logger.info(f"   📊 График анализа порогов: threshold_analysis.png")
    logger.info(f"   📂 Детальные метрики: metrics/")
    logger.info(f"   🏆 Лучшая модель: best_model_step_*/") 