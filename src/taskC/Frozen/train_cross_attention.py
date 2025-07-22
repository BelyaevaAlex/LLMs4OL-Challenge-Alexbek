"""
Cross-Attention model initialization and training system
Includes initialization from Qwen3, weighted loss, metrics and saving best models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoConfig
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import time
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Для серверного окружения без GUI

from cross_attention_model import CrossAttentionModel
from dataset import CrossAttentionDataset, create_train_test_datasets


def create_experiment_name(args) -> str:
    """
    Создание уникального имени эксперимента на основе аргументов и времени
    
    Args:
        args: аргументы командной строки
        
    Returns:
        experiment_name: уникальное имя эксперимента
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Основные параметры для имени
    parts = [
        args.dataset_name,
        f"ep{args.epochs}",
        f"lr{args.lr:.0e}",
        f"bs{args.batch_size}",
        f"eval{args.eval_every}",
        f"seed{getattr(args, 'seed', 42)}"
    ]
    
    # Добавляем стратегии если они не по умолчанию
    if args.dataset_strategy != "single":
        parts.append(f"ds_{args.dataset_strategy}")
    
    if args.sampling_strategy != "balanced":
        parts.append(f"samp_{args.sampling_strategy}")
    
    if args.positive_ratio != 1.0:
        parts.append(f"pos{args.positive_ratio}")
        
    if args.use_qwen3:
        parts.append("qwen3")
    
    if args.max_steps:
        parts.append(f"max{args.max_steps}")
    
    experiment_name = f"{timestamp}_{'_'.join(parts)}"
    
    return experiment_name


def create_best_results_summary(
    threshold_analysis: Dict[str, Any],
    final_metrics: Dict[str, float],
    training_log: List[Dict],
    args,
    save_path: str
):
    """
    Создание итогового JSON с лучшими найденными параметрами
    
    Args:
        threshold_analysis: результаты анализа порогов
        final_metrics: финальные метрики
        training_log: лог обучения
        args: аргументы обучения
        save_path: путь для сохранения
    """
    
    # Лучшие пороги и их значения
    best_thresholds = threshold_analysis['best_thresholds']
    best_values = threshold_analysis['best_values']
    
    # Информация об обучении
    training_info = {
        'total_steps': len(training_log),
        'final_train_loss': training_log[-1]['train_loss'] if training_log else None,
        'final_test_loss': training_log[-1].get('test_loss', None) if training_log else None,
        'best_train_loss': min(entry['train_loss'] for entry in training_log) if training_log else None,
        'best_test_loss': min(entry.get('test_loss', float('inf')) for entry in training_log 
                             if 'test_loss' in entry) if training_log else None,
    }
    
    # Создаем сводку
    summary = {
        'experiment_info': {
            'dataset': args.dataset_name,
            'timestamp': datetime.now().isoformat(),
            'experiment_args': {
                'epochs': args.epochs,
                'learning_rate': args.lr,
                'batch_size': args.batch_size,
                'eval_every': args.eval_every,
                'seed': getattr(args, 'seed', 42),
                'dataset_strategy': args.dataset_strategy,
                'sampling_strategy': args.sampling_strategy,
                'positive_ratio': args.positive_ratio,
                'use_qwen3': args.use_qwen3,
                'max_steps': args.max_steps,
                'test_size': args.test_size
            }
        },
        'best_results': {
            'roc_auc': final_metrics.get('roc_auc', 0.0),
            'best_thresholds': {
                'accuracy': {
                    'threshold': float(best_thresholds['accuracy']),
                    'value': float(best_values['accuracy'])
                },
                'precision': {
                    'threshold': float(best_thresholds['precision']),
                    'value': float(best_values['precision'])
                },
                'recall': {
                    'threshold': float(best_thresholds['recall']),
                    'value': float(best_values['recall'])
                },
                'f1_score': {
                    'threshold': float(best_thresholds['f1']),
                    'value': float(best_values['f1'])
                }
            }
        },
        'training_summary': training_info,
        'files_generated': {
            'training_curves': 'training_curves.png',
            'metrics_curves': 'metrics_curves.png',
            'threshold_analysis': 'threshold_analysis.png',
            'detailed_metrics': 'metrics/',
            'training_log': 'metrics/training_log.json',
            'threshold_analysis_detailed': 'metrics/threshold_analysis.json',
            'best_models': 'best_models/',
            'best_model_threshold_analysis': 'best_models/best_model_threshold_analysis.png',
            'best_model_threshold_analysis_detailed': 'best_models/best_model_threshold_analysis.json',
            'best_model_evaluation_summary': 'best_models/best_model_evaluation_summary.json'
        }
    }
    
    # Сохраняем сводку
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 Сводка лучших результатов сохранена: {save_path}")
    print(f"🎯 Лучшие пороги:")
    for metric, info in summary['best_results']['best_thresholds'].items():
        print(f"   {metric.capitalize()}: {info['threshold']:.2f} (значение: {info['value']:.4f})")


def setup_logging(log_dir: str, log_level: str = "INFO") -> logging.Logger:
    """
    Настройка логирования в файл и консоль
    
    Args:
        log_dir: директория для логов
        log_level: уровень логирования
        
    Returns:
        logger: настроенный логгер
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Создаем логгер
    logger = logging.getLogger("cross_attention_training")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Очищаем существующие handlers
    logger.handlers.clear()
    
    # Форматтер для логов
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler для файла
    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Handler для консоли
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Добавляем handlers к логгеру
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Логирование настроено. Файл лога: {log_file}")
    
    return logger


def initialize_from_qwen3(
    model_name: str, 
    layer_idx: int = -1,
    device: str = "auto"
) -> CrossAttentionModel:
    """
    Инициализация CrossAttentionModel весами из Qwen3
    
    Args:
        model_name: имя модели Qwen3 (например, "Qwen/Qwen3-4B")
        layer_idx: индекс слоя для инициализации (-1 для последнего)
        device: устройство для загрузки модели
        
    Returns:
        model: инициализированная CrossAttentionModel
    """
    print(f"🔄 Загрузка Qwen3 модели: {model_name}")
    
    # Определяем устройство
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Загрузка Qwen3 модели
    try:
        qwen3_model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device if device == "cuda" else None,
            trust_remote_code=True
        )
        config = qwen3_model.config
        print(f"✅ Qwen3 модель загружена на {device}")
    except Exception as e:
        print(f"❌ Ошибка загрузки Qwen3: {e}")
        raise
    
    # Выбираем слой для инициализации
    if layer_idx == -1:
        layer_idx = len(qwen3_model.layers) - 1
        
    if layer_idx >= len(qwen3_model.layers):
        raise ValueError(f"layer_idx {layer_idx} >= количества слоев {len(qwen3_model.layers)}")
    
    layer = qwen3_model.layers[layer_idx]
    attention = layer.self_attn
    
    print(f"🎯 Используем слой {layer_idx} для инициализации")
    
    # Создание CrossAttentionModel с теми же параметрами
    cross_model = CrossAttentionModel(config, layer_idx=layer_idx)
    
    # Перенос весов
    print(f"📋 Копирование весов...")
    try:
        # Query projection
        cross_model.q_proj.weight.data = attention.q_proj.weight.data.clone().to(torch.float32)
        if cross_model.q_proj.bias is not None and attention.q_proj.bias is not None:
            cross_model.q_proj.bias.data = attention.q_proj.bias.data.clone().to(torch.float32)
        
        # Key projection
        cross_model.k_proj.weight.data = attention.k_proj.weight.data.clone().to(torch.float32)
        if cross_model.k_proj.bias is not None and attention.k_proj.bias is not None:
            cross_model.k_proj.bias.data = attention.k_proj.bias.data.clone().to(torch.float32)
        
        # Нормализация
        cross_model.q_norm.weight.data = attention.q_norm.weight.data.clone().to(torch.float32)
        cross_model.k_norm.weight.data = attention.k_norm.weight.data.clone().to(torch.float32)
        
        cross_model.initialized_from_qwen3 = True
        print(f"✅ Веса успешно скопированы из слоя {layer_idx}")
        
    except Exception as e:
        print(f"❌ Ошибка копирования весов: {e}")
        raise
    
    # Освобождаем память от Qwen3
    del qwen3_model
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # Перемещаем модель на устройство
    cross_model = cross_model.to(device)
    
    print(f"🚀 CrossAttentionModel готова к обучению!")
    print(f"   Параметры: {sum(p.numel() for p in cross_model.parameters() if p.requires_grad):,}")
    print(f"   Устройство: {device}")
    
    return cross_model


def weighted_bce_loss(pred_matrix: torch.Tensor, gt_matrix: torch.Tensor) -> torch.Tensor:
    """
    Взвешенный Binary Cross Entropy Loss
    loss = loss[ones_places]/len(ones_places) + loss[zeros_places]/len(ones_places)
    
    Args:
        pred_matrix: (n, m) предсказанная матрица внимания
        gt_matrix: (n, m) ground truth матрица отношений
        
    Returns:
        loss: взвешенный лосс
    """
    # Разделение на единицы и нули
    ones_mask = (gt_matrix == 1)
    zeros_mask = (gt_matrix == 0)
    
    ones_count = ones_mask.sum().item()
    zeros_count = zeros_mask.sum().item()
    
    if ones_count == 0:
        # Если нет положительных примеров, используем обычный BCE
        logger = logging.getLogger("cross_attention_training")
        logger.warning("No positive examples found, using regular BCE")
        return F.binary_cross_entropy(pred_matrix, gt_matrix)
    
    # Вычисление лосса отдельно для единиц и нулей
    loss_ones = 0.0
    if ones_mask.sum() > 0:
        loss_ones = F.binary_cross_entropy(pred_matrix[ones_mask], gt_matrix[ones_mask])
    
    loss_zeros = 0.0
    if zeros_mask.sum() > 0:
        loss_zeros = F.binary_cross_entropy(pred_matrix[zeros_mask], gt_matrix[zeros_mask])
    
    # Взвешенное усреднение
    total_loss = loss_ones / ones_count + loss_zeros / zeros_count
    
    return total_loss


class BestModelSaver:
    """Класс для сохранения лучших моделей по ROC AUC"""
    
    def __init__(self, save_dir: str, keep_top_k: int = 2, test_loader: DataLoader = None, device: str = "cuda"):
        self.save_dir = save_dir
        self.keep_top_k = keep_top_k
        self.best_scores = []  # [(roc_auc, step, model_path), ...]
        self.test_loader = test_loader
        self.device = device
        os.makedirs(save_dir, exist_ok=True)
        
    def save_if_best(self, model: CrossAttentionModel, roc_auc: float, step: int, 
                     metrics: Dict[str, float] = None):
        """Сохранить модель если она в топе по ROC AUC"""
        
        model_filename = f"model_step_{step}_auc_{roc_auc:.4f}.pt"
        model_path = os.path.join(self.save_dir, model_filename)
        
        # Сохранить модель
        # Создаем config на основе фактических размеров весов модели
        state_dict = model.state_dict()
        
        # Определяем размеры из фактических весов
        q_proj_weight = state_dict['q_proj.weight']
        k_proj_weight = state_dict['k_proj.weight']
        q_norm_weight = state_dict['q_norm.weight']
        
        # Вычисляем параметры архитектуры из размеров весов
        num_attention_heads_actual = q_proj_weight.shape[0] // q_norm_weight.shape[0]
        num_key_value_heads_actual = k_proj_weight.shape[0] // q_norm_weight.shape[0]
        head_dim_actual = q_norm_weight.shape[0]
        hidden_size_actual = q_proj_weight.shape[1]
        
        # Создаем правильный config на основе фактических размеров
        config_dict = {
            'hidden_size': hidden_size_actual,
            'num_attention_heads': num_attention_heads_actual,
            'num_key_value_heads': num_key_value_heads_actual,
            'head_dim': head_dim_actual,
            'rms_norm_eps': getattr(model.config, 'rms_norm_eps', 1e-6) if hasattr(model.config, 'rms_norm_eps') else model.config.get('rms_norm_eps', 1e-6),
            'attention_bias': getattr(model.config, 'attention_bias', False) if hasattr(model.config, 'attention_bias') else model.config.get('attention_bias', False)
        }
        
        checkpoint = {
            'model_state_dict': state_dict,
            'config': config_dict,
            'roc_auc': float(roc_auc),  # Убеждаемся что это Python float
            'step': int(step),          # Убеждаемся что это Python int
            'layer_idx': int(model.layer_idx),
            'initialized_from_qwen3': bool(model.initialized_from_qwen3),
            'metrics': {k: float(v) if isinstance(v, (int, float)) else v 
                       for k, v in (metrics or {}).items()},
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, model_path)
        
        # Обновить список лучших
        self.best_scores.append((roc_auc, step, model_path))
        self.best_scores.sort(reverse=True)  # По убыванию ROC AUC
        
        # Удалить лишние модели
        while len(self.best_scores) > self.keep_top_k:
            _, _, old_path = self.best_scores.pop()
            if os.path.exists(old_path):
                os.remove(old_path)
                print(f"🗑️ Удалена старая модель: {os.path.basename(old_path)}")
        
        print(f"💾 Сохранена модель: {model_filename} (ROC AUC: {roc_auc:.4f})")
        
        # Выполняем анализ порогов для лучшей модели
        if self.test_loader is not None:
            print(f"🔍 Выполняем анализ порогов для лучшей модели...")
            threshold_analysis = analyze_thresholds(
                model, 
                self.test_loader, 
                self.device,
                save_dir=self.save_dir  # Сохраняем график в папку best_models
            )
            
            # Сохраняем результаты анализа порогов в папку best_models
            with open(os.path.join(self.save_dir, 'best_model_threshold_analysis.json'), 'w') as f:
                json.dump(threshold_analysis, f, indent=2)
            
            # Создаем итоговую сводку для лучшей модели
            best_model_summary = self._create_best_model_summary(
                threshold_analysis, 
                metrics or {}, 
                roc_auc, 
                step
            )
            
            # Сохраняем сводку лучшей модели
            with open(os.path.join(self.save_dir, 'best_model_evaluation_summary.json'), 'w') as f:
                json.dump(best_model_summary, f, indent=2)
            
            print(f"📊 Анализ порогов для лучшей модели сохранен в: {self.save_dir}")
            print(f"📋 Сводка лучшей модели сохранена: best_model_evaluation_summary.json")
            print(f"🎯 Лучшие пороги для лучшей модели:")
            for metric, threshold in threshold_analysis['best_thresholds'].items():
                value = threshold_analysis['best_values'][metric]
                print(f"   {metric.capitalize()}: {threshold:.2f} (значение: {value:.4f})")
        
        # Сохраняем информацию о лучших моделях
        best_models_info = {
            'best_models': [
                {
                    'roc_auc': score,
                    'step': step,
                    'model_path': path,
                    'model_filename': os.path.basename(path)
                }
                for score, step, path in self.best_scores
            ]
        }
        
        with open(os.path.join(self.save_dir, 'best_models.json'), 'w') as f:
            json.dump(best_models_info, f, indent=2)
    
    def _create_best_model_summary(
        self, 
        threshold_analysis: Dict[str, Any], 
        metrics: Dict[str, float], 
        roc_auc: float, 
        step: int
    ) -> Dict[str, Any]:
        """
        Создание итоговой сводки для лучшей модели
        
        Args:
            threshold_analysis: результаты анализа порогов
            metrics: метрики модели
            roc_auc: ROC AUC модели
            step: шаг обучения
            
        Returns:
            summary: сводка лучшей модели
        """
        # Лучшие пороги и их значения
        best_thresholds = threshold_analysis['best_thresholds']
        best_values = threshold_analysis['best_values']
        
        # Создаем сводку для лучшей модели
        summary = {
            'best_model_info': {
                'step': int(step),
                'roc_auc': float(roc_auc),
                'timestamp': datetime.now().isoformat(),
                'model_filename': f"model_step_{step}_auc_{roc_auc:.4f}.pt"
            },
            'best_results': {
                'roc_auc': float(roc_auc),
                'best_thresholds': {
                    'accuracy': {
                        'threshold': float(best_thresholds['accuracy']),
                        'value': float(best_values['accuracy'])
                    },
                    'precision': {
                        'threshold': float(best_thresholds['precision']),
                        'value': float(best_values['precision'])
                    },
                    'recall': {
                        'threshold': float(best_thresholds['recall']),
                        'value': float(best_values['recall'])
                    },
                    'f1_score': {
                        'threshold': float(best_thresholds['f1']),
                        'value': float(best_values['f1'])
                    }
                }
            },
            'evaluation_metrics': {
                'fixed_thresholds': {
                    threshold: {
                        'accuracy': float(metrics.get(f'acc_{threshold}', 0.0)),
                        'precision': float(metrics.get(f'precision_{threshold}', 0.0)),
                        'recall': float(metrics.get(f'recall_{threshold}', 0.0)),
                        'f1_score': float(metrics.get(f'f1_{threshold}', 0.0))
                    }
                    for threshold in [0.05, 0.15, 0.25, 0.5]
                    if f'acc_{threshold}' in metrics
                }
            },
            'files_generated': {
                'model_checkpoint': f"model_step_{step}_auc_{roc_auc:.4f}.pt",
                'threshold_analysis_plot': 'best_model_threshold_analysis.png',
                'threshold_analysis_data': 'best_model_threshold_analysis.json',
                'models_list': 'best_models.json'
            }
        }
        
        return summary


def evaluate_model(
    model: CrossAttentionModel, 
    dataloader: DataLoader, 
    device: str
) -> Dict[str, float]:
    """
    Оценка модели на тестовых данных
    
    Args:
        model: модель для оценки
        dataloader: DataLoader с тестовыми данными
        device: устройство для вычислений
        
    Returns:
        metrics: словарь с метриками
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for vectors_1, vectors_2, gt_matrix, dataset_name in dataloader:
            # Убираем batch размерность от DataLoader
            vectors_1 = vectors_1.squeeze(0).to(device)
            vectors_2 = vectors_2.squeeze(0).to(device)
            gt_matrix = gt_matrix.squeeze(0).to(device)
            
            # Предсказание
            pred_matrix = model(vectors_1, vectors_2)
            
            # Собираем все предсказания и target'ы
            all_predictions.extend(pred_matrix.flatten().cpu().numpy())
            all_targets.extend(gt_matrix.flatten().cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Вычисление метрик
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


def save_metrics(metrics: Dict[str, Any], filepath: str):
    """Сохранение метрик в JSON файл"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)


def analyze_thresholds(
    model: CrossAttentionModel, 
    dataloader: DataLoader, 
    device: str,
    save_dir: str = None
) -> Dict[str, Any]:
    """
    Анализ модели с различными порогами от 0.01 до 0.60
    
    Args:
        model: модель для анализа
        dataloader: DataLoader с тестовыми данными
        device: устройство для вычислений
        save_dir: директория для сохранения графиков
        
    Returns:
        analysis: результаты анализа с лучшими порогами
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    # Собираем все предсказания
    with torch.no_grad():
        for vectors_1, vectors_2, gt_matrix, dataset_name in dataloader:
            vectors_1 = vectors_1.squeeze(0).to(device)
            vectors_2 = vectors_2.squeeze(0).to(device)
            gt_matrix = gt_matrix.squeeze(0).to(device)
            
            pred_matrix = model(vectors_1, vectors_2)
            
            all_predictions.extend(pred_matrix.flatten().cpu().numpy())
            all_targets.extend(gt_matrix.flatten().cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Анализируем пороги от 0.01 до 0.60 с шагом 0.01
    thresholds = np.arange(0.01, 0.61, 0.01)
    
    results = {
        'thresholds': thresholds.tolist(),
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    # Вычисляем метрики для каждого порога
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
    
    # Находим лучшие пороги для каждой метрики
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
        # Определяем имя файла в зависимости от контекста
        if 'best_models' in save_dir:
            plot_filename = 'best_model_threshold_analysis.png'
        else:
            plot_filename = 'threshold_analysis.png'
        
        plot_path = os.path.join(save_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 График анализа порогов сохранен: {plot_path}")
    
    # Формируем итоговый результат
    analysis = {
        'threshold_analysis': results,
        'best_thresholds': best_thresholds,
        'best_values': best_values,
        'summary': {
            f'best_accuracy_threshold_{best_thresholds["accuracy"]:.2f}': best_values['accuracy'],
            f'best_precision_threshold_{best_thresholds["precision"]:.2f}': best_values['precision'],
            f'best_recall_threshold_{best_thresholds["recall"]:.2f}': best_values['recall'],
            f'best_f1_threshold_{best_thresholds["f1"]:.2f}': best_values['f1'],
        }
    }
    
    return analysis


def plot_training_curves(training_log: List[Dict], save_dir: str):
    """
    Построение графиков обучения из лога
    
    Args:
        training_log: список записей логов обучения
        save_dir: директория для сохранения графиков
    """
    if not training_log:
        print("⚠️ Нет данных для построения графиков")
        return
    
    # Сортируем лог по шагам для правильного порядка точек
    training_log_sorted = sorted(training_log, key=lambda x: x.get('step', 0))
    
    steps = [entry['step'] for entry in training_log_sorted]
    train_losses = [entry['train_loss'] for entry in training_log_sorted]
    
    # Для test_loss берем только те записи, где он есть
    test_steps = [entry['step'] for entry in training_log_sorted if 'test_loss' in entry]
    test_losses = [entry['test_loss'] for entry in training_log_sorted if 'test_loss' in entry]
    
    plt.figure(figsize=(15, 10))
    
    # График лоссов
    plt.subplot(2, 3, 1)
    plt.plot(steps, train_losses, 'b-', label='Train Loss', linewidth=2)
    if test_losses:
        plt.plot(test_steps, test_losses, 'r-', label='Test Loss', linewidth=2, marker='o', markersize=3)
    plt.title('Training and Test Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График learning rate
    if 'lr' in training_log_sorted[0]:
        lrs = [entry['lr'] for entry in training_log_sorted]
        plt.subplot(2, 3, 2)
        plt.plot(steps, lrs, 'g-', linewidth=2)
        plt.title('Learning Rate')
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.grid(True, alpha=0.3)
    
    # График разности лоссов (только для точек, где есть test_loss)
    if test_losses:
        plt.subplot(2, 3, 3)
        # Интерполируем train_loss для соответствующих test_steps
        test_train_losses = []
        for test_step in test_steps:
            # Находим ближайший train_loss для этого шага
            closest_idx = min(range(len(steps)), key=lambda i: abs(steps[i] - test_step))
            test_train_losses.append(train_losses[closest_idx])
        
        loss_diff = [abs(train - test) for train, test in zip(test_train_losses, test_losses)]
        plt.plot(test_steps, loss_diff, 'purple', linewidth=2, marker='o', markersize=3)
        plt.title('|Train Loss - Test Loss|')
        plt.xlabel('Step')
        plt.ylabel('Loss Difference')
        plt.grid(True, alpha=0.3)
    
    # Добавляем сглаженные версии лоссов
    if len(steps) > 10:
        window_size = max(5, len(steps) // 20)
        
        def smooth(y, window_size):
            return np.convolve(y, np.ones(window_size), 'valid') / window_size
        
        smooth_steps = steps[window_size-1:]
        smooth_train = smooth(train_losses, window_size)
        
        plt.subplot(2, 3, 4)
        plt.plot(smooth_steps, smooth_train, 'b-', label='Smooth Train Loss', linewidth=2)
        
        if test_losses and len(test_losses) > window_size:
            smooth_test = smooth(test_losses, window_size)
            smooth_test_steps = test_steps[window_size-1:]
            plt.plot(smooth_test_steps, smooth_test, 'r-', label='Smooth Test Loss', linewidth=2, marker='o', markersize=2)
        
        plt.title(f'Smoothed Loss (window={window_size})')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Статистика
    plt.subplot(2, 3, 5)
    stats_text = f"""
    Статистика обучения:
    
    Шагов: {len(steps)}
    Оценок test_loss: {len(test_losses)}
    
    Train Loss:
    • Начальный: {train_losses[0]:.4f}
    • Финальный: {train_losses[-1]:.4f}
    • Минимум: {min(train_losses):.4f}
    • Максимум: {max(train_losses):.4f}
    """
    
    if test_losses:
        stats_text += f"""
    Test Loss:
    • Начальный: {test_losses[0]:.4f}
    • Финальный: {test_losses[-1]:.4f}
    • Минимум: {min(test_losses):.4f}
    • Максимум: {max(test_losses):.4f}
    """
    
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=10, fontfamily='monospace')
    plt.axis('off')
    
    # Последние значения
    plt.subplot(2, 3, 6)
    recent_steps = steps[-min(50, len(steps)):]
    recent_train = train_losses[-min(50, len(steps)):]
    
    plt.plot(recent_steps, recent_train, 'b-', label='Train Loss', linewidth=2)
    
    if test_losses:
        # Берем только последние test_loss точки
        recent_test_count = min(50, len(test_losses))
        recent_test_steps = test_steps[-recent_test_count:]
        recent_test_losses = test_losses[-recent_test_count:]
        
        plt.plot(recent_test_steps, recent_test_losses, 'r-', label='Test Loss', linewidth=2, marker='o', markersize=3)
    
    plt.title('Recent Loss (last 50 steps)')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Графики обучения сохранены: {plot_path}")
    print(f"   • Train Loss: {len(steps)} точек")
    print(f"   • Test Loss: {len(test_losses)} точек")


def plot_metrics_curves(metrics_files: List[str], save_dir: str):
    """
    Построение графиков метрик из файлов
    
    Args:
        metrics_files: список путей к файлам метрик
        save_dir: директория для сохранения
    """
    if not metrics_files:
        print("⚠️ Нет файлов метрик для построения графиков")
        return
    
    # Загружаем все метрики
    all_metrics = []
    for file_path in metrics_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                metrics = json.load(f)
                all_metrics.append(metrics)
    
    if not all_metrics:
        print("⚠️ Не удалось загрузить файлы метрик")
        return
    
    # Сортируем метрики по шагам для правильного порядка точек
    all_metrics.sort(key=lambda x: x.get('step', 0))
    
    # Извлекаем данные
    steps = [m['step'] for m in all_metrics if 'step' in m]
    roc_aucs = [m['roc_auc'] for m in all_metrics if 'roc_auc' in m]
    
    if not steps or not roc_aucs:
        print("⚠️ Недостаточно данных для графиков метрик")
        return
    
    # Графики для разных порогов
    thresholds = [0.05, 0.15, 0.25, 0.5]
    
    plt.figure(figsize=(15, 10))
    
    # ROC AUC
    plt.subplot(2, 3, 1)
    plt.plot(steps, roc_aucs, 'b-', linewidth=2, marker='o')
    plt.title('ROC AUC')
    plt.xlabel('Step')
    plt.ylabel('ROC AUC')
    plt.grid(True, alpha=0.3)
    
    # Метрики для каждого порога
    for i, threshold in enumerate(thresholds):
        acc_key = f'acc_{threshold}'
        f1_key = f'f1_{threshold}'
        
        if all(acc_key in m for m in all_metrics):
            accs = [m[acc_key] for m in all_metrics]
            plt.subplot(2, 3, i+2)
            plt.plot(steps, accs, 'g-', linewidth=2, marker='o', label='Accuracy')
            
            if all(f1_key in m for m in all_metrics):
                f1s = [m[f1_key] for m in all_metrics]
                plt.plot(steps, f1s, 'r-', linewidth=2, marker='s', label='F1')
            
            plt.title(f'Threshold {threshold}')
            plt.xlabel('Step')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'metrics_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Графики метрик сохранены: {plot_path}")


def train_cross_attention_model(
    model: CrossAttentionModel,
    train_dataset: CrossAttentionDataset,
    test_dataset: CrossAttentionDataset,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    device: str = "auto",
    save_dir: str = "./checkpoints",
    eval_every_steps: int = 100,
    save_metrics_every_steps: int = 50,
    batch_size: int = 1,
    num_workers: int = 0,
    max_steps: int = None
):
    """
    Обучение Cross-Attention модели
    
    Args:
        model: модель для обучения
        train_dataset: тренировочный датасет
        test_dataset: тестовый датасет
        num_epochs: количество эпох
        learning_rate: скорость обучения
        device: устройство для обучения
        save_dir: директория для сохранения
        eval_every_steps: частота полной оценки
        save_metrics_every_steps: частота сохранения метрик
        batch_size: размер батча для DataLoader
        num_workers: количество workers для DataLoader
    """
    
    # Настройка логирования
    logger = setup_logging(os.path.join(save_dir, "logs"))
    
    # Определяем устройство
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    
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
    
    # Оптимизатор
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Планировщик
    total_steps = len(train_loader) * num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Сохранение лучших моделей
    model_saver = BestModelSaver(
        os.path.join(save_dir, "best_models"), 
        keep_top_k=2,
        test_loader=test_loader,
        device=device
    )
    
    # Создаем структуру папок
    results_dir = save_dir  # Главная папка эксперимента
    metrics_dir = os.path.join(save_dir, "metrics")  # Подпапка для JSON файлов
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    logger.info(f"🚀 Начало обучения:")
    logger.info(f"   Эпохи: {num_epochs}")
    logger.info(f"   Шагов на эпоху: {len(train_loader)}")
    logger.info(f"   Всего шагов: {total_steps}")
    logger.info(f"   Устройство: {device}")
    logger.info(f"   Learning rate: {learning_rate}")
    logger.info(f"   Результаты: {results_dir}")
    logger.info(f"   Метрики: {metrics_dir}")
    
    # Логирование
    training_log = []
    step_counter = 0
    best_roc_auc = 0.0
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (vectors_1, vectors_2, gt_matrix, dataset_name) in enumerate(train_loader):
            # Перемещаем данные на устройство
            vectors_1 = vectors_1.squeeze(0).to(device)  # Убираем batch размерность от DataLoader
            vectors_2 = vectors_2.squeeze(0).to(device)
            gt_matrix = gt_matrix.squeeze(0).to(device)
            
            # Forward pass
            pred_matrix = model(vectors_1, vectors_2)
            
            # Loss
            loss = weighted_bce_loss(pred_matrix, gt_matrix)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            step_counter += 1
            
            # Записываем основные метрики на каждом шаге для детальных графиков
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
            
            # Проверка max_steps
            if max_steps is not None and step_counter >= max_steps:
                logger.info(f"🛑 Достигнуто максимальное количество шагов: {max_steps}")
                break
            
            # Логирование каждые N шагов (только для консоли, без записи в training_log)
            if step_counter % save_metrics_every_steps == 0:
                log_message = (f"Шаг {step_counter:5d} | Эпоха {epoch+1:2d}/{num_epochs} | "
                              f"Train Loss: {loss.item():.4f} | "
                              f"LR: {scheduler.get_last_lr()[0]:.6f} | Dataset: {dataset_name}")
                logger.info(log_message)
            
            # Полная оценка каждые N шагов
            if step_counter % eval_every_steps == 0:
                logger.info(f"\n📊 Полная оценка на шаге {step_counter}...")
                
                # Быстрый test_loss для training_curves
                model.eval()
                with torch.no_grad():
                    test_vectors_1, test_vectors_2, test_gt_matrix, test_dataset_name = next(iter(test_loader))
                    test_vectors_1 = test_vectors_1.squeeze(0).to(device)
                    test_vectors_2 = test_vectors_2.squeeze(0).to(device)
                    test_gt_matrix = test_gt_matrix.squeeze(0).to(device)
                    
                    test_pred_matrix = model(test_vectors_1, test_vectors_2)
                    test_loss_for_curves = weighted_bce_loss(test_pred_matrix, test_gt_matrix)
                
                # Добавляем test_loss к текущей записи в training_log
                training_log[-1]['test_loss'] = test_loss_for_curves.item()
                
                # Полная оценка метрик
                metrics = evaluate_model(model, test_loader, device)
                
                logger.info(f"🎯 Метрики:")
                logger.info(f"   ROC AUC: {metrics['roc_auc']:.4f}")
                for threshold in [0.05, 0.15, 0.25, 0.5]:
                    threshold_msg = (f"   Порог {threshold}: Acc={metrics[f'acc_{threshold}']:.3f}, "
                                   f"F1={metrics[f'f1_{threshold}']:.3f}, "
                                   f"Prec={metrics[f'precision_{threshold}']:.3f}, "
                                   f"Rec={metrics[f'recall_{threshold}']:.3f}")
                    logger.info(threshold_msg)
                
                # Сохранение метрик в подпапку metrics/
                metrics['step'] = step_counter
                metrics['epoch'] = epoch
                save_metrics(metrics, f"{metrics_dir}/metrics_step_{step_counter}.json")
                
                # Сохранение лучших моделей
                current_roc_auc = metrics['roc_auc']
                if current_roc_auc > best_roc_auc:
                    best_roc_auc = current_roc_auc
                    model_saver.save_if_best(model, current_roc_auc, step_counter, metrics)
                
                model.train()  # Возвращаем модель в режим обучения
                logger.info("")
        
        # Проверка max_steps для выхода из цикла по эпохам
        if max_steps is not None and step_counter >= max_steps:
            logger.info(f"🛑 Прерывание обучения - достигнуто максимальное количество шагов: {max_steps}")
            break
        
        # Статистика эпохи
        avg_epoch_loss = epoch_loss / len(train_loader)
        elapsed_time = time.time() - start_time
        
        logger.info(f"\n📈 Эпоха {epoch+1}/{num_epochs} завершена:")
        logger.info(f"   Средний loss: {avg_epoch_loss:.4f}")
        logger.info(f"   Время: {elapsed_time/60:.1f} мин")
        logger.info(f"   Лучший ROC AUC: {best_roc_auc:.4f}")
        logger.info("-" * 60)
    
    # Сохранение лога обучения в подпапку metrics/
    with open(f"{metrics_dir}/training_log.json", 'w') as f:
        json.dump(training_log, f, indent=2)
    
    # Построение графиков обучения (сохраняем в главную папку)
    logger.info(f"\n📈 Построение графиков обучения...")
    plot_training_curves(training_log, results_dir)
    
    # Построение графиков метрик (сохраняем в главную папку)
    metrics_files = [f for f in os.listdir(metrics_dir) if f.startswith('metrics_step_') and f.endswith('.json')]
    # Сортируем файлы по номеру шага, а не по строке
    def extract_step_from_filename(filename):
        try:
            # Извлекаем число из имени файла "metrics_step_123.json"
            step_str = filename.replace('metrics_step_', '').replace('.json', '')
            return int(step_str)
        except ValueError:
            return 0
    
    metrics_files.sort(key=extract_step_from_filename)
    metrics_files = [os.path.join(metrics_dir, f) for f in metrics_files]
    if metrics_files:
        plot_metrics_curves(metrics_files, results_dir)
    
    # Финальная оценка
    logger.info(f"\n🏁 Финальная оценка...")
    final_metrics = evaluate_model(model, test_loader, device)
    save_metrics(final_metrics, f"{metrics_dir}/final_metrics.json")
    
    # Анализ порогов (график в главную папку, JSON в metrics/)
    logger.info(f"\n🔍 Анализ порогов (0.01 - 0.60)...")
    threshold_analysis = analyze_thresholds(model, test_loader, device, results_dir)
    save_metrics(threshold_analysis, f"{metrics_dir}/threshold_analysis.json")
    
    # Выводим результаты анализа порогов
    logger.info(f"📊 Лучшие пороги:")
    for metric, threshold in threshold_analysis['best_thresholds'].items():
        value = threshold_analysis['best_values'][metric]
        logger.info(f"   {metric.capitalize()}: {threshold:.2f} (значение: {value:.4f})")
    
    logger.info(f"✅ Обучение завершено!")
    logger.info(f"   Финальный ROC AUC: {final_metrics['roc_auc']:.4f}")
    logger.info(f"   Лучший ROC AUC: {best_roc_auc:.4f}")
    logger.info(f"   Время обучения: {(time.time() - start_time)/60:.1f} мин")
    logger.info(f"   📁 Структура файлов:")
    logger.info(f"      📊 Графики: {results_dir}/*.png")
    logger.info(f"      📋 Метрики: {metrics_dir}/*.json")
    
    return model, training_log, final_metrics, threshold_analysis


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cross-Attention Model Training")
    parser.add_argument("--entities_path", type=str, required=True, help="Path to entities JSON file")
    parser.add_argument("--relations_path", type=str, required=True, help="Path to relations JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--eval_every", type=int, default=10, help="Evaluate every N steps")
    parser.add_argument("--save_every", type=int, default=50, help="Save every N steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split size")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum training steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_qwen3", action="store_true", help="Initialize from Qwen3")
    parser.add_argument("--qwen3_model", type=str, default="Qwen/Qwen3-4B", help="Qwen3 model name")
    parser.add_argument("--dataset_strategy", type=str, default="single", choices=["single", "weighted"], help="Dataset selection strategy")
    parser.add_argument("--sampling_strategy", type=str, default="balanced", choices=["random", "balanced"], help="Entity sampling strategy within dataset")
    parser.add_argument("--positive_ratio", type=float, default=1.0, help="Ratio of positive pairs for balanced sampling (1.0 = maximum possible)")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    parser.add_argument("--num_attention_heads", type=int, default=8, help="Number of attention heads")
    
    args = parser.parse_args()
    
    # Создаем уникальную папку для эксперимента
    experiment_name = create_experiment_name(args)
    experiment_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Настройка логирования для основного скрипта
    logger = setup_logging(experiment_dir, args.log_level)
    
    logger.info(f"🚀 Начало эксперимента: {experiment_name}")
    logger.info(f"📁 Папка эксперимента: {experiment_dir}")
    logger.info(f"🎲 Seed: {args.seed}")
    
    # Создаем датасеты с фиксированным seed
    logger.info(f"🔄 Создание датасетов...")
    logger.info(f"   Entities: {args.entities_path}")
    logger.info(f"   Relations: {args.relations_path}")
    
    train_dataset, test_dataset = create_train_test_datasets(
        args.entities_path,
        args.relations_path,
        batch_size_1=args.batch_size,
        batch_size_2=args.batch_size,
        dataset_strategy=args.dataset_strategy,
        sampling_strategy=args.sampling_strategy,
        positive_ratio=args.positive_ratio,
        test_part=args.test_size,
        random_state=args.seed  # Фиксируем seed для воспроизводимости
    )
    
    logger.info(f"✅ Датасеты созданы: Train={len(train_dataset)}, Test={len(test_dataset)}")
    
    # Создаем или инициализируем модель
    if args.use_qwen3:
        logger.info(f"🔄 Инициализация из Qwen3: {args.qwen3_model}")
        model = initialize_from_qwen3(args.qwen3_model)
    else:
        logger.info(f"🔄 Создание стандартной модели...")
        # Создаем стандартную конфигурацию (размерность должна совпадать с эмбеддингами)
        test_config = {
            'hidden_size': 2560,  # Размерность эмбеддингов из данных
            'num_attention_heads': args.num_attention_heads,
            'num_key_value_heads': args.num_attention_heads,
            'rms_norm_eps': 1e-6,
            'attention_bias': False
        }
        model = CrossAttentionModel(test_config)
    
    logger.info(f"✅ Модель создана: {model}")
    
    # Запускаем обучение в уникальной папке эксперимента
    logger.info(f"\n🚀 Начинаем обучение...")
    trained_model, training_log, final_metrics, threshold_analysis = train_cross_attention_model(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        save_dir=experiment_dir,  # Используем уникальную папку эксперимента
        eval_every_steps=args.eval_every,
        save_metrics_every_steps=args.save_every,
        batch_size=1,  # DataLoader batch size (внутренний batch размер управляется датасетом)
        num_workers=0,
        max_steps=args.max_steps
    )
    
    # Создаем итоговую сводку результатов в главной папке эксперимента
    logger.info(f"\n📋 Создание итоговой сводки...")
    create_best_results_summary(
        threshold_analysis,
        final_metrics,
        training_log,
        args,
        os.path.join(experiment_dir, "best_results.json")
    )
    
    logger.info(f"\n✅ Обучение завершено! Результаты сохранены в: {experiment_dir}")
    logger.info(f"   📊 Графики: training_curves.png, metrics_curves.png, threshold_analysis.png")
    logger.info(f"   📋 Итоговая сводка: best_results.json")
    logger.info(f"   📂 Детальные метрики: metrics/")
    logger.info(f"   🏆 Лучшие модели: best_models/ (включая анализ порогов)")
    logger.info(f"       • best_models.json")
    logger.info(f"       • best_model_threshold_analysis.json")
    logger.info(f"       • best_model_threshold_analysis.png")
    logger.info(f"       • best_model_evaluation_summary.json")
    