#!/usr/bin/env python3
"""
Script for evaluating the best models from all experiments
Loads the best models, recreates test data and performs threshold analysis
"""

import os
import json
import re
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import argparse
from datetime import datetime
import logging

# Импортируем необходимые компоненты
from cross_attention_model import CrossAttentionModel
from dataset import create_train_test_datasets
from train_cross_attention import (
    evaluate_model, 
    analyze_thresholds, 
    save_metrics
)


def setup_logging() -> logging.Logger:
    """Настройка логирования"""
    logger = logging.getLogger("evaluate_best_models")
    logger.setLevel(logging.INFO)
    
    # Очищаем существующие handlers
    logger.handlers.clear()
    
    # Форматтер для логов
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler для консоли
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger


def parse_experiment_name(experiment_name: str) -> Dict[str, Any]:
    """
    Парсинг имени эксперимента для извлечения параметров
    
    Args:
        experiment_name: имя папки эксперимента
        
    Returns:
        dict с параметрами эксперимента
    """
    # Убираем timestamp в начале (дата и время: 20250708_134939_...)
    parts = experiment_name.split('_')
    
    # Пропускаем первые две части если они выглядят как дата и время
    start_idx = 0
    if len(parts) >= 2 and parts[0].isdigit() and len(parts[0]) == 8 and parts[1].isdigit():
        start_idx = 2  # Пропускаем дату и время
    elif len(parts) >= 1 and parts[0].isdigit():
        start_idx = 1  # Пропускаем только первую часть
    
    parts = parts[start_idx:]
    
    params = {}
    
    for i, part in enumerate(parts):
        if part.startswith('ep'):
            params['epochs'] = int(part[2:])
        elif part.startswith('lr'):
            # Преобразуем научную нотацию
            lr_str = part[2:]
            if 'e-' in lr_str:
                params['lr'] = float(lr_str)
            else:
                params['lr'] = float(lr_str)
        elif part.startswith('bs'):
            params['batch_size'] = int(part[2:])
        elif part.startswith('eval'):
            params['eval_every'] = int(part[4:])
        elif part.startswith('seed'):
            params['seed'] = int(part[4:])
        elif part == 'qwen3':
            params['use_qwen3'] = True
        elif part.startswith('ds_'):
            params['dataset_strategy'] = part[3:]
        elif part.startswith('samp_'):
            params['sampling_strategy'] = part[5:]
        elif part.startswith('pos'):
            params['positive_ratio'] = float(part[3:])
        elif part.startswith('max'):
            params['max_steps'] = int(part[3:])
        elif i == 0:  # Первая часть после timestamp - это dataset
            params['dataset_name'] = part
    
    # Устанавливаем значения по умолчанию
    params.setdefault('use_qwen3', False)
    params.setdefault('dataset_strategy', 'single')
    params.setdefault('sampling_strategy', 'balanced')
    params.setdefault('positive_ratio', 1.0)
    params.setdefault('max_steps', None)
    params.setdefault('test_size', 0.2)
    
    return params


def find_best_model(experiment_path: str) -> Optional[Dict[str, Any]]:
    """
    Находит лучшую модель в папке эксперимента
    
    Args:
        experiment_path: путь к папке эксперимента
        
    Returns:
        dict с информацией о лучшей модели или None
    """
    best_models_path = os.path.join(experiment_path, 'best_models')
    best_models_json = os.path.join(best_models_path, 'best_models.json')
    
    if not os.path.exists(best_models_json):
        return None
    
    try:
        with open(best_models_json, 'r') as f:
            data = json.load(f)
        
        if not data.get('best_models'):
            return None
            
        # Лучшая модель - первая в списке (с наивысшим ROC AUC)
        best_model = data['best_models'][0]
        
        # Проверяем, существует ли файл модели
        model_path = best_model['model_path']
        if not os.path.exists(model_path):
            return None
            
        return best_model
        
    except Exception as e:
        print(f"Ошибка при чтении {best_models_json}: {e}")
        return None


def get_dataset_paths(dataset_name: str) -> Tuple[str, str]:
    """
    Получение путей к файлам датасета
    
    Args:
        dataset_name: название датасета
        
    Returns:
        tuple с путями к entities и relations
    """
    # Путь к папке с данными
    data_dir = "/home/jovyan/rahmatullaev/rand_exps/LLMs4OL-Challenge/2025/TaskC-TaxonomyDiscovery"
    
    # Структура: DatasetName/train/datasetname_train_types_embeddings.json и datasetname_train_pairs.json
    dataset_train_dir = os.path.join(data_dir, dataset_name, "train")
    
    # Имена файлов используют lowercase имя датасета
    dataset_lower = dataset_name.lower()
    
    entities_path = os.path.join(dataset_train_dir, f"{dataset_lower}_train_types_embeddings.json")
    relations_path = os.path.join(dataset_train_dir, f"{dataset_lower}_train_pairs.json")
    
    return entities_path, relations_path


def load_model(model_path: str, use_qwen3: bool = False) -> CrossAttentionModel:
    """
    Загрузка модели из файла
    
    Args:
        model_path: путь к файлу модели
        use_qwen3: была ли модель инициализирована из Qwen3 (используется только для информации)
        
    Returns:
        загруженная модель
    """
    # Загружаем checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Получаем config из checkpoint или создаем правильный
    if 'config' in checkpoint and not use_qwen3:
        # Для обычных моделей используем config из checkpoint
        config = checkpoint['config']
    else:
        # Захардкоженные параметры для совместимости с существующими checkpoint'ами
        if use_qwen3:
            # Параметры для Qwen3 моделей (из фактических размеров весов)
            config = {
                'hidden_size': 2560,
                'num_attention_heads': 32,  # 4096 / 128
                'num_key_value_heads': 8,   # 1024 / 128  
                'head_dim': 128,            # q_norm.weight.shape[0]
                'rms_norm_eps': 1e-6,
                'attention_bias': False
            }
        else:
            # Стандартная конфигурация для обычных моделей
            config = {
                'hidden_size': 2560,
                'num_attention_heads': 8,
                'num_key_value_heads': 8,
                'head_dim': 320,  # 2560 / 8
                'rms_norm_eps': 1e-6,
                'attention_bias': False
            }
    
    # Создаем модель с правильным config
    model = CrossAttentionModel(config)
    
    # Восстанавливаем дополнительные атрибуты если они есть
    if 'layer_idx' in checkpoint:
        model.layer_idx = checkpoint['layer_idx']
    if 'initialized_from_qwen3' in checkpoint:
        model.initialized_from_qwen3 = checkpoint['initialized_from_qwen3']
    
    # Загружаем веса - проверяем различные форматы сохранения
    if 'model_state_dict' in checkpoint:
        # Модель сохранена как структура с дополнительными полями
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'q_proj.weight' in checkpoint:
        # Модель сохранена как простой state_dict
        model.load_state_dict(checkpoint)
    else:
        # Пытаемся загрузить как есть
        model.load_state_dict(checkpoint)
    
    return model


def evaluate_experiment(experiment_path: str, logger: logging.Logger) -> bool:
    """
    Оценка одного эксперимента
    
    Args:
        experiment_path: путь к папке эксперимента
        logger: логгер
        
    Returns:
        True если оценка прошла успешно
    """
    experiment_name = os.path.basename(experiment_path)
    logger.info(f"🔍 Обработка эксперимента: {experiment_name}")
    
    # Парсим параметры эксперимента
    try:
        params = parse_experiment_name(experiment_name)
        logger.info(f"   Параметры: {params}")
    except Exception as e:
        logger.error(f"   ❌ Ошибка парсинга имени: {e}")
        return False
    
    # Находим лучшую модель
    best_model_info = find_best_model(experiment_path)
    if not best_model_info:
        logger.warning(f"   ⚠️ Лучшая модель не найдена")
        return False
        
    logger.info(f"   📊 Лучшая модель: ROC AUC = {best_model_info['roc_auc']:.4f}, Step = {best_model_info['step']}")
    
    # Получаем пути к данным
    try:
        entities_path, relations_path = get_dataset_paths(params['dataset_name'])
        if not os.path.exists(entities_path) or not os.path.exists(relations_path):
            logger.error(f"   ❌ Файлы данных не найдены: {entities_path}, {relations_path}")
            return False
    except Exception as e:
        logger.error(f"   ❌ Ошибка получения путей к данным: {e}")
        return False
    
    # Воссоздаем датасеты с тем же seed
    try:
        logger.info(f"   🔄 Воссоздание датасетов с seed={params['seed']}")
        train_dataset, test_dataset = create_train_test_datasets(
            entities_path,
            relations_path,
            batch_size_1=params['batch_size'],
            batch_size_2=params['batch_size'],
            dataset_strategy=params['dataset_strategy'],
            sampling_strategy=params['sampling_strategy'],
            positive_ratio=params['positive_ratio'],
            test_part=params['test_size'],
            random_state=params['seed']
        )
        logger.info(f"   ✅ Датасеты созданы: Train={len(train_dataset)}, Test={len(test_dataset)}")
    except Exception as e:
        logger.error(f"   ❌ Ошибка создания датасетов: {e}")
        return False
    
    # Загружаем модель
    try:
        logger.info(f"   🔄 Загрузка модели: {best_model_info['model_path']}")
        model = load_model(best_model_info['model_path'], params['use_qwen3'])
        logger.info(f"   ✅ Модель загружена")
    except Exception as e:
        logger.error(f"   ❌ Ошибка загрузки модели: {e}")
        return False
    
    # Создаем DataLoader для тестирования
    from torch.utils.data import DataLoader
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device == "cuda" else False
    )
    
    # Проводим оценку
    try:
        logger.info(f"   🎯 Оценка модели на тестовых данных")
        final_metrics = evaluate_model(model, test_loader, device)
        logger.info(f"   📊 ROC AUC: {final_metrics['roc_auc']:.4f}")
        
        # Анализ порогов
        logger.info(f"   🔍 Анализ порогов")
        threshold_analysis = analyze_thresholds(model, test_loader, device, None)
        
        # Сохраняем результаты в папку best_models
        best_models_dir = os.path.join(experiment_path, 'best_models')
        
        # Сохраняем финальные метрики
        final_metrics_path = os.path.join(best_models_dir, 'best_model_final_metrics.json')
        save_metrics(final_metrics, final_metrics_path)
        
        # Сохраняем анализ порогов
        threshold_analysis_path = os.path.join(best_models_dir, 'best_model_threshold_analysis.json')
        save_metrics(threshold_analysis, threshold_analysis_path)
        
        # Создаем сводку
        summary = {
            'experiment_name': experiment_name,
            'experiment_params': params,
            'best_model_info': best_model_info,
            'evaluation_timestamp': datetime.now().isoformat(),
            'final_metrics': final_metrics,
            'best_thresholds': threshold_analysis['best_thresholds'],
            'best_values': threshold_analysis['best_values']
        }
        
        summary_path = os.path.join(best_models_dir, 'best_model_evaluation_summary.json')
        save_metrics(summary, summary_path)
        
        logger.info(f"   ✅ Результаты сохранены в {best_models_dir}")
        logger.info(f"   🎯 Лучшие пороги:")
        for metric, threshold in threshold_analysis['best_thresholds'].items():
            value = threshold_analysis['best_values'][metric]
            logger.info(f"      {metric.capitalize()}: {threshold:.2f} (значение: {value:.4f})")
        
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Ошибка оценки: {e}")
        return False


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description="Оценка лучших моделей из всех экспериментов")
    parser.add_argument("--results_dir", type=str, 
                       default="/home/jovyan/rahmatullaev/rand_exps/LLMs4OL-Challenge/src/taskC/method_v5_hm/results",
                       help="Папка с результатами экспериментов")
    parser.add_argument("--experiment_filter", type=str, default=None,
                       help="Фильтр для имен экспериментов (регулярное выражение)")
    parser.add_argument("--max_experiments", type=int, default=None,
                       help="Максимальное количество экспериментов для обработки")
    
    args = parser.parse_args()
    
    # Настройка логирования
    logger = setup_logging()
    
    logger.info(f"🚀 Начало оценки лучших моделей")
    logger.info(f"📁 Папка с результатами: {args.results_dir}")
    
    # Находим все папки с экспериментами
    experiment_dirs = []
    for item in os.listdir(args.results_dir):
        item_path = os.path.join(args.results_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            # Пропускаем папки grid_search_*
            if item.startswith('grid_search_'):
                continue
            # Применяем фильтр если указан
            if args.experiment_filter:
                if not re.search(args.experiment_filter, item):
                    continue
            experiment_dirs.append(item_path)
    
    # Сортируем по имени
    experiment_dirs.sort()
    
    # Ограничиваем количество если указано
    if args.max_experiments:
        experiment_dirs = experiment_dirs[:args.max_experiments]
    
    logger.info(f"📊 Найдено экспериментов: {len(experiment_dirs)}")
    
    # Обрабатываем каждый эксперимент
    success_count = 0
    error_count = 0
    
    for i, experiment_path in enumerate(experiment_dirs, 1):
        logger.info(f"\n📍 Прогресс: {i}/{len(experiment_dirs)}")
        
        try:
            if evaluate_experiment(experiment_path, logger):
                success_count += 1
            else:
                error_count += 1
        except Exception as e:
            logger.error(f"❌ Критическая ошибка при обработке {experiment_path}: {e}")
            error_count += 1
    
    logger.info(f"\n🏁 Обработка завершена!")
    logger.info(f"   ✅ Успешно обработано: {success_count}")
    logger.info(f"   ❌ Ошибок: {error_count}")
    logger.info(f"   📊 Всего экспериментов: {len(experiment_dirs)}")


if __name__ == "__main__":
    main() 