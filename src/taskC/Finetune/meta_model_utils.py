"""
Утилиты для работы с Qwen3CrossAttentionMetaModel

Включает функции для:
- Создания датасета для мета-модели
- Оценки производительности
- Визуализации результатов
- Конвертации данных
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import time

from meta_model import Qwen3CrossAttentionMetaModel


class MetaModelDataset(Dataset):
    """
    Датасет для работы с мета-моделью
    
    Принимает список примеров в формате:
    [
        {
            "phrases_1": ["phrase1", "phrase2"],
            "phrases_2": ["phrase3", "phrase4"],
            "labels": [[0, 1], [1, 0]]  # матрица отношений
        }
    ]
    """
    
    def __init__(self, examples: List[Dict[str, Any]]):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            "phrases_1": example["phrases_1"],
            "phrases_2": example["phrases_2"],
            "labels": torch.tensor(example["labels"], dtype=torch.float32)
        }


def create_meta_dataset_from_cross_attention_data(
    entities_path: str,
    relations_path: str,
    sample_size: int = 1000,
    max_phrases_per_sample: int = 20,
    random_state: int = 42
) -> List[Dict[str, Any]]:
    """
    Создание датасета для мета-модели из данных CrossAttentionDataset
    
    Args:
        entities_path: путь к entities.json
        relations_path: путь к relations.json
        sample_size: количество примеров в датасете
        max_phrases_per_sample: максимальное количество фраз в одном примере
        random_state: seed для воспроизводимости
        
    Returns:
        examples: список примеров для MetaModelDataset
    """
    import random
    random.seed(random_state)
    
    # Загружаем данные
    with open(entities_path, 'r', encoding='utf-8') as f:
        entities = json.load(f)
    
    with open(relations_path, 'r', encoding='utf-8') as f:
        relations = json.load(f)
    
    # Создаем mapping от text_description к ID
    text_to_id = {entity['text_description']: entity['id'] for entity in entities}
    id_to_text = {entity['id']: entity['text_description'] for entity in entities}
    
    # Создаем set отношений для быстрого поиска
    relation_set = set()
    for rel in relations:
        if isinstance(rel, dict):
            parent_text = rel.get('parent', '')
            child_text = rel.get('child', '')
            if parent_text and child_text:
                relation_set.add((parent_text, child_text))
        else:
            # Старый формат [id1, id2]
            parent_id, child_id = rel
            parent_text = id_to_text.get(parent_id, '')
            child_text = id_to_text.get(child_id, '')
            if parent_text and child_text:
                relation_set.add((parent_text, child_text))
    
    # Получаем все уникальные термины
    all_terms = list(set(entity['text_description'] for entity in entities 
                        if entity.get('text_description', '')))
    
    examples = []
    
    for _ in range(sample_size):
        # Случайно выбираем фразы
        num_phrases = random.randint(5, max_phrases_per_sample)
        selected_terms = random.sample(all_terms, min(num_phrases, len(all_terms)))
        
        # Создаем матрицу отношений
        labels = []
        for child_term in selected_terms:
            row = []
            for parent_term in selected_terms:
                # Проверяем, есть ли отношение child -> parent
                has_relation = (child_term, parent_term) in relation_set
                row.append(1 if has_relation else 0)
            labels.append(row)
        
        examples.append({
            "phrases_1": selected_terms,
            "phrases_2": selected_terms,
            "labels": labels
        })
    
    return examples


def evaluate_meta_model(
    meta_model: Qwen3CrossAttentionMetaModel,
    test_examples: List[Dict[str, Any]],
    thresholds: List[float] = None,
    device: str = "auto"
) -> Dict[str, Any]:
    """
    Оценка производительности мета-модели
    
    Args:
        meta_model: мета-модель для оценки
        test_examples: тестовые примеры
        thresholds: пороги для оценки
        device: устройство для вычислений
        
    Returns:
        results: результаты оценки
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    meta_model.eval()
    
    all_predictions = []
    all_labels = []
    
    print(f"Оценка модели на {len(test_examples)} примерах...")
    
    with torch.no_grad():
        for i, example in enumerate(test_examples):
            if i % 10 == 0:
                print(f"Обработано {i}/{len(test_examples)} примеров")
            
            phrases_1 = example["phrases_1"]
            phrases_2 = example["phrases_2"]
            labels = example["labels"]
            
            # Получаем предсказания
            predictions = meta_model(phrases_1, phrases_2)
            
            # Приводим к numpy
            predictions_np = predictions.cpu().numpy().flatten()
            labels_np = labels.numpy().flatten()
            
            all_predictions.extend(predictions_np)
            all_labels.extend(labels_np)
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Вычисляем ROC-AUC
    try:
        roc_auc = roc_auc_score(all_labels, all_predictions)
    except:
        roc_auc = 0.0
    
    # Оценка для разных порогов
    threshold_results = {}
    
    for threshold in thresholds:
        binary_predictions = (all_predictions > threshold).astype(int)
        
        try:
            accuracy = accuracy_score(all_labels, binary_predictions)
            precision = precision_score(all_labels, binary_predictions, zero_division=0)
            recall = recall_score(all_labels, binary_predictions, zero_division=0)
            f1 = f1_score(all_labels, binary_predictions, zero_division=0)
        except:
            accuracy = precision = recall = f1 = 0.0
        
        threshold_results[threshold] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    
    # Находим лучший порог
    best_threshold = max(thresholds, key=lambda t: threshold_results[t]["f1_score"])
    
    results = {
        "roc_auc": roc_auc,
        "best_threshold": best_threshold,
        "best_metrics": threshold_results[best_threshold],
        "threshold_results": threshold_results,
        "num_examples": len(test_examples),
        "num_predictions": len(all_predictions),
        "positive_ratio": np.mean(all_labels)
    }
    
    return results


def visualize_evaluation_results(
    results: Dict[str, Any],
    save_path: str = None,
    show_plot: bool = True
):
    """
    Визуализация результатов оценки
    
    Args:
        results: результаты оценки от evaluate_meta_model
        save_path: путь для сохранения графика
        show_plot: показать ли график
    """
    threshold_results = results["threshold_results"]
    
    # Извлекаем данные для графика
    thresholds = list(threshold_results.keys())
    accuracies = [threshold_results[t]["accuracy"] for t in thresholds]
    precisions = [threshold_results[t]["precision"] for t in thresholds]
    recalls = [threshold_results[t]["recall"] for t in thresholds]
    f1_scores = [threshold_results[t]["f1_score"] for t in thresholds]
    
    # Создаем график
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(thresholds, accuracies, 'b-', marker='o', label='Accuracy')
    plt.axvline(x=results["best_threshold"], color='r', linestyle='--', alpha=0.7, label=f'Best threshold: {results["best_threshold"]}')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(thresholds, precisions, 'g-', marker='o', label='Precision')
    plt.axvline(x=results["best_threshold"], color='r', linestyle='--', alpha=0.7, label=f'Best threshold: {results["best_threshold"]}')
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.title('Precision vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(thresholds, recalls, 'orange', marker='o', label='Recall')
    plt.axvline(x=results["best_threshold"], color='r', linestyle='--', alpha=0.7, label=f'Best threshold: {results["best_threshold"]}')
    plt.xlabel('Threshold')
    plt.ylabel('Recall')
    plt.title('Recall vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.plot(thresholds, f1_scores, 'purple', marker='o', label='F1 Score')
    plt.axvline(x=results["best_threshold"], color='r', linestyle='--', alpha=0.7, label=f'Best threshold: {results["best_threshold"]}')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Добавляем общий заголовок
    plt.suptitle(f'Meta-Model Evaluation Results (ROC-AUC: {results["roc_auc"]:.4f})', 
                 fontsize=16, y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График сохранен в {save_path}")
    
    if show_plot:
        plt.show()


def benchmark_meta_model_performance(
    meta_model: Qwen3CrossAttentionMetaModel,
    phrase_counts: List[int] = None,
    num_trials: int = 5
) -> Dict[str, Any]:
    """
    Бенчмарк производительности мета-модели
    
    Args:
        meta_model: мета-модель для тестирования
        phrase_counts: количества фраз для тестирования
        num_trials: количество испытаний для каждого размера
        
    Returns:
        benchmark_results: результаты бенчмарка
    """
    if phrase_counts is None:
        phrase_counts = [5, 10, 20, 50, 100]
    
    # Создаем тестовые фразы
    test_phrases = [
        f"test phrase {i}" for i in range(max(phrase_counts))
    ]
    
    results = {}
    
    meta_model.eval()
    
    for phrase_count in phrase_counts:
        print(f"Тестирование с {phrase_count} фразами...")
        
        current_phrases = test_phrases[:phrase_count]
        times = []
        
        for trial in range(num_trials):
            start_time = time.time()
            
            with torch.no_grad():
                _ = meta_model(current_phrases, current_phrases)
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        results[phrase_count] = {
            "avg_time": avg_time,
            "std_time": std_time,
            "times": times,
            "matrix_size": (phrase_count, phrase_count),
            "total_comparisons": phrase_count * phrase_count
        }
        
        print(f"  Среднее время: {avg_time:.4f} ± {std_time:.4f} сек")
        print(f"  Время на сравнение: {avg_time / (phrase_count * phrase_count) * 1000:.2f} мс")
    
    return results


def visualize_performance_benchmark(
    benchmark_results: Dict[str, Any],
    save_path: str = None,
    show_plot: bool = True
):
    """
    Визуализация результатов бенчмарка
    
    Args:
        benchmark_results: результаты от benchmark_meta_model_performance
        save_path: путь для сохранения графика
        show_plot: показать ли график
    """
    phrase_counts = list(benchmark_results.keys())
    avg_times = [benchmark_results[count]["avg_time"] for count in phrase_counts]
    std_times = [benchmark_results[count]["std_time"] for count in phrase_counts]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.errorbar(phrase_counts, avg_times, yerr=std_times, 
                marker='o', capsize=5, capthick=2, linewidth=2)
    plt.xlabel('Количество фраз')
    plt.ylabel('Время (секунды)')
    plt.title('Время обработки vs Количество фраз')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    time_per_comparison = [avg_times[i] / (phrase_counts[i] ** 2) * 1000 
                          for i in range(len(phrase_counts))]
    plt.plot(phrase_counts, time_per_comparison, 'r-', marker='o', linewidth=2)
    plt.xlabel('Количество фраз')
    plt.ylabel('Время на сравнение (мс)')
    plt.title('Время на одно сравнение')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График сохранен в {save_path}")
    
    if show_plot:
        plt.show()


def convert_cross_attention_results_to_meta_format(
    terms_file: str,
    prediction_matrix_file: str,
    threshold: float = 0.5
) -> List[Dict[str, str]]:
    """
    Конвертация результатов CrossAttention в формат для мета-модели
    
    Args:
        terms_file: файл с терминами
        prediction_matrix_file: файл с матрицей предсказаний
        threshold: порог для бинаризации
        
    Returns:
        relationships: список отношений
    """
    # Загружаем термины
    with open(terms_file, 'r', encoding='utf-8') as f:
        terms = [line.strip() for line in f.readlines()]
    
    # Загружаем матрицу предсказаний
    prediction_matrix = np.load(prediction_matrix_file)
    
    # Применяем порог и извлекаем отношения
    binary_matrix = (prediction_matrix > threshold).astype(int)
    
    relationships = []
    for i in range(len(terms)):
        for j in range(len(terms)):
            if binary_matrix[i, j] and i != j:  # Исключаем самозацикливания
                relationships.append({
                    "child": terms[i],
                    "parent": terms[j],
                    "confidence": float(prediction_matrix[i, j])
                })
    
    return relationships


def save_evaluation_report(
    results: Dict[str, Any],
    meta_model_info: Dict[str, Any],
    save_path: str
):
    """
    Сохранение отчета об оценке в JSON
    
    Args:
        results: результаты оценки
        meta_model_info: информация о мета-модели
        save_path: путь для сохранения отчета
    """
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "meta_model_info": meta_model_info,
        "evaluation_results": results,
        "summary": {
            "best_threshold": results["best_threshold"],
            "best_f1_score": results["best_metrics"]["f1_score"],
            "best_accuracy": results["best_metrics"]["accuracy"],
            "best_precision": results["best_metrics"]["precision"],
            "best_recall": results["best_metrics"]["recall"],
            "roc_auc": results["roc_auc"]
        }
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"Отчет сохранен в {save_path}")


if __name__ == "__main__":
    # Пример использования утилит
    print("🔧 Тестирование утилит для мета-модели...")
    
    # Создаем тестовые примеры
    test_examples = [
        {
            "phrases_1": ["machine learning", "deep learning"],
            "phrases_2": ["artificial intelligence", "computer science"],
            "labels": [[0.8, 0.9], [0.7, 0.8]]
        },
        {
            "phrases_1": ["neural networks", "algorithms"],
            "phrases_2": ["programming", "mathematics"],
            "labels": [[0.6, 0.7], [0.5, 0.9]]
        }
    ]
    
    print("✅ Утилиты готовы к использованию!") 