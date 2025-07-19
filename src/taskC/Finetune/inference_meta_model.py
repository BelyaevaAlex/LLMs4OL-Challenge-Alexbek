#!/usr/bin/env python3
"""
Инференс для Meta-Model
Лаконичная функция для получения полной матрицы предсказаний из списка терминов
"""

import json
import numpy as np
import torch
from typing import Dict, List, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm

from meta_model import Qwen3CrossAttentionMetaModel


def run_meta_model_inference(
    terms: List[str],
    model: Qwen3CrossAttentionMetaModel,
    batch_size: int = 32,
    device: str = "auto",
    save_path: str = None,
) -> Dict:
    """
    Запуск инференса Meta-Model на списке терминов
    
    Args:
        terms: список терминов для инференса
        model: обученная Meta-Model
        batch_size: размер батча
        device: устройство для вычислений
        
    Returns:
        results: {
            'prediction_matrix': np.ndarray,  # Полная матрица предсказаний terms x terms
            'terms': List[str],               # Исходные термины
            'statistics': Dict                # Статистика
        }
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    n_terms = len(terms)
    pred_matrix = np.zeros((n_terms, n_terms), dtype=np.float32)
    
    model.eval()
    
    print(f"🚀 Запуск инференса: {n_terms} терминов, batch_size={batch_size}")
    
    with torch.no_grad():
        # Обрабатываем по блокам для экономии памяти
        for i in tqdm(range(0, n_terms, batch_size), desc="Inference"):
            end_i = min(i + batch_size, n_terms)
            batch_terms_1 = terms[i:end_i]
            
            for j in range(0, n_terms, batch_size):
                end_j = min(j + batch_size, n_terms)
                batch_terms_2 = terms[j:end_j]
                
                # Получаем предсказания для блока
                batch_pred = model(batch_terms_1, batch_terms_2)
                pred_matrix[i:end_i, j:end_j] = batch_pred.cpu().numpy()
    
    # Статистика
    statistics = {
        'num_terms': n_terms,
        'matrix_shape': pred_matrix.shape,
        'matrix_min': float(pred_matrix.min()),
        'matrix_max': float(pred_matrix.max()),
        'matrix_mean': float(pred_matrix.mean()),
        'matrix_std': float(pred_matrix.std())
    }
    
    print(f"✅ Инференс завершен:")
    print(f"   Матрица: {pred_matrix.shape}")
    print(f"   Статистика: min={statistics['matrix_min']:.4f}, max={statistics['matrix_max']:.4f}, mean={statistics['matrix_mean']:.4f}")
    
    results = {
        'prediction_matrix': pred_matrix,
        'terms': terms,
        'statistics': statistics
    }
    
    if save_path:
        save_results(results, save_path)
    
    return results


def threshold_analysis(
    pred_matrix: np.ndarray,
    gt_relationships: List[Dict],
    terms: List[str],
    thresholds: Optional[List[float]] = None
) -> Dict:
    """
    Анализ различных порогов для матрицы предсказаний
    
    Args:
        pred_matrix: матрица предсказаний
        gt_relationships: список отношений {"parent": "...", "child": "..."}
        terms: список терминов
        thresholds: пороги для анализа
        
    Returns:
        analysis: результаты анализа
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 0.95, 0.01)
    
    # Создаем ground truth матрицу
    term_to_idx = {term: i for i, term in enumerate(terms)}
    gt_matrix = np.zeros_like(pred_matrix)
    
    for rel in gt_relationships:
        parent = rel.get('parent')
        child = rel.get('child')
        if parent in term_to_idx and child in term_to_idx:
            child_idx = term_to_idx[child]
            parent_idx = term_to_idx[parent]
            gt_matrix[child_idx, parent_idx] = 1
    
    # Убираем диагональ для анализа
    mask = np.eye(pred_matrix.shape[0], dtype=bool)
    pred_flat = pred_matrix[~mask]
    gt_flat = gt_matrix[~mask]
    
    results = {
        'thresholds': thresholds.tolist(),
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    for threshold in thresholds:
        pred_binary = (pred_flat > threshold).astype(int)
        
        # Метрики
        tp = np.sum((pred_binary == 1) & (gt_flat == 1))
        fp = np.sum((pred_binary == 1) & (gt_flat == 0))
        tn = np.sum((pred_binary == 0) & (gt_flat == 0))
        fn = np.sum((pred_binary == 0) & (gt_flat == 1))
        
        acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
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
    
    return {
        'threshold_analysis': results,
        'best_thresholds': best_thresholds,
        'best_values': best_values,
        'gt_stats': {
            'total_relationships': len(gt_relationships),
            'positive_pairs': int(gt_flat.sum()),
            'negative_pairs': int((gt_flat == 0).sum()),
            'positive_ratio': float(gt_flat.sum() / len(gt_flat))
        }
    }


def plot_prediction_distribution(
    pred_matrix: np.ndarray,
    thresholds: List[float] = None,
    save_path: str = None
) -> str:
    """
    Визуализация распределения предсказаний и доли ответов больше порогов
    
    Args:
        pred_matrix: матрица предсказаний
        thresholds: пороги для анализа
        save_path: путь для сохранения
        
    Returns:
        plot_path: путь к сохраненному графику
    """
    if thresholds is None:
        thresholds = [0.05, 0.15, 0.25, 0.5, 0.75]
    
    # Убираем диагональ для анализа
    mask = np.eye(pred_matrix.shape[0], dtype=bool)
    off_diagonal = pred_matrix[~mask]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # График 1: Гистограмма распределения
    ax1.hist(off_diagonal, bins=50, alpha=0.7, edgecolor='black')
    for thr in thresholds:
        ax1.axvline(thr, color='red', linestyle='--', alpha=0.7, label=f'{thr:.2f}')
    ax1.set_xlabel('Prediction Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Prediction Scores')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График 2: Логарифмическая шкала
    ax2.hist(off_diagonal, bins=50, alpha=0.7, edgecolor='black')
    for thr in thresholds:
        ax2.axvline(thr, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Prediction Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution (Log Scale)')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # График 3: Доля ответов больше порогов
    threshold_range = np.arange(0.01, 0.95, 0.01)
    ratios = []
    for thr in threshold_range:
        ratio = np.sum(off_diagonal > thr) / len(off_diagonal)
        ratios.append(ratio)
    
    ax3.plot(threshold_range, ratios, 'b-', linewidth=2)
    for thr in thresholds:
        ratio = np.sum(off_diagonal > thr) / len(off_diagonal)
        ax3.axvline(thr, color='red', linestyle='--', alpha=0.7)
        ax3.text(thr, ratio + 0.02, f'{ratio:.3f}', ha='center', va='bottom')
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('Ratio of Predictions > Threshold')
    ax3.set_title('Ratio of Predictions Above Threshold')
    ax3.grid(True, alpha=0.3)
    
    # График 4: Статистика
    stats_text = f"""
Matrix Statistics:

Shape: {pred_matrix.shape}
Total pairs: {pred_matrix.shape[0] * (pred_matrix.shape[1] - 1)}

Score Distribution:
Min: {off_diagonal.min():.4f}
Max: {off_diagonal.max():.4f}
Mean: {off_diagonal.mean():.4f}
Std: {off_diagonal.std():.4f}

Threshold Analysis:
"""
    for thr in thresholds:
        ratio = np.sum(off_diagonal > thr) / len(off_diagonal)
        stats_text += f">{thr:.2f}: {ratio:.3f} ({np.sum(off_diagonal > thr):,} pairs)\n"
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
             verticalalignment='top', fontsize=9, fontfamily='monospace')
    ax4.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plot_path = save_path
    else:
        plot_path = 'prediction_distribution.png'
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 График сохранен: {plot_path}")
    return plot_path


def extract_relationships_from_matrix(
    pred_matrix: np.ndarray,
    terms: List[str],
    threshold: float,
    remove_self_loops: bool = True,
    min_confidence: float = None,
    max_relationships: int = None
) -> List[Dict[str, str]]:
    """
    Извлечение relationships из матрицы предсказаний в формате TaskC
    
    Args:
        pred_matrix: матрица предсказаний размера (n_terms, n_terms)
        terms: список терминов соответствующий строкам/столбцам матрицы
        threshold: порог для извлечения отношений
        remove_self_loops: убрать диагональные элементы (self-loops)
        min_confidence: минимальная уверенность (дополнительная фильтрация)
        max_relationships: максимальное количество отношений (топ по confidence)
        
    Returns:
        relationships: список отношений в формате [{"ID": "...", "parent": "...", "child": "..."}, ...]
        
    Note:
        pred_matrix[i][j] > threshold означает что terms[i] является child термина terms[j] (parent)
    """
    import uuid
    
    n_terms = len(terms)
    relationships = []
    
    print(f"🔍 Извлечение relationships: threshold={threshold:.4f}")
    
    # Собираем все кандидаты с их confidence
    candidates = []
    
    for i in range(n_terms):
        for j in range(n_terms):
            # Пропускаем диагональ если нужно
            if remove_self_loops and i == j:
                continue
                
            confidence = float(pred_matrix[i, j])
            
            # Проверяем threshold
            if confidence > threshold:
                # Дополнительная фильтрация по min_confidence
                if min_confidence is not None and confidence < min_confidence:
                    continue
                    
                candidates.append({
                    'child_idx': i,
                    'parent_idx': j,
                    'confidence': confidence,
                    'child': terms[i],
                    'parent': terms[j]
                })
    
    # Сортируем по убыванию confidence
    candidates.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Ограничиваем количество если нужно
    if max_relationships is not None:
        candidates = candidates[:max_relationships]
    
    # Создаем relationships в нужном формате
    for idx, candidate in enumerate(candidates):
        relationship_id = f"REL_{uuid.uuid4().hex[:8]}"
        
        relationship = {
            "ID": relationship_id,
            "parent": candidate['parent'],
            "child": candidate['child']
        }
        
        relationships.append(relationship)
    
    print(f"✅ Извлечено {len(relationships)} relationships")
    
    # Статистика
    if len(relationships) > 0:
        print(f"   Диапазон confidence: {candidates[0]['confidence']:.4f} - {candidates[-1]['confidence']:.4f}")
        confidences = [c['confidence'] for c in candidates]
        print(f"   Средняя confidence: {np.mean(confidences):.4f} ± {np.std(confidences):.4f}")
        
        # Уникальные термины
        unique_parents = set(r['parent'] for r in relationships)
        unique_children = set(r['child'] for r in relationships)
        print(f"   Уникальные parents: {len(unique_parents)}")
        print(f"   Уникальные children: {len(unique_children)}")
        
        # Топ-3 relationships
        print(f"   Топ-3 relationships:")
        for i, rel in enumerate(relationships[:3]):
            conf = candidates[i]['confidence']
            print(f"     {i+1}. {rel['child']} -> {rel['parent']} (conf: {conf:.4f})")
    
    return relationships


def save_relationships_to_json(
    relationships: List[Dict[str, str]],
    save_path: str,
    metadata: Dict = None
):
    """
    Сохранение relationships в JSON файл в формате TaskC
    
    Args:
        relationships: список relationships
        save_path: путь для сохранения
        metadata: дополнительные метаданные для записи в файл
    """
    save_path = Path(save_path)
    
    # Если указана папка, создаем имя файла
    if save_path.is_dir() or not save_path.suffix:
        save_path = save_path / "predicted_relationships.json"
    
    # Создаем папку если нужно
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Подготавливаем данные для сохранения
    output_data = relationships
    
    # Добавляем метаданные как комментарий в начало если есть
    if metadata:
        # JSON не поддерживает комментарии, но мы можем добавить метаданные в отдельный файл
        metadata_path = save_path.with_suffix('.metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"📊 Метаданные сохранены: {metadata_path}")
    
    # Сохраняем relationships
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Relationships сохранены: {save_path}")
    print(f"   Всего отношений: {len(relationships)}")
    
    return str(save_path)


def save_results(
    results: Dict,
    save_path: str,
    threshold_analysis: Dict = None,
    relationships: List[Dict] = None
):
    """Сохранение результатов инференса"""
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохранение матрицы
    np.save(save_dir / "prediction_matrix.npy", results['prediction_matrix'])
    
    # Сохранение терминов
    with open(save_dir / "terms.json", 'w', encoding='utf-8') as f:
        json.dump(results['terms'], f, ensure_ascii=False, indent=2)
    
    # Сохранение статистики
    metadata = {
        'statistics': results['statistics'],
        'threshold_analysis': threshold_analysis,
        'relationships': relationships
    }
    
    with open(save_dir / "results.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Результаты сохранены в {save_path}")


# Пример использования
if __name__ == "__main__":
    print("🚀 Лаконичный инференс Meta-Model")
    print("Пример:")
    print("""
    from dataset import MetaModelDataset
    from meta_model import Qwen3CrossAttentionMetaModel
    from inference_meta_model import (
        run_meta_model_inference, 
        threshold_analysis, 
        plot_prediction_distribution,
        extract_relationships_from_matrix,
        save_relationships_to_json
    )
    
    # Загружаем данные
    dataset = MetaModelDataset(terms_path="data/terms.txt", relations_path="data/relations.json")
    terms = dataset.get_all_terms()
    gt_relationships = dataset.get_all_relations()
    
    # Загружаем модель и запускаем инференс
    model = Qwen3CrossAttentionMetaModel.from_pretrained("path/to/best_model")
    results = run_meta_model_inference(terms, model, batch_size=32)
    pred_matrix = results['prediction_matrix']
    
    # Анализ порогов
    analysis = threshold_analysis(pred_matrix, gt_relationships, terms)
    best_threshold = analysis['best_thresholds']['f1']
    
    # Извлечение relationships с оптимальным порогом
    relationships = extract_relationships_from_matrix(
        pred_matrix=pred_matrix,
        terms=terms,
        threshold=best_threshold,
        remove_self_loops=True,
        max_relationships=10000  # ограничиваем количество если нужно
    )
    
    # Сохранение relationships в формате TaskC
    metadata = {
        'threshold': best_threshold,
        'num_terms': len(terms),
        'model_info': model.get_model_info(),
        'statistics': results['statistics'],
        'threshold_analysis': analysis
    }
    
    save_relationships_to_json(
        relationships=relationships,
        save_path="predicted_relationships.json",
        metadata=metadata
    )
    
    # Визуализация и сохранение полных результатов
    plot_prediction_distribution(pred_matrix, save_path="distribution.png")
    save_results(results, "inference_results/", analysis, relationships)
    
    print(f"✅ Готово! Извлечено {len(relationships)} relationships с порогом {best_threshold:.4f}")
    """) 