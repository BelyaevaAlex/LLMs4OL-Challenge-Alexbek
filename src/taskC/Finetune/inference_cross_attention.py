"""
Скрипт инференса для Cross-Attention модели
Применяет обученную модель к новым данным для построения иерархий терминов
"""

import os
import json
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModel
import argparse
from pathlib import Path
import logging
from datetime import datetime
import tqdm
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Импорт модели и утилит
from cross_attention_model import CrossAttentionModel


def setup_logging(log_file: str = None) -> logging.Logger:
    """
    Настройка логирования
    
    Args:
        log_file: путь к файлу лога
        
    Returns:
        logger: настроенный логгер
    """
    logger = logging.getLogger("cross_attention_inference")
    logger.setLevel(logging.INFO)
    
    # Очищаем существующие handlers
    logger.handlers.clear()
    
    # Форматтер для логов
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler для консоли
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler для файла если указан
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_embedding_model():
    """
    Загрузка модели для создания эмбеддингов
    
    Returns:
        model, tokenizer: модель и токенизатор для эмбеддингов
    """
    model_name = "Qwen/Qwen3-Embedding-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    
    return model, tokenizer


def get_term_embeddings(terms: List[str], model, tokenizer, batch_size: int = 32) -> List[List[float]]:
    """
    Генерация эмбеддингов для списка терминов
    
    Args:
        terms: список терминов
        model: модель для эмбеддингов
        tokenizer: токенизатор
        batch_size: размер батча
        
    Returns:
        embeddings: список эмбеддингов
    """
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(terms), batch_size), desc="Generating embeddings"):
            batch_terms = terms[i:i+batch_size]
            
            # Токенизация батча
            inputs = tokenizer(
                batch_terms, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Получение эмбеддингов
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
            embeddings.extend(batch_embeddings.tolist())
    
    return embeddings


def load_trained_model(results_dir: str) -> Tuple[CrossAttentionModel, Dict, float]:
    """
    Загрузка обученной модели и информации о лучших результатах
    
    Args:
        results_dir: директория с результатами обучения
        
    Returns:
        model: загруженная модель
        best_results: информация о лучших результатах
        f1_threshold: лучший порог для F1 score
    """
    results_path = Path(results_dir)
    
    # Загрузка информации о лучших результатах
    best_results_file = results_path / "best_results.json"
    if not best_results_file.exists():
        raise FileNotFoundError(f"best_results.json not found in {results_dir}")
    
    with open(best_results_file, 'r') as f:
        best_results = json.load(f)
    
    # Получение лучшего порога для F1
    f1_threshold = best_results['best_results']['best_thresholds']['f1_score']['threshold']
    
    # Загрузка информации о лучших моделях
    best_models_file = results_path / "best_models" / "best_models.json"
    if not best_models_file.exists():
        raise FileNotFoundError(f"best_models.json not found in {results_dir}/best_models/")
    
    with open(best_models_file, 'r') as f:
        best_models_info = json.load(f)
    
    # Выбираем лучшую модель (первую в списке)
    best_model_info = best_models_info['best_models'][0]
    model_path = best_model_info['model_path']
    
    if not os.path.exists(model_path):
        # Пробуем относительный путь
        model_filename = best_model_info['model_filename']
        model_path = results_path / "best_models" / model_filename
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Загрузка модели с поддержкой PyTorch 2.6
    try:
        # Сначала пытаемся загрузить безопасно
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    except Exception:
        # Если не получается, используем старый способ (для старых моделей)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Создание модели
    config = checkpoint['config']
    # Убеждаемся что config - это словарь
    if not isinstance(config, dict):
        if hasattr(config, 'to_dict'):
            config = config.to_dict()
        else:
            # Если это объект, преобразуем его атрибуты в словарь
            config = {k: v for k, v in config.__dict__.items() 
                     if not k.startswith('_')}
    
    model = CrossAttentionModel(config, layer_idx=checkpoint.get('layer_idx', 0))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    model.eval()
    
    return model, best_results, f1_threshold


def build_prediction_matrix(
    embeddings: List[List[float]], 
    model: CrossAttentionModel, 
    batch_size: int = 64,
    logger: logging.Logger = None
) -> np.ndarray:
    """
    Построение полной матрицы предсказаний terms x terms
    
    Args:
        embeddings: список эмбеддингов терминов
        model: обученная модель
        batch_size: размер батча для обработки
        logger: логгер для вывода прогресса
        
    Returns:
        pred_matrix: матрица предсказаний размера (n_terms, n_terms)
    """
    n_terms = len(embeddings)
    pred_matrix = np.zeros((n_terms, n_terms), dtype=np.float32)
    
    # Преобразуем эмбеддинги в тензор
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    if torch.cuda.is_available():
        embeddings_tensor = embeddings_tensor.cuda()
    
    if logger:
        logger.info(f"Строим матрицу предсказаний {n_terms}x{n_terms}")
    
    with torch.no_grad():
        # Обрабатываем по блокам для экономии памяти
        for i in tqdm.tqdm(range(0, n_terms, batch_size), desc="Building prediction matrix"):
            end_i = min(i + batch_size, n_terms)
            batch_embeddings_1 = embeddings_tensor[i:end_i]  # (batch_size, embedding_dim)
            
            # Для каждого блока строк вычисляем предсказания со всеми терминами
            for j in range(0, n_terms, batch_size):
                end_j = min(j + batch_size, n_terms)
                batch_embeddings_2 = embeddings_tensor[j:end_j]  # (batch_size, embedding_dim)
                
                # Вычисляем предсказания для блока
                # batch_embeddings_1: (batch_i, embedding_dim)
                # batch_embeddings_2: (batch_j, embedding_dim)
                # Нужно получить: (batch_i, batch_j)
                
                batch_pred = model(batch_embeddings_1, batch_embeddings_2)
                pred_matrix[i:end_i, j:end_j] = batch_pred.cpu().numpy()
    
    return pred_matrix


def apply_threshold_and_extract_pairs(
    pred_matrix: np.ndarray,
    terms: List[str],
    threshold: float,
    logger: logging.Logger = None
) -> List[Dict[str, str]]:
    """
    Применение порога к матрице предсказаний и извлечение пар child-parent
    
    Args:
        pred_matrix: матрица предсказаний
        terms: список терминов
        threshold: порог для бинаризации
        logger: логгер
        
    Returns:
        pairs: список пар в формате [{"parent": "...", "child": "..."}]
    """
    pairs = []
    
    # Применяем порог
    binary_matrix = (pred_matrix > threshold).astype(int)
    
    # Извлекаем пары (i, j) где binary_matrix[i, j] = 1
    # i - индекс child, j - индекс parent
    child_indices, parent_indices = np.where(binary_matrix == 1)
    
    for child_idx, parent_idx in zip(child_indices, parent_indices):
        # Пропускаем самосвязи
        if child_idx == parent_idx:
            continue
            
        pairs.append({
            "parent": terms[parent_idx],
            "child": terms[child_idx]
        })
    
    if logger:
        logger.info(f"Извлечено {len(pairs)} пар с порогом {threshold}")
        
        # Статистика по матрице
        total_pairs = len(terms) * (len(terms) - 1)  # Без диагонали
        positive_pairs = len(pairs)
        logger.info(f"Всего возможных пар: {total_pairs}")
        logger.info(f"Положительных пар: {positive_pairs} ({positive_pairs/total_pairs*100:.2f}%)")
    
    return pairs


def save_results(
    pred_matrix: np.ndarray,
    pairs: List[Dict[str, str]],
    terms: List[str],
    output_dir: str,
    threshold: float,
    best_results: Dict,
    logger: logging.Logger = None
):
    """
    Сохранение результатов инференса
    
    Args:
        pred_matrix: матрица предсказаний
        pairs: пары child-parent
        terms: список терминов
        output_dir: директория для сохранения
        threshold: использованный порог
        best_results: информация о лучших результатах
        logger: логгер
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Сохранение матрицы предсказаний
    matrix_file = output_path / "prediction_matrix.npy"
    np.save(matrix_file, pred_matrix)
    
    # Сохранение пар в JSON
    pairs_file = output_path / "predicted_pairs.json"
    with open(pairs_file, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    
    # Сохранение терминов
    terms_file = output_path / "terms.json"
    with open(terms_file, 'w', encoding='utf-8') as f:
        json.dump(terms, f, ensure_ascii=False, indent=2)
    
    # Сохранение метаданных
    metadata = {
        "inference_info": {
            "timestamp": datetime.now().isoformat(),
            "n_terms": len(terms),
            "n_pairs": len(pairs),
            "threshold_used": threshold,
            "matrix_shape": pred_matrix.shape,
            "matrix_stats": {
                "min": float(pred_matrix.min()),
                "max": float(pred_matrix.max()),
                "mean": float(pred_matrix.mean()),
                "std": float(pred_matrix.std())
            }
        },
        "model_info": {
            "best_f1_threshold": best_results['best_results']['best_thresholds']['f1_score']['threshold'],
            "best_f1_value": best_results['best_results']['best_thresholds']['f1_score']['value'],
            "model_roc_auc": best_results['best_results']['roc_auc'],
            "dataset": best_results['experiment_info']['dataset']
        }
    }
    
    metadata_file = output_path / "inference_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # Визуализация распределения предсказаний
    plot_prediction_distribution(pred_matrix, threshold, output_path)
    
    if logger:
        logger.info(f"Результаты сохранены в {output_dir}")
        logger.info(f"  📊 Матрица предсказаний: {matrix_file}")
        logger.info(f"  🔗 Пары: {pairs_file}")
        logger.info(f"  📝 Термины: {terms_file}")
        logger.info(f"  📋 Метаданные: {metadata_file}")


def plot_prediction_distribution(pred_matrix: np.ndarray, threshold: float, output_dir: Path):
    """
    Построение графика распределения предсказаний
    
    Args:
        pred_matrix: матрица предсказаний
        threshold: использованный порог
        output_dir: директория для сохранения
    """
    plt.figure(figsize=(12, 8))
    
    # Убираем диагональ для анализа
    mask = np.eye(pred_matrix.shape[0], dtype=bool)
    off_diagonal = pred_matrix[~mask]
    
    # Гистограмма предсказаний
    plt.subplot(2, 2, 1)
    plt.hist(off_diagonal, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.3f}')
    plt.xlabel('Prediction Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Логарифмическая шкала
    plt.subplot(2, 2, 2)
    plt.hist(off_diagonal, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.3f}')
    plt.xlabel('Prediction Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Scores (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Кумулятивное распределение
    plt.subplot(2, 2, 3)
    sorted_scores = np.sort(off_diagonal)
    y_vals = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    plt.plot(sorted_scores, y_vals, linewidth=2)
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.3f}')
    plt.xlabel('Prediction Score')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution of Prediction Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Статистика
    plt.subplot(2, 2, 4)
    stats_text = f"""
    Matrix Statistics:
    
    Shape: {pred_matrix.shape}
    Total pairs: {pred_matrix.shape[0] * (pred_matrix.shape[1] - 1)}
    
    Score Distribution:
    Min: {off_diagonal.min():.4f}
    Max: {off_diagonal.max():.4f}
    Mean: {off_diagonal.mean():.4f}
    Std: {off_diagonal.std():.4f}
    
    Threshold: {threshold:.3f}
    Above threshold: {np.sum(off_diagonal > threshold)} ({np.sum(off_diagonal > threshold)/len(off_diagonal)*100:.2f}%)
    Below threshold: {np.sum(off_diagonal <= threshold)} ({np.sum(off_diagonal <= threshold)/len(off_diagonal)*100:.2f}%)
    """
    
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=10, fontfamily='monospace')
    plt.axis('off')
    
    plt.tight_layout()
    plot_path = output_dir / 'prediction_distribution.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 График распределения предсказаний сохранен: {plot_path}")


def load_existing_inference_results(results_dir: str) -> Tuple[np.ndarray, List[str], Dict, float]:
    """
    Загрузка уже готовых результатов инференса
    
    Args:
        results_dir: папка с готовыми результатами инференса
        
    Returns:
        pred_matrix: матрица предсказаний
        terms: список терминов
        metadata: метаданные инференса
        original_threshold: изначально использованный порог
    """
    results_path = Path(results_dir)
    
    # Загрузка матрицы
    matrix_file = results_path / "prediction_matrix.npy"
    if not matrix_file.exists():
        raise FileNotFoundError(f"Матрица предсказаний не найдена: {matrix_file}")
    
    pred_matrix = np.load(matrix_file)
    
    # Загрузка терминов
    terms_file = results_path / "terms.json"
    if not terms_file.exists():
        raise FileNotFoundError(f"Файл с терминами не найден: {terms_file}")
    
    with open(terms_file, 'r', encoding='utf-8') as f:
        terms = json.load(f)
    
    # Загрузка метаданных
    metadata_file = results_path / "inference_metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Метаданные не найдены: {metadata_file}")
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    original_threshold = metadata['inference_info']['threshold_used']
    
    return pred_matrix, terms, metadata, original_threshold


def run_threshold_experiment(
    existing_results_dir: str,
    new_threshold: float,
    output_dir: str,
    output_suffix: str = None,
    log_file: str = None
) -> Dict:
    """
    Применение нового порога к уже готовой матрице предсказаний
    
    Args:
        existing_results_dir: папка с готовыми результатами инференса
        new_threshold: новый порог для извлечения пар
        output_dir: папка для сохранения новых результатов
        output_suffix: суффикс для имен файлов (например, "_threshold_05")
        log_file: файл лога
        
    Returns:
        results: словарь с результатами
    """
    # Настройка логирования
    logger = setup_logging(log_file)
    
    logger.info("🎯 Запуск эксперимента с новым порогом")
    logger.info(f"📁 Готовые результаты: {existing_results_dir}")
    logger.info(f"🎚️ Новый порог: {new_threshold}")
    logger.info(f"💾 Выходная папка: {output_dir}")
    
    try:
        # 1. Загрузка готовых результатов
        logger.info("📖 Загрузка готовых результатов...")
        pred_matrix, terms, metadata, original_threshold = load_existing_inference_results(existing_results_dir)
        
        logger.info(f"✅ Результаты загружены:")
        logger.info(f"   Матрица: {pred_matrix.shape}")
        logger.info(f"   Термины: {len(terms)}")
        logger.info(f"   Исходный порог: {original_threshold:.3f}")
        logger.info(f"   Новый порог: {new_threshold:.3f}")
        
        # 2. Применение нового порога
        logger.info("🎯 Применение нового порога...")
        pairs = apply_threshold_and_extract_pairs(pred_matrix, terms, new_threshold, logger)
        
        # 3. Создание обновленных метаданных
        updated_metadata = metadata.copy()
        updated_metadata['threshold_experiment'] = {
            'original_threshold': original_threshold,
            'new_threshold': new_threshold,
            'original_results_dir': existing_results_dir,
            'experiment_timestamp': datetime.now().isoformat()
        }
        updated_metadata['inference_info']['threshold_used'] = new_threshold
        
        # 4. Сохранение результатов с новым порогом
        logger.info("💾 Сохранение результатов с новым порогом...")
        
        # Определяем суффикс для файлов
        if output_suffix is None:
            output_suffix = f"_threshold_{new_threshold:.3f}".replace(".", "")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Сохранение пар
        pairs_file = output_path / f"predicted_pairs{output_suffix}.json"
        with open(pairs_file, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)
        
        # Сохранение терминов (копируем)
        terms_file = output_path / f"terms{output_suffix}.json"
        with open(terms_file, 'w', encoding='utf-8') as f:
            json.dump(terms, f, ensure_ascii=False, indent=2)
        
        # Сохранение матрицы (создаем символическую ссылку или копируем)
        matrix_file = output_path / f"prediction_matrix{output_suffix}.npy"
        if not matrix_file.exists():
            original_matrix_file = Path(existing_results_dir) / "prediction_matrix.npy"
            try:
                # Пытаемся создать символическую ссылку (экономит место)
                matrix_file.symlink_to(original_matrix_file.absolute())
                logger.info(f"   Создана символическая ссылка на матрицу: {matrix_file}")
            except (OSError, NotImplementedError):
                # Если не получается, копируем файл
                import shutil
                shutil.copy2(original_matrix_file, matrix_file)
                logger.info(f"   Скопирована матрица: {matrix_file}")
        
        # Сохранение обновленных метаданных
        metadata_file = output_path / f"inference_metadata{output_suffix}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(updated_metadata, f, ensure_ascii=False, indent=2)
        
        # Визуализация распределения с новым порогом
        plot_prediction_distribution(pred_matrix, new_threshold, output_path)
        
        logger.info("✅ Эксперимент с порогом завершен успешно!")
        logger.info(f"📊 Результаты:")
        logger.info(f"   Исходный порог: {original_threshold:.3f}")
        logger.info(f"   Новый порог: {new_threshold:.3f}")
        logger.info(f"   Исходное количество пар: {len(json.load(open(Path(existing_results_dir) / 'predicted_pairs.json')))}")
        logger.info(f"   Новое количество пар: {len(pairs)}")
        logger.info(f"   Разница: {len(pairs) - len(json.load(open(Path(existing_results_dir) / 'predicted_pairs.json')))}")
        logger.info(f"   Папка с результатами: {output_dir}")
        
        # Возвращаем результаты
        return {
            'pred_matrix': pred_matrix,
            'pairs': pairs,
            'terms': terms,
            'threshold': new_threshold,
            'original_threshold': original_threshold,
            'metadata': updated_metadata,
            'statistics': {
                'total_terms': len(terms),
                'total_pairs': len(pairs),
                'original_pairs': len(json.load(open(Path(existing_results_dir) / 'predicted_pairs.json'))),
                'pairs_difference': len(pairs) - len(json.load(open(Path(existing_results_dir) / 'predicted_pairs.json'))),
                'matrix_shape': pred_matrix.shape,
                'matrix_min': float(pred_matrix.min()),
                'matrix_max': float(pred_matrix.max()),
                'matrix_mean': float(pred_matrix.mean()),
                'matrix_std': float(pred_matrix.std())
            },
            'output_dir': output_dir,
            'files_created': {
                'pairs': str(pairs_file),
                'terms': str(terms_file),
                'matrix': str(matrix_file),
                'metadata': str(metadata_file),
                'distribution_plot': str(output_path / 'prediction_distribution.png')
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Ошибка во время эксперимента с порогом: {e}")
        raise


def run_inference(
    results_dir: str,
    terms_file: str,
    output_dir: str,
    embedding_batch_size: int = 32,
    prediction_batch_size: int = 64,
    custom_threshold: float = None,
    log_file: str = None,
    # Новые параметры для работы с готовой матрицей
    existing_results_dir: str = None,
    threshold_only: bool = False
) -> Dict:
    """
    Основная функция инференса для использования в Jupyter notebook
    
    Args:
        results_dir: Directory with training results (only needed if existing_results_dir is None)
        terms_file: Path to .txt file with terms (only needed if existing_results_dir is None)
        output_dir: Directory to save inference results
        embedding_batch_size: Batch size for embedding generation
        prediction_batch_size: Batch size for prediction matrix construction
        custom_threshold: Custom threshold (if None, uses best F1 threshold from training)
        log_file: Path to log file (optional)
        existing_results_dir: Directory with existing inference results (optional)
        threshold_only: Whether to return only the threshold and statistics (optional)
        
    Returns:
        results: Dictionary with inference results and statistics
    """
    # Настройка логирования
    logger = setup_logging(log_file)
    
    # Режим 1: Работа с готовой матрицей (только новый порог)
    if existing_results_dir is not None:
        logger.info("🎯 Режим работы с готовой матрицей")
        
        # Загрузка обученной модели для получения порога (если нужно)
        if custom_threshold is None:
            logger.info("🔄 Загрузка модели для получения порога...")
            _, _, f1_threshold = load_trained_model(results_dir)
            threshold = f1_threshold
        else:
            threshold = custom_threshold
        
        # Запуск эксперимента с порогом
        return run_threshold_experiment(
            existing_results_dir=existing_results_dir,
            new_threshold=threshold,
            output_dir=output_dir,
            log_file=log_file
        )
    
    # Режим 2: Полный инференс
    logger.info("🚀 Режим полного инференса Cross-Attention модели")
    logger.info(f"📁 Папка с результатами: {results_dir}")
    logger.info(f"📝 Файл с терминами: {terms_file}")
    logger.info(f"💾 Папка для вывода: {output_dir}")
    
    try:
        # 1. Загрузка обученной модели
        logger.info("🔄 Загрузка обученной модели...")
        model, best_results, f1_threshold = load_trained_model(results_dir)
        
        # Выбор порога
        threshold = custom_threshold if custom_threshold is not None else f1_threshold
        
        logger.info(f"✅ Модель загружена:")
        logger.info(f"   ROC AUC: {best_results['best_results']['roc_auc']:.4f}")
        logger.info(f"   Лучший F1 порог: {f1_threshold:.3f}")
        logger.info(f"   Используемый порог: {threshold:.3f}")
        
        # 2. Чтение терминов
        logger.info("📖 Чтение терминов...")
        with open(terms_file, 'r', encoding='utf-8') as f:
            terms = [line.strip() for line in f if line.strip()]
        
        logger.info(f"✅ Загружено {len(terms)} терминов")
        
        # 3. Генерация эмбеддингов
        logger.info("🔄 Загрузка модели для эмбеддингов...")
        embedding_model, tokenizer = load_embedding_model()
        
        logger.info("⚡ Генерация эмбеддингов...")
        embeddings = get_term_embeddings(terms, embedding_model, tokenizer, embedding_batch_size)
        
        logger.info(f"✅ Эмбеддинги созданы: {len(embeddings)} x {len(embeddings[0])}")
        
        # Освобождаем память от модели эмбеддингов
        del embedding_model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 4. Построение матрицы предсказаний
        logger.info("🔮 Построение матрицы предсказаний...")
        pred_matrix = build_prediction_matrix(embeddings, model, prediction_batch_size, logger)
        
        logger.info(f"✅ Матрица предсказаний построена: {pred_matrix.shape}")
        logger.info(f"   Статистика: min={pred_matrix.min():.4f}, max={pred_matrix.max():.4f}, mean={pred_matrix.mean():.4f}")
        
        # 5. Применение порога и извлечение пар
        logger.info("🎯 Применение порога и извлечение пар...")
        pairs = apply_threshold_and_extract_pairs(pred_matrix, terms, threshold, logger)
        
        # 6. Сохранение результатов
        logger.info("💾 Сохранение результатов...")
        save_results(pred_matrix, pairs, terms, output_dir, threshold, best_results, logger)
        
        logger.info("✅ Инференс завершен успешно!")
        logger.info(f"📊 Результаты:")
        logger.info(f"   Всего терминов: {len(terms)}")
        logger.info(f"   Найдено пар: {len(pairs)}")
        logger.info(f"   Использованный порог: {threshold:.3f}")
        logger.info(f"   Папка с результатами: {output_dir}")
        
        # Возвращаем результаты для дальнейшего анализа
        return {
            'pred_matrix': pred_matrix,
            'pairs': pairs,
            'terms': terms,
            'threshold': threshold,
            'best_results': best_results,
            'model_info': {
                'roc_auc': best_results['best_results']['roc_auc'],
                'f1_threshold': f1_threshold,
                'dataset': best_results['experiment_info']['dataset']
            },
            'statistics': {
                'total_terms': len(terms),
                'total_pairs': len(pairs),
                'matrix_shape': pred_matrix.shape,
                'matrix_min': float(pred_matrix.min()),
                'matrix_max': float(pred_matrix.max()),
                'matrix_mean': float(pred_matrix.mean()),
                'matrix_std': float(pred_matrix.std())
            },
            'output_dir': output_dir
        }
        
    except Exception as e:
        logger.error(f"❌ Ошибка во время инференса: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Cross-Attention Model Inference")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory with training results (e.g., results/20250707_192128_MatOnto_...)")
    parser.add_argument("--terms_file", type=str, required=True,
                        help="Path to .txt file with terms")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save inference results")
    parser.add_argument("--embedding_batch_size", type=int, default=32,
                        help="Batch size for embedding generation")
    parser.add_argument("--prediction_batch_size", type=int, default=64,
                        help="Batch size for prediction matrix construction")
    parser.add_argument("--custom_threshold", type=float, default=None,
                        help="Custom threshold (if not provided, uses best F1 threshold from training)")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Path to log file")
    parser.add_argument("--existing_results_dir", type=str, default=None,
                        help="Directory with existing inference results (optional)")
    parser.add_argument("--threshold_only", action="store_true",
                        help="Whether to return only the threshold and statistics")
    
    args = parser.parse_args()
    
    # Вызов основной функции
    results = run_inference(
        results_dir=args.results_dir,
        terms_file=args.terms_file,
        output_dir=args.output_dir,
        embedding_batch_size=args.embedding_batch_size,
        prediction_batch_size=args.prediction_batch_size,
        custom_threshold=args.custom_threshold,
        log_file=args.log_file,
        existing_results_dir=args.existing_results_dir,
        threshold_only=args.threshold_only
    )
    
    # Вывод краткой сводки
    print(f"\n📈 Краткая сводка:")
    print(f"   Терминов: {results['statistics']['total_terms']}")
    print(f"   Пар: {results['statistics']['total_pairs']}")
    print(f"   Порог: {results['threshold']:.3f}")
    print(f"   ROC AUC модели: {results['model_info']['roc_auc']:.4f}")
    print(f"   Результаты: {results['output_dir']}")


if __name__ == "__main__":
    main() 