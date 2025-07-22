"""
Analysis of Cross-Attention model inference results
Creating statistics, visualizations and exports in different formats
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from pathlib import Path
import argparse
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
import networkx as nx
from datetime import datetime


def load_inference_results(results_dir: str) -> Tuple[np.ndarray, List[Dict], List[str], Dict]:
    """
    Load inference results
    
    Args:
        results_dir: directory with results
        
    Returns:
        pred_matrix: prediction matrix
        pairs: list of pairs
        terms: list of terms
        metadata: metadata
    """
    results_path = Path(results_dir)
    
    # Load matrix
    matrix_file = results_path / "prediction_matrix.npy"
    pred_matrix = np.load(matrix_file)
    
    # Load pairs
    pairs_file = results_path / "predicted_pairs.json"
    with open(pairs_file, 'r', encoding='utf-8') as f:
        pairs = json.load(f)
    
    # Load terms
    terms_file = results_path / "terms.json"
    with open(terms_file, 'r', encoding='utf-8') as f:
        terms = json.load(f)
    
    # Load metadata
    metadata_file = results_path / "inference_metadata.json"
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    return pred_matrix, pairs, terms, metadata


def analyze_hierarchy_structure(pairs: List[Dict], terms: List[str]) -> Dict:
    """
    Анализ структуры иерархии
    
    Args:
        pairs: список пар child-parent
        terms: список всех терминов
        
    Returns:
        analysis: результаты анализа
    """
    
    # Создаем граф
    G = nx.DiGraph()
    
    # Добавляем все термины как узлы
    G.add_nodes_from(terms)
    
    # Добавляем рёбра из пар
    for pair in pairs:
        G.add_edge(pair['child'], pair['parent'])
    
    # Подсчёт степеней узлов
    parent_counts = Counter()
    child_counts = Counter()
    
    for pair in pairs:
        parent_counts[pair['parent']] += 1
        child_counts[pair['child']] += 1
    
    # Классификация узлов
    roots = []  # Только родители, не дети
    leaves = []  # Только дети, не родители
    intermediate = []  # И родители, и дети
    isolated = []  # Ни родители, ни дети
    
    for term in terms:
        is_parent = term in parent_counts
        is_child = term in child_counts
        
        if is_parent and not is_child:
            roots.append(term)
        elif is_child and not is_parent:
            leaves.append(term)
        elif is_parent and is_child:
            intermediate.append(term)
        else:
            isolated.append(term)
    
    # Анализ связности
    connected_components = list(nx.weakly_connected_components(G))
    largest_component = max(connected_components, key=len) if connected_components else set()
    
    # Анализ циклов
    cycles = list(nx.simple_cycles(G))
    
    # Максимальная глубина
    max_depth = 0
    for root in roots:
        try:
            depths = nx.single_source_shortest_path_length(G, root)
            max_depth = max(max_depth, max(depths.values()) if depths else 0)
        except:
            pass
    
    analysis = {
        'basic_stats': {
            'total_terms': len(terms),
            'total_pairs': len(pairs),
            'unique_parents': len(parent_counts),
            'unique_children': len(child_counts),
            'density': len(pairs) / (len(terms) * (len(terms) - 1)) if len(terms) > 1 else 0
        },
        'node_classification': {
            'roots': len(roots),
            'leaves': len(leaves),
            'intermediate': len(intermediate),
            'isolated': len(isolated),
            'root_terms': roots[:10],  # Первые 10 для примера
            'leaf_terms': leaves[:10],
            'top_parents': parent_counts.most_common(10),
            'top_children': child_counts.most_common(10)
        },
        'connectivity': {
            'connected_components': len(connected_components),
            'largest_component_size': len(largest_component),
            'largest_component_ratio': len(largest_component) / len(terms) if terms else 0,
            'average_component_size': np.mean([len(c) for c in connected_components]) if connected_components else 0
        },
        'structure': {
            'cycles_count': len(cycles),
            'max_depth': max_depth,
            'cycles': cycles[:5]  # Первые 5 циклов для примера
        }
    }
    
    return analysis


def create_detailed_visualizations(
    pred_matrix: np.ndarray,
    pairs: List[Dict],
    terms: List[str],
    metadata: Dict,
    output_dir: str
):
    """
    Создание детальных визуализаций
    
    Args:
        pred_matrix: матрица предсказаний
        pairs: список пар
        terms: список терминов
        metadata: метаданные
        output_dir: папка для сохранения
    """
    output_path = Path(output_dir)
    
    # 1. Тепловая карта (для небольших матриц)
    if len(terms) <= 50:
        plt.figure(figsize=(12, 10))
        im = plt.imshow(pred_matrix, cmap='viridis', aspect='auto')
        
        # Настройка меток осей
        plt.xticks(range(len(terms)), terms, rotation=45, ha='right')
        plt.yticks(range(len(terms)), terms, rotation=0)
        
        # Добавление colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Prediction Score')
        
        plt.title('Prediction Matrix Heatmap')
        plt.xlabel('Parent (columns)')
        plt.ylabel('Child (rows)')
        plt.tight_layout()
        plt.savefig(output_path / 'prediction_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. График распределения по терминам
    plt.figure(figsize=(15, 10))
    
    # Подсчёт количества связей для каждого термина
    parent_counts = Counter()
    child_counts = Counter()
    
    for pair in pairs:
        parent_counts[pair['parent']] += 1
        child_counts[pair['child']] += 1
    
    # Топ родителей
    plt.subplot(2, 2, 1)
    top_parents = parent_counts.most_common(20)
    if top_parents:
        parents, counts = zip(*top_parents)
        plt.barh(range(len(parents)), counts)
        plt.yticks(range(len(parents)), parents)
        plt.xlabel('Number of Children')
        plt.title('Top 20 Parents (by number of children)')
        plt.gca().invert_yaxis()
    
    # Топ детей
    plt.subplot(2, 2, 2)
    top_children = child_counts.most_common(20)
    if top_children:
        children, counts = zip(*top_children)
        plt.barh(range(len(children)), counts)
        plt.yticks(range(len(children)), children)
        plt.xlabel('Number of Parents')
        plt.title('Top 20 Children (by number of parents)')
        plt.gca().invert_yaxis()
    
    # Распределение количества детей
    plt.subplot(2, 2, 3)
    if parent_counts:
        plt.hist(list(parent_counts.values()), bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Number of Children')
        plt.ylabel('Number of Parents')
        plt.title('Distribution of Children per Parent')
        plt.grid(True, alpha=0.3)
    
    # Распределение количества родителей
    plt.subplot(2, 2, 4)
    if child_counts:
        plt.hist(list(child_counts.values()), bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Number of Parents')
        plt.ylabel('Number of Children')
        plt.title('Distribution of Parents per Child')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'hierarchy_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Граф структуры (для небольших графов)
    if len(terms) <= 30:
        plt.figure(figsize=(15, 12))
        
        G = nx.DiGraph()
        G.add_nodes_from(terms)
        
        for pair in pairs:
            G.add_edge(pair['child'], pair['parent'])
        
        # Позиционирование узлов
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Размеры узлов пропорциональны количеству связей
        node_sizes = []
        for node in G.nodes():
            total_connections = G.in_degree(node) + G.out_degree(node)
            node_sizes.append(max(300, total_connections * 100))
        
        # Рисование
        nx.draw(G, pos, 
                node_size=node_sizes,
                node_color='lightblue',
                with_labels=True,
                font_size=8,
                font_weight='bold',
                arrows=True,
                arrowsize=20,
                edge_color='gray',
                alpha=0.7)
        
        plt.title('Hierarchy Graph Structure')
        plt.savefig(output_path / 'hierarchy_graph.png', dpi=300, bbox_inches='tight')
        plt.close()


def export_to_formats(pairs: List[Dict], terms: List[str], output_dir: str):
    """
    Экспорт в различные форматы
    
    Args:
        pairs: список пар
        terms: список терминов
        output_dir: папка для сохранения
    """
    output_path = Path(output_dir)
    
    # 1. CSV формат
    df = pd.DataFrame(pairs)
    df.to_csv(output_path / 'predicted_pairs.csv', index=False, encoding='utf-8')
    
    # 2. TSV формат
    df.to_csv(output_path / 'predicted_pairs.tsv', sep='\t', index=False, encoding='utf-8')
    
    # 3. GraphML формат для Gephi/Cytoscape
    G = nx.DiGraph()
    G.add_nodes_from(terms)
    
    for pair in pairs:
        G.add_edge(pair['child'], pair['parent'])
    
    nx.write_graphml(G, output_path / 'hierarchy_graph.graphml')
    
    # 4. DOT формат для Graphviz
    nx.write_dot(G, output_path / 'hierarchy_graph.dot')
    
    # 5. Adjacency matrix
    adj_matrix = nx.adjacency_matrix(G, nodelist=terms).toarray()
    np.savetxt(output_path / 'adjacency_matrix.csv', adj_matrix, delimiter=',', fmt='%d')
    
    # 6. Edge list
    with open(output_path / 'edge_list.txt', 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(f"{pair['child']}\t{pair['parent']}\n")


def generate_report(analysis: Dict, metadata: Dict, output_dir: str):
    """
    Генерация текстового отчёта
    
    Args:
        analysis: результаты анализа
        metadata: метаданные
        output_dir: папка для сохранения
    """
    output_path = Path(output_dir)
    
    report = f"""
# Отчёт по результатам инференса Cross-Attention модели

## Основная информация
- Дата инференса: {metadata['inference_info']['timestamp']}
- Размер матрицы: {metadata['inference_info']['matrix_shape']}
- Использованный порог: {metadata['inference_info']['threshold_used']:.3f}

## Статистика модели
- ROC AUC: {metadata['model_info']['model_roc_auc']:.4f}
- Лучший F1 порог: {metadata['model_info']['best_f1_threshold']:.3f}
- Лучший F1 score: {metadata['model_info']['best_f1_value']:.4f}
- Датасет: {metadata['model_info']['dataset']}

## Базовая статистика
- Всего терминов: {analysis['basic_stats']['total_terms']}
- Найдено пар: {analysis['basic_stats']['total_pairs']}
- Уникальных родителей: {analysis['basic_stats']['unique_parents']}
- Уникальных детей: {analysis['basic_stats']['unique_children']}
- Плотность связей: {analysis['basic_stats']['density']:.4f}

## Классификация узлов
- Корни (только родители): {analysis['node_classification']['roots']}
- Листья (только дети): {analysis['node_classification']['leaves']}
- Промежуточные (и родители, и дети): {analysis['node_classification']['intermediate']}
- Изолированные (без связей): {analysis['node_classification']['isolated']}

## Структура иерархии
- Связных компонент: {analysis['connectivity']['connected_components']}
- Размер наибольшей компоненты: {analysis['connectivity']['largest_component_size']}
- Доля в наибольшей компоненте: {analysis['connectivity']['largest_component_ratio']:.2%}
- Средний размер компоненты: {analysis['connectivity']['average_component_size']:.1f}

## Анализ структуры
- Найдено циклов: {analysis['structure']['cycles_count']}
- Максимальная глубина: {analysis['structure']['max_depth']}

## Топ родителей (по количеству детей)
"""
    
    for parent, count in analysis['node_classification']['top_parents']:
        report += f"- {parent}: {count} детей\n"
    
    report += "\n## Топ детей (по количеству родителей)\n"
    
    for child, count in analysis['node_classification']['top_children']:
        report += f"- {child}: {count} родителей\n"
    
    if analysis['structure']['cycles']:
        report += "\n## Обнаруженные циклы\n"
        for i, cycle in enumerate(analysis['structure']['cycles'], 1):
            report += f"{i}. {' -> '.join(cycle)} -> {cycle[0]}\n"
    
    report += f"\n## Примеры корневых терминов\n"
    for term in analysis['node_classification']['root_terms']:
        report += f"- {term}\n"
    
    report += f"\n## Примеры листовых терминов\n"
    for term in analysis['node_classification']['leaf_terms']:
        report += f"- {term}\n"
    
    # Сохранение отчёта
    with open(output_path / 'analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)


def analyze_results(
    results_dir: str,
    output_dir: str = None,
    create_visualizations: bool = False,
    export_formats: bool = False
) -> Dict:
    """
    Основная функция анализа результатов инференса для использования в Jupyter notebook
    
    Args:
        results_dir: Directory with inference results
        output_dir: Output directory for analysis (default: results_dir/analysis)
        create_visualizations: Create detailed visualizations
        export_formats: Export to multiple formats
        
    Returns:
        analysis_results: Dictionary with complete analysis results
    """
    # Определение выходной папки
    if output_dir is None:
        output_dir = f"{results_dir}/analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🔍 Анализ результатов инференса...")
    print(f"📁 Папка с результатами: {results_dir}")
    print(f"💾 Папка для анализа: {output_dir}")
    
    # Загрузка данных
    print("📖 Загрузка результатов...")
    pred_matrix, pairs, terms, metadata = load_inference_results(results_dir)
    
    print(f"✅ Данные загружены:")
    print(f"   Матрица: {pred_matrix.shape}")
    print(f"   Пары: {len(pairs)}")
    print(f"   Термины: {len(terms)}")
    
    # Анализ структуры
    print("🔍 Анализ структуры иерархии...")
    analysis = analyze_hierarchy_structure(pairs, terms)
    
    # Сохранение результатов анализа
    with open(f"{output_dir}/analysis_results.json", 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    
    # Генерация отчёта
    print("📋 Генерация отчёта...")
    generate_report(analysis, metadata, output_dir)
    
    # Создание визуализаций
    if create_visualizations:
        print("📊 Создание визуализаций...")
        create_detailed_visualizations(pred_matrix, pairs, terms, metadata, output_dir)
    
    # Экспорт в форматы
    if export_formats:
        print("💾 Экспорт в различные форматы...")
        export_to_formats(pairs, terms, output_dir)
    
    print("✅ Анализ завершён!")
    print(f"📁 Результаты сохранены в: {output_dir}")
    print(f"📋 Отчёт: {output_dir}/analysis_report.md")
    print(f"📊 Данные анализа: {output_dir}/analysis_results.json")
    
    # Краткая сводка
    print("\n📈 Краткая сводка:")
    print(f"   Терминов: {analysis['basic_stats']['total_terms']}")
    print(f"   Пар: {analysis['basic_stats']['total_pairs']}")
    print(f"   Корни: {analysis['node_classification']['roots']}")
    print(f"   Листья: {analysis['node_classification']['leaves']}")
    print(f"   Компоненты: {analysis['connectivity']['connected_components']}")
    print(f"   Циклы: {analysis['structure']['cycles_count']}")
    
    # Возвращаем полные результаты для дальнейшего анализа
    return {
        'pred_matrix': pred_matrix,
        'pairs': pairs,
        'terms': terms,
        'metadata': metadata,
        'analysis': analysis,
        'output_dir': output_dir,
        'files_created': {
            'analysis_json': f"{output_dir}/analysis_results.json",
            'report_md': f"{output_dir}/analysis_report.md",
            'visualizations': f"{output_dir}/*.png" if create_visualizations else None,
            'export_formats': [
                f"{output_dir}/predicted_pairs.csv",
                f"{output_dir}/predicted_pairs.tsv",
                f"{output_dir}/hierarchy_graph.graphml",
                f"{output_dir}/hierarchy_graph.dot",
                f"{output_dir}/adjacency_matrix.csv",
                f"{output_dir}/edge_list.txt"
            ] if export_formats else None
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze Cross-Attention Inference Results")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory with inference results")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for analysis (default: results_dir/analysis)")
    parser.add_argument("--create_visualizations", action="store_true",
                        help="Create detailed visualizations")
    parser.add_argument("--export_formats", action="store_true",
                        help="Export to multiple formats")
    
    args = parser.parse_args()
    
    # Вызов основной функции
    results = analyze_results(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        create_visualizations=args.create_visualizations,
        export_formats=args.export_formats
    )
    
    # Дополнительная информация о созданных файлах
    print(f"\n📋 Созданные файлы:")
    print(f"   📊 JSON с анализом: {results['files_created']['analysis_json']}")
    print(f"   📝 Отчёт: {results['files_created']['report_md']}")
    
    if results['files_created']['visualizations']:
        print(f"   📈 Визуализации: {results['files_created']['visualizations']}")
    
    if results['files_created']['export_formats']:
        print(f"   💾 Экспорт форматы: {len(results['files_created']['export_formats'])} файлов")
    
    return results


if __name__ == "__main__":
    main() 