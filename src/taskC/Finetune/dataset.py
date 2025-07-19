"""
MetaModelDataset - dataset for Meta-Model (Qwen3 + CrossAttention)
Works with text files of terms and relationship files,
returns two sets of texts and a relationship matrix.
"""

import json
import random
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Set, Optional, Any
from sklearn.model_selection import train_test_split
import numpy as np
from collections import defaultdict


class MetaModelDataset(Dataset):
    """
    Dataset for Meta-Model (Qwen3 + CrossAttention)

    Reads terms from txt files and relationships from json files,
    returns two sets of texts and a relationship matrix.

    Args:
        terms_path: path to terms file (*.txt) or tuple of two paths (terms_path, types_path)
        relations_path: path to relationships file (*.json)
        batch_size_1: size of first text set (children/queries)
        batch_size_2: size of second text set (parents/keys)
        dataset_strategy: dataset selection strategy ("random", "sequential", "weighted", "single")
        sampling_strategy: sampling strategy ("random", "balanced")
        positive_ratio: proportion of positive pairs for balanced strategy
        mode: operation mode ("train", "test", "all")
        test_part: proportion of test data
        random_state: seed for reproducibility
    """

    def __init__(
        self,
        terms_path: str | Tuple[str, str],
        relations_path: str,
        batch_size_1: int = 32,
        batch_size_2: int = 32,
        dataset_strategy: str = "single",
        sampling_strategy: str = "random",
        positive_ratio: float = 1.0,
        mode: str = "all",
        test_part: float = 0.2,
        random_state: int = 42
    ):
        self.terms_path = terms_path
        self.relations_path = relations_path
        self.batch_size_1 = batch_size_1
        self.batch_size_2 = batch_size_2
        self.dataset_strategy = dataset_strategy
        self.sampling_strategy = sampling_strategy
        self.positive_ratio = positive_ratio
        self.mode = mode
        self.test_part = test_part
        self.random_state = random_state

        # Set seed
        random.seed(random_state)
        np.random.seed(random_state)

        # Load data
        print("📁 Loading terms...")
        self.terms = self._load_terms(terms_path)

        # Create indices
        print("📊 Creating indices...")
        self.term_to_id = {term: i for i, term in enumerate(self.terms)}
        self.id_to_term = {i: term for i, term in enumerate(self.terms)}

        print("📁 Loading relationships...")
        self.relations = self._load_relations(relations_path)

        # Group by datasets (if there are several)
        print("🔗 Grouping data...")
        self.terms_by_dataset = self._group_terms_by_dataset()
        self.available_datasets = list(self.terms_by_dataset.keys())

        # Create relationship indices
        self.relations_by_dataset = self._group_relations_by_dataset()

        # Split into Train/Test if needed
        if mode in ["train", "test"]:
            print(f"✂️ Splitting into train/test ({mode} mode)...")
            self._split_train_test()
        else:
            # Build child -> parents index for fast search
            self.child_to_parents = self._build_child_to_parents_index()

        # Dataset selection logic
        self.current_dataset_idx = 0

        # For weighted dataset selection
        if self.dataset_strategy == "weighted":
            self.dataset_weights = self._calculate_dataset_weights()

        # If only one dataset, ignore selection strategy
        if len(self.available_datasets) == 1:
            self.dataset_strategy = "single"

        # Print statistics
        self._print_statistics()

    def _load_terms(self, terms_path: str | Tuple[str, str]) -> List[str]:
        """Load terms from txt file(s)"""
        if isinstance(terms_path, tuple):
            # Two files: terms and types
            terms_path_1, terms_path_2 = terms_path

            # Load terms from first file
            with open(terms_path_1, 'r', encoding='utf-8') as f:
                terms_1 = [line.strip() for line in f if line.strip()]

            # Load terms from second file
            with open(terms_path_2, 'r', encoding='utf-8') as f:
                terms_2 = [line.strip() for line in f if line.strip()]

            # Combine and remove duplicates
            all_terms = list(set(terms_1 + terms_2))
            print(f"✅ Loaded {len(terms_1)} terms from {terms_path_1}")
            print(f"✅ Loaded {len(terms_2)} terms from {terms_path_2}")
            print(f"✅ Total unique terms: {len(all_terms)}")

        else:
            # Single terms file
            with open(terms_path, 'r', encoding='utf-8') as f:
                all_terms = [line.strip() for line in f if line.strip()]
            print(f"✅ Loaded {len(all_terms)} terms from {terms_path}")

        return all_terms

    def _load_relations(self, relations_path: str) -> List[Dict[str, str]]:
        """Load relationships from json file"""
        with open(relations_path, 'r', encoding='utf-8') as f:
            relations_data = json.load(f)

        # Data validation
        valid_relations = []
        skipped_relations = 0

        for i, rel in enumerate(relations_data):
            if not isinstance(rel, dict):
                print(f"⚠️ Relationship {i} is not an object: {rel}")
                skipped_relations += 1
                continue

            parent_text = rel.get('parent', '')
            child_text = rel.get('child', '')

            if not parent_text or not child_text:
                print(f"⚠️ Relationship {i} does not contain parent or child")
                skipped_relations += 1
                continue

            # Check that terms are in our dictionary
            if parent_text not in self.term_to_id:
                print(f"⚠️ Term '{parent_text}' not found in terms list")
                skipped_relations += 1
                continue

            if child_text not in self.term_to_id:
                print(f"⚠️ Term '{child_text}' not found in terms list")
                skipped_relations += 1
                continue

            valid_relations.append({
                'id': rel.get('ID', f'rel_{i}'),
                'parent': parent_text,
                'child': child_text
            })

        print(f"✅ Loaded {len(valid_relations)} valid relationships")
        if skipped_relations > 0:
            print(f"⚠️ Skipped {skipped_relations} relationships")

        return valid_relations

    def _group_terms_by_dataset(self) -> Dict[str, List[str]]:
        """Group terms by datasets (currently all terms in one dataset)"""
        # Currently all terms in one dataset "default"
        # In the future, logic for determining dataset by file path can be added

        dataset_name = "default"
        if hasattr(self, 'relations_path') and self.relations_path:
            # Try to extract dataset name from path
            path_parts = self.relations_path.split('/')
            for part in path_parts:
                if part in ['DOID', 'FoodOn', 'GO', 'CHEBI']:
                    dataset_name = part
                    break

        groups = {dataset_name: self.terms}

        print(f"📂 Dataset: {dataset_name} with {len(self.terms)} terms")
        return groups

    def _group_relations_by_dataset(self) -> Dict[str, List[Tuple[int, int]]]:
        """Создание списка отношений для каждого датасета"""
        relations_by_dataset = {}

        for dataset_name in self.available_datasets:
            relations_by_dataset[dataset_name] = []

        for rel in self.relations:
            parent_id = self.term_to_id[rel['parent']]
            child_id = self.term_to_id[rel['child']]
            relations_by_dataset[dataset_name].append((parent_id, child_id))

        print("🔗 Распределение отношений по датасетам:")
        for dataset, relations in relations_by_dataset.items():
            print(f"  {dataset}: {len(relations)} отношений")

        return relations_by_dataset

    def _build_child_to_parents_index(self) -> Dict[str, Dict[int, Set[int]]]:
        """
        Создание индекса child -> {parent1, parent2, ...} для быстрого поиска отношений

        АРХИТЕКТУРА:
        - texts_1 = children (queries) - кто ищет родителей
        - texts_2 = parents (keys/values) - среди кого ищем родителей
        - matrix[i, j] = 1 если child_i имеет parent_j

        Returns:
            Dict[dataset_name, Dict[child_id, Set[parent_ids]]]
        """
        child_to_parents = {}

        for dataset_name, relations_list in self.relations_by_dataset.items():
            child_to_parents[dataset_name] = defaultdict(set)

            for parent_id, child_id in relations_list:
                child_to_parents[dataset_name][child_id].add(parent_id)

        # Конвертируем defaultdict в обычный dict
        result = {}
        for dataset_name, child_dict in child_to_parents.items():
            result[dataset_name] = dict(child_dict)

        return result

    def _split_train_test(self):
        """Разделение данных на train и test"""
        train_terms, test_terms = {}, {}
        train_relations, test_relations = [], []

        for dataset_name, dataset_terms in self.terms_by_dataset.items():
            if len(dataset_terms) < 2:
                print(
                    f"⚠️ Датасет {dataset_name} содержит менее 2 терминов, пропускаем split")
                if self.mode == "train":
                    train_terms[dataset_name] = dataset_terms
                continue

            # Разделение терминов
            train_terms_list, test_terms_list = train_test_split(
                dataset_terms,
                test_size=self.test_part,
                random_state=self.random_state
            )

            train_terms[dataset_name] = train_terms_list
            test_terms[dataset_name] = test_terms_list

            # Создаем новые mapping для train/test
            if self.mode == "train":
                selected_terms = train_terms_list
            else:
                selected_terms = test_terms_list

            # Обновляем индексы для выбранных терминов
            new_term_to_id = {term: i for i, term in enumerate(selected_terms)}
            new_id_to_term = {i: term for i, term in enumerate(selected_terms)}

            # Фильтруем отношения для выбранных терминов
            valid_relations = []
            for rel in self.relations:
                if rel['parent'] in new_term_to_id and rel['child'] in new_term_to_id:
                    valid_relations.append(rel)

            print(f"📊 Датасет {dataset_name} ({self.mode}):")
            print(f"  Термины: {len(selected_terms)}")
            print(f"  Отношения: {len(valid_relations)}")

        # Обновляем основные структуры данных
        if self.mode == "train":
            self.terms_by_dataset = train_terms
        else:
            self.terms_by_dataset = test_terms

        # Обновляем список терминов и индексы
        self.terms = []
        for dataset_terms in self.terms_by_dataset.values():
            self.terms.extend(dataset_terms)

        self.term_to_id = {term: i for i, term in enumerate(self.terms)}
        self.id_to_term = {i: term for i, term in enumerate(self.terms)}

        # Обновляем отношения
        # Сохраняем копию оригинальных отношений
        original_relations = self.relations.copy()
        self.relations = []
        for rel in original_relations:  # Итерируем по копии
            if rel['parent'] in self.term_to_id and rel['child'] in self.term_to_id:
                self.relations.append(rel)

        # Пересоздаем индексы отношений
        self.relations_by_dataset = self._group_relations_by_dataset()
        self.child_to_parents = self._build_child_to_parents_index()

    def _calculate_dataset_weights(self) -> Dict[str, float]:
        """Вычисление весов датасетов для weighted стратегии"""
        weights = {}
        total_relations = sum(len(relations)
                              for relations in self.relations_by_dataset.values())

        for dataset_name, relations in self.relations_by_dataset.items():
            weights[dataset_name] = len(
                relations) / total_relations if total_relations > 0 else 0

        return weights

    def _select_dataset(self, idx: int) -> str:
        """Выбор датасета согласно стратегии"""
        if self.dataset_strategy == "single" or len(
                self.available_datasets) == 1:
            return self.available_datasets[0]
        elif self.dataset_strategy == "sequential":
            return self.available_datasets[idx % len(self.available_datasets)]
        elif self.dataset_strategy == "weighted":
            return self._weighted_dataset_selection()
        else:  # random
            return random.choice(self.available_datasets)

    def _weighted_dataset_selection(self) -> str:
        """Взвешенный выбор датасета"""
        datasets = list(self.dataset_weights.keys())
        weights = list(self.dataset_weights.values())
        return np.random.choice(datasets, p=weights)

    def _sample_terms_random(self, dataset_name: str,
                             batch_size: int) -> Tuple[List[str], List[int]]:
        """Случайное семплирование терминов"""
        dataset_terms = self.terms_by_dataset[dataset_name]

        if len(dataset_terms) == 0:
            return [], []

        # Семплируем с повторениями если нужно
        if len(dataset_terms) < batch_size:
            sampled_terms = random.choices(dataset_terms, k=batch_size)
        else:
            sampled_terms = random.sample(dataset_terms, batch_size)

        sampled_ids = [self.term_to_id[term] for term in sampled_terms]
        return sampled_terms, sampled_ids

    def _sample_terms_balanced(self,
                               dataset_name: str,
                               batch_size_1: int,
                               batch_size_2: int) -> Tuple[List[str],
                                                           List[int],
                                                           List[str],
                                                           List[int]]:
        """Сбалансированное семплирование терминов с учетом отношений"""
        dataset_relations = self.relations_by_dataset[dataset_name]
        dataset_terms = self.terms_by_dataset[dataset_name]

        if len(dataset_relations) == 0 or len(dataset_terms) == 0:
            # Fallback к случайному семплированию
            texts_1, ids_1 = self._sample_terms_random(
                dataset_name, batch_size_1)
            texts_2, ids_2 = self._sample_terms_random(
                dataset_name, batch_size_2)
            return texts_1, ids_1, texts_2, ids_2

        # Собираем все уникальные parent и child ID
        all_parents = set()
        all_children = set()

        for parent_id, child_id in dataset_relations:
            all_parents.add(parent_id)
            all_children.add(child_id)

        # Семплируем children (queries)
        available_children = list(all_children)
        if len(available_children) == 0:
            # Крайний случай: нет доступных children
            sampled_children_ids = []
        elif len(available_children) < batch_size_1:
            sampled_children_ids = random.choices(
                available_children, k=batch_size_1)
        else:
            sampled_children_ids = random.sample(
                available_children, batch_size_1)

        # Семплируем parents (keys) - стараемся включить связанные с children
        target_positive_pairs = int(
            batch_size_1 *
            batch_size_2 *
            self.positive_ratio)

        # Собираем всех родителей для наших children
        related_parents = set()
        child_to_parents_map = self.child_to_parents.get(dataset_name, {})

        for child_id in sampled_children_ids:
            if child_id in child_to_parents_map:
                related_parents.update(child_to_parents_map[child_id])

        # Семплируем parents
        available_parents = list(all_parents)
        related_parents_list = list(related_parents)

        if len(related_parents_list) > 0:
            # Берем часть связанных родителей
            num_related = min(len(related_parents_list), batch_size_2 // 2)
            sampled_related_parents = random.sample(
                related_parents_list, num_related)

            # Дополняем случайными родителями
            remaining_slots = batch_size_2 - num_related
            if remaining_slots > 0:
                other_parents = [
                    p for p in available_parents if p not in sampled_related_parents]
                if len(other_parents) >= remaining_slots:
                    sampled_other_parents = random.sample(
                        other_parents, remaining_slots)
                elif len(other_parents) > 0:
                    # Если other_parents не пуст, но меньше remaining_slots
                    sampled_other_parents = random.choices(
                        other_parents, k=remaining_slots)
                else:
                    # Если other_parents пуст, используем available_parents с повторениями
                    sampled_other_parents = random.choices(
                        available_parents, k=remaining_slots)

                sampled_parents_ids = sampled_related_parents + sampled_other_parents
            else:
                sampled_parents_ids = sampled_related_parents
        else:
            # Если нет связанных родителей, берем случайных
            if len(available_parents) >= batch_size_2:
                sampled_parents_ids = random.sample(
                    available_parents, batch_size_2)
            elif len(available_parents) > 0:
                sampled_parents_ids = random.choices(
                    available_parents, k=batch_size_2)
            else:
                # Если вообще нет доступных родителей, используем все children как parents
                # Это крайний случай для очень маленьких датасетов
                if len(all_children) > 0:
                    sampled_parents_ids = random.choices(
                        list(all_children), k=batch_size_2)
                else:
                    # Если и children нет, возвращаем пустые списки
                    sampled_parents_ids = []

        # Конвертируем ID в тексты
        sampled_children_texts = [self.id_to_term[id]
                                  for id in sampled_children_ids] if sampled_children_ids else []
        sampled_parents_texts = [self.id_to_term[id]
                                 for id in sampled_parents_ids] if sampled_parents_ids else []

        return sampled_children_texts, sampled_children_ids, sampled_parents_texts, sampled_parents_ids

    def _sample_terms_from_dataset(
            self, dataset_name: str) -> Tuple[List[str], List[int], List[str], List[int]]:
        """Семплирование терминов из датасета согласно стратегии"""
        if self.sampling_strategy == "balanced":
            return self._sample_terms_balanced(
                dataset_name, self.batch_size_1, self.batch_size_2)
        else:  # random
            texts_1, ids_1 = self._sample_terms_random(
                dataset_name, self.batch_size_1)
            texts_2, ids_2 = self._sample_terms_random(
                dataset_name, self.batch_size_2)
            return texts_1, ids_1, texts_2, ids_2

    def _build_relation_matrix(
            self,
            ids_1: List[int],
            ids_2: List[int],
            dataset_name: str) -> torch.Tensor:
        """
        Построение матрицы отношений для батча

        АРХИТЕКТУРА:
        - ids_1 = children (queries) - кто ищет родителей
        - ids_2 = parents (keys/values) - среди кого ищем родителей
        - matrix[i, j] = 1 если child_i имеет parent_j

        Args:
            ids_1: список ID детей (queries)
            ids_2: список ID родителей (keys/values)
            dataset_name: название датасета

        Returns:
            torch.Tensor: матрица отношений размера (len(ids_1), len(ids_2))
        """
        matrix = torch.zeros(len(ids_1), len(ids_2), dtype=torch.float32)

        child_to_parents_map = self.child_to_parents.get(dataset_name, {})

        for i, child_id in enumerate(ids_1):
            if child_id in child_to_parents_map:
                parent_ids = child_to_parents_map[child_id]
                for j, parent_id in enumerate(ids_2):
                    if parent_id in parent_ids:
                        matrix[i, j] = 1.0

        return matrix

    def __len__(self) -> int:
        """Возвращает количество батчей в эпохе"""
        # Используем фиксированное количество батчей в эпохе
        return max(100, len(self.terms) //
                   max(self.batch_size_1, self.batch_size_2))

    def __getitem__(self,
                    idx: int) -> Tuple[List[str],
                                       List[str],
                                       torch.Tensor,
                                       str]:
        """
        Получение батча данных

        Returns:
            Tuple[List[str], List[str], torch.Tensor, str]:
                - texts_1: список текстов children (queries)
                - texts_2: список текстов parents (keys/values)
                - matrix: матрица отношений
                - dataset_name: название датасета
        """
        # Выбираем датасет
        dataset_name = self._select_dataset(idx)

        # Семплируем термины
        texts_1, ids_1, texts_2, ids_2 = self._sample_terms_from_dataset(
            dataset_name)

        # Строим матрицу отношений
        matrix = self._build_relation_matrix(ids_1, ids_2, dataset_name)

        return texts_1, texts_2, matrix, dataset_name

    def _print_statistics(self):
        """Печать статистики датасета"""
        print("\n" + "=" * 50)
        print("📊 СТАТИСТИКА METAMODEL DATASET")
        print("=" * 50)

        print(f"📁 Общее количество терминов: {len(self.terms)}")
        print(f"🔗 Общее количество отношений: {len(self.relations)}")
        print(f"📂 Количество датасетов: {len(self.available_datasets)}")
        print(f"🎯 Режим: {self.mode}")
        print(f"📏 Размер батча: {self.batch_size_1} x {self.batch_size_2}")
        print(f"🎲 Стратегия выбора датасета: {self.dataset_strategy}")
        print(f"🎯 Стратегия семплирования: {self.sampling_strategy}")

        if self.sampling_strategy == "balanced":
            print(f"⚖️ Доля положительных пар: {self.positive_ratio}")

        print(f"📊 Длина эпохи: {len(self)} батчей")
        print("=" * 50)

    def get_dataset_info(self) -> Dict[str, Any]:
        """Получение информации о датасете"""
        return {
            "total_terms": len(
                self.terms),
            "total_relations": len(
                self.relations),
            "num_datasets": len(
                self.available_datasets),
            "available_datasets": self.available_datasets,
            "mode": self.mode,
            "batch_size_1": self.batch_size_1,
            "batch_size_2": self.batch_size_2,
            "dataset_strategy": self.dataset_strategy,
            "sampling_strategy": self.sampling_strategy,
            "positive_ratio": self.positive_ratio,
            "epoch_length": len(self),
            "terms_by_dataset": {
                k: len(v) for k,
                v in self.terms_by_dataset.items()},
            "relations_by_dataset": {
                k: len(v) for k,
                v in self.relations_by_dataset.items()}}

    def get_all_terms(self) -> List[str]:
        """Получение всех терминов"""
        return self.terms.copy()

    def get_all_relations(self) -> List[Dict[str, str]]:
        """Получение всех отношений"""
        return self.relations.copy()

    def get_term_by_id(self, term_id: int) -> str:
        """Получение термина по ID"""
        return self.id_to_term.get(term_id, "")

    def get_id_by_term(self, term: str) -> int:
        """Получение ID по термину"""
        return self.term_to_id.get(term, -1)


def create_train_test_datasets(
    terms_path: str | Tuple[str, str],
    relations_path: str,
    batch_size_1: int = 32,
    batch_size_2: int = 32,
    dataset_strategy: str = "single",
    sampling_strategy: str = "balanced",
    positive_ratio: float = 1.0,
    test_part: float = 0.2,
    random_state: int = 42
) -> Tuple['MetaModelDataset', 'MetaModelDataset']:
    """
    Создание train и test датасетов

    Args:
        terms_path: путь к файлу с терминами или кортеж путей
        relations_path: путь к файлу с отношениями
        batch_size_1: размер первого батча
        batch_size_2: размер второго батча
        dataset_strategy: стратегия выбора датасета
        sampling_strategy: стратегия семплирования
        positive_ratio: доля положительных пар
        test_part: доля тестовых данных
        random_state: seed для воспроизводимости

    Returns:
        Tuple[MetaModelDataset, MetaModelDataset]: train и test датасеты
    """
    train_dataset = MetaModelDataset(
        terms_path=terms_path,
        relations_path=relations_path,
        batch_size_1=batch_size_1,
        batch_size_2=batch_size_2,
        dataset_strategy=dataset_strategy,
        sampling_strategy=sampling_strategy,
        positive_ratio=positive_ratio,
        mode="train",
        test_part=test_part,
        random_state=random_state
    )

    test_dataset = MetaModelDataset(
        terms_path=terms_path,
        relations_path=relations_path,
        batch_size_1=batch_size_1,
        batch_size_2=batch_size_2,
        dataset_strategy=dataset_strategy,
        sampling_strategy=sampling_strategy,
        positive_ratio=positive_ratio,
        mode="test",
        test_part=test_part,
        random_state=random_state
    )

    return train_dataset, test_dataset


# Для обратной совместимости
CrossAttentionDataset = MetaModelDataset
