"""
Qwen3CrossAttentionMetaModel - мета-модель, объединяющая Qwen3 и CrossAttentionModel

Эта модель принимает список фраз, получает эмбеддинги через Qwen3,
а затем применяет CrossAttentionModel для построения матрицы внимания.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple, Optional, Any, Union
import json
import os
from pathlib import Path
import numpy as np
import logging

from cross_attention_model import CrossAttentionModel


def last_token_pool(
    last_hidden_states: Tensor,
     attention_mask: Tensor) -> Tensor:
    """Extract last token embeddings with proper handling of padding"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(
            batch_size, device=last_hidden_states.device), sequence_lengths]


class Qwen3CrossAttentionMetaModel(nn.Module):
    """
    Мета-модель, объединяющая Qwen3 и CrossAttentionModel

    Принимает два списка фраз и возвращает матрицу внимания между ними.

    Args:
        cross_attention_model: обученная CrossAttentionModel
        qwen3_model_name: имя модели Qwen3 для эмбеддингов
        max_length: максимальная длина последовательности для токенизации
        embedding_batch_size: размер батча для получения эмбеддингов
        freeze_qwen3: заморозить ли веса Qwen3
        device: устройство для вычислений
        embedding_pooling: метод pooling для эмбеддингов ("mean", "last_token", "last_token_norm")
    """

    def __init__(
        self,
        qwen3_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        max_length: int = 512,
        embedding_batch_size: int = 32,
        freeze_qwen3: bool = True,
        device: str = "auto",
        embedding_pooling: str = "last_token_norm",
        cross_attention_model: Union[CrossAttentionModel, None, str] = None,
    ):
        super().__init__()

        # Валидация pooling метода
        if embedding_pooling not in ["mean", "last_token", "last_token_norm"]:
            raise ValueError(
                f"embedding_pooling должен быть 'mean' или 'last_token', получен: {embedding_pooling}")

        self.qwen3_model_name = qwen3_model_name
        self.max_length = max_length
        self.embedding_batch_size = embedding_batch_size
        self.freeze_qwen3 = freeze_qwen3
        self.embedding_pooling = embedding_pooling

        # Инициализируем f1_threshold как None (будет установлен при загрузке
        # из checkpoint)
        self.f1_threshold = None

        # Определяем устройство
        if device == "auto":
            self.device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Загрузка Qwen3 модели и токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(qwen3_model_name)
        self.qwen3_model = AutoModel.from_pretrained(qwen3_model_name)

        # Замораживаем веса Qwen3 если нужно
        if freeze_qwen3:
            for param in self.qwen3_model.parameters():
                param.requires_grad = False

        # Сохраняем CrossAttentionModel
        if cross_attention_model is None:
            cross_attention_model = CrossAttentionModel(
                self.qwen3_model.config)
            self.cross_attention_model = cross_attention_model
        elif isinstance(cross_attention_model, str):
            self.cross_attention_model = CrossAttentionModel.from_pretrained(
                cross_attention_model)
        elif isinstance(cross_attention_model, CrossAttentionModel):
            self.cross_attention_model = cross_attention_model
        else:
            raise ValueError(
                f"Неизвестный тип cross_attention_model: {type(cross_attention_model)}")

        # Перемещаем на устройство
        self.qwen3_model = self.qwen3_model.to(self.device)
        self.cross_attention_model = self.cross_attention_model.to(self.device)

        # Переводим в режим оценки
        self.qwen3_model.eval()

        # Получаем размерность эмбеддингов
        self.embedding_dim = self.qwen3_model.config.hidden_size

        # Проверяем совместимость размерностей
        if self.embedding_dim != cross_attention_model.hidden_size:
            raise ValueError(
    f"Размерность эмбеддингов Qwen3 ({self.embedding_dim}) "
    f"не совпадает с размерностью CrossAttentionModel ({cross_attention_model.hidden_size})")

    def get_embeddings(self, phrases: List[str]) -> torch.Tensor:
        """
        Получение эмбеддингов для списка фраз

        Args:
            phrases: список фраз

        Returns:
            embeddings: тензор размера (len(phrases), embedding_dim)
        """
        all_embeddings = []

        # Обрабатываем фразы батчами
        for i in range(0, len(phrases), self.embedding_batch_size):
            batch_phrases = phrases[i:i + self.embedding_batch_size]

            # Токенизация
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch_phrases,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)

                # Получаем эмбеддинги
                outputs = self.qwen3_model(**inputs)

                # Применяем выбранный метод pooling
                if self.embedding_pooling == "mean":
                    # Mean pooling последнего слоя
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                elif self.embedding_pooling == "last_token":
                    # Last token pooling с правильной обработкой padding
                    batch_embeddings = last_token_pool(
                        outputs.last_hidden_state, inputs["attention_mask"])
                elif self.embedding_pooling == "last_token_norm":
                    # Last token pooling с правильной обработкой padding
                    batch_embeddings = last_token_pool(
                        outputs.last_hidden_state, inputs["attention_mask"])
                    batch_embeddings = F.normalize(
                        batch_embeddings, p=2, dim=1)
                else:
                    raise ValueError(
                        f"Неизвестный метод pooling: {self.embedding_pooling}")

                all_embeddings.append(batch_embeddings)

        # Объединяем все батчи
        embeddings = torch.cat(all_embeddings, dim=0)

        return embeddings

    def forward(
        self,
        phrases_1: List[str],
        phrases_2: List[str]) -> torch.Tensor:
        """
        Forward pass мета-модели

        Args:
            phrases_1: первый список фраз (query)
            phrases_2: второй список фраз (key)

        Returns:
            attention_matrix: матрица внимания размером (len(phrases_1), len(phrases_2))
        """
        # Получаем эмбеддинги для обоих списков
        embeddings_1 = self.get_embeddings(phrases_1)
        embeddings_2 = self.get_embeddings(phrases_2)

        # Применяем CrossAttentionModel
        attention_matrix = self.cross_attention_model(
            embeddings_1, embeddings_2)

        return attention_matrix

    def predict_relationships(
        self,
        phrases_1: List[str],
        phrases_2: List[str],
        threshold: Union[float, None] = None,
        remove_self_loops: bool = True
    ) -> List[Dict[str, str]]:
        """
        Предсказание отношений между фразами с применением порога

        Args:
            phrases_1: первый список фраз (потенциальные дети)
            phrases_2: второй список фраз (потенциальные родители)
            threshold: порог для бинаризации (если None, используется оптимальный F1 threshold из обучения)
            remove_self_loops: удалить ли самозацикливания (когда фраза ссылается сама на себя)

        Returns:
            relationships: список отношений в формате [{"child": "...", "parent": "..."}]
        """
        # Используем оптимальный F1 threshold если не задан явно
        if threshold is None:
            threshold = getattr(self, 'f1_threshold', None)
            if threshold is None:
                threshold = 0.5

        # Получаем матрицу внимания
        attention_matrix = self.forward(phrases_1, phrases_2)

        # Применяем порог
        binary_matrix = (attention_matrix > threshold).cpu().numpy()

        # Получаем индексы ненулевых элементов (векторизованно)
        nonzero_indices = np.nonzero(binary_matrix)
        i_indices, j_indices = nonzero_indices[0], nonzero_indices[1]

        # Извлекаем отношения только для ненулевых элементов
        relationships = []
        for idx in range(len(i_indices)):
            i, j = i_indices[idx], j_indices[idx]

            # Пропускаем самозацикливания если нужно
            if remove_self_loops and phrases_1[i] == phrases_2[j]:
                continue

            relationships.append({
                "child": phrases_1[i],
                "parent": phrases_2[j],
                "confidence": float(attention_matrix[i, j].item())
            })

        return relationships

    def predict_hierarchy(
        self,
        phrases: List[str],
        threshold: float = None,
        remove_self_loops: bool = True
    ) -> List[Dict[str, str]]:
        """
        Предсказание иерархии для списка фраз (phrases × phrases)

        Args:
            phrases: список фраз для построения иерархии
            threshold: порог для бинаризации (если None, используется оптимальный F1 threshold из обучения)
            remove_self_loops: удалить ли самозацикливания

        Returns:
            hierarchy: список отношений в формате [{"child": "...", "parent": "..."}]
        """
        # Используем оптимальный F1 threshold если не задан явно
        if threshold is None:
            threshold = getattr(self, 'f1_threshold', None)
            if threshold is None:
                threshold = 0.5

        # Получаем матрицу внимания для всех пар
        attention_matrix = self.forward(phrases, phrases)

        # Применяем порог
        binary_matrix = (attention_matrix > threshold).cpu().numpy()

        # Получаем индексы ненулевых элементов (векторизованно)
        nonzero_indices = np.nonzero(binary_matrix)
        i_indices, j_indices = nonzero_indices[0], nonzero_indices[1]

        # Извлекаем отношения только для ненулевых элементов
        hierarchy = []
        for idx in range(len(i_indices)):
            # Пропускаем самозацикливания если нужно
            i, j = i_indices[idx], j_indices[idx]
            if remove_self_loops and i == j:
                    continue

            hierarchy.append({
                    "child": phrases[i],
                    "parent": phrases[j],
                    "confidence": float(attention_matrix[i, j].item())
                })

        return hierarchy

    def save_pretrained(self, save_directory: str):
        """
        Сохранение мета-модели (включая все веса)
        Если модель содержит LORA адаптеры, они автоматически "склеиваются" в цельные веса

        Args:
            save_directory: путь к директории для сохранения
        """
        os.makedirs(save_directory, exist_ok=True)

        # Проверяем, есть ли LORA адаптеры
        has_lora = self._check_has_lora()

        if has_lora:
            print(f"🔄 Обнаружены LORA адаптеры, склеиваем в цельные веса...")
            # Склеиваем LORA адаптеры в цельную модель
            merged_state_dict = self._merge_lora_weights()
        else:
            # Обычное состояние модели
            merged_state_dict = self.state_dict()

        # Сохраняем состояние модели с цельными весами
        model_state = {
            'model_state_dict': merged_state_dict,
            'qwen3_model_name': self.qwen3_model_name,
            'max_length': self.max_length,
            'embedding_batch_size': self.embedding_batch_size,
            'freeze_qwen3': self.freeze_qwen3,
            'embedding_dim': self.embedding_dim,
            'embedding_pooling': self.embedding_pooling,
            'f1_threshold': getattr(self, 'f1_threshold', None),
            # Добавляем информацию о стратегии заморозки
            "freeze_strategy": getattr(self, 'freeze_strategy', 'unknown'),
            "lora_rank": getattr(self, 'lora_rank', None),
            "lora_alpha": getattr(self, 'lora_alpha', None),
            "has_lora": self._check_has_lora()
        }

        # Сохраняем веса модели
        torch.save(model_state, os.path.join(save_directory, "model.pt"))

        # Сохраняем конфигурацию мета-модели (для совместимости)
        meta_config = {
            "model_type": "qwen3_cross_attention_meta_model",
            "qwen3_model_name": self.qwen3_model_name,
            "max_length": self.max_length,
            "embedding_batch_size": self.embedding_batch_size,
            "freeze_qwen3": self.freeze_qwen3,
            "embedding_dim": self.embedding_dim,
            "embedding_pooling": self.embedding_pooling,
            "f1_threshold": getattr(self, 'f1_threshold', None),
            # Добавляем информацию о стратегии заморозки
            "freeze_strategy": getattr(self, 'freeze_strategy', 'unknown'),
            "lora_rank": getattr(self, 'lora_rank', None),
            "lora_alpha": getattr(self, 'lora_alpha', None),
            "has_lora": has_lora
        }

        with open(os.path.join(save_directory, "config.json"), 'w') as f:
            json.dump(meta_config, f, indent=2)

        # Также сохраняем CrossAttentionModel отдельно (для совместимости)
        cross_attention_dir = os.path.join(
    save_directory, "cross_attention_model")
        self.cross_attention_model.save_pretrained(cross_attention_dir)

        print(f"Мета-модель сохранена в {save_directory}")
        print(f"   - Веса модели: model.pt")
        print(f"   - Конфигурация: config.json")
        print(f"   - CrossAttention: cross_attention_model/")

    def _check_has_lora(self) -> bool:
        """Проверяет, есть ли LORA адаптеры в модели"""
        try:
            from peft import PeftModel
            return isinstance(self.qwen3_model, PeftModel)
        except BaseException:
            # Альтернативная проверка через названия параметров
            for name, _ in self.qwen3_model.named_parameters():
                if 'lora_A' in name or 'lora_B' in name or 'base_model' in name:
                    return True
            return False

    def _merge_lora_weights(self) -> dict:
        """Склеивает LORA адаптеры в цельные веса"""
        try:
            # Пытаемся использовать встроенный метод PEFT
            from peft import PeftModel
            if isinstance(self.qwen3_model, PeftModel):
                print(f"   Используем встроенный merge PEFT...")
                # Создаем копию модели и сливаем LORA
                merged_model = self.qwen3_model.merge_and_unload()

                # Получаем state_dict объединенной модели и исходной модели
                original_state = self.state_dict()
                merged_qwen3_state = merged_model.state_dict()

                # Создаем финальное состояние модели
                final_state = {}

                # Копируем все non-qwen3 веса как есть
                for key, value in original_state.items():
                    if not key.startswith('qwen3_model.'):
                        final_state[key] = value

                # Заменяем qwen3 веса на склеенные
                for merged_key, merged_value in merged_qwen3_state.items():
                    # Добавляем префикс qwen3_model к ключу из merged модели
                    final_key = f"qwen3_model.{merged_key}"
                    final_state[final_key] = merged_value
                    print(f"   Склеен ключ: {merged_key} -> {final_key}")

                print(f"   ✅ LORA адаптеры успешно склеены")
                return final_state

        except Exception as e:
            print(f"   ⚠️ Ошибка при склеивании LORA: {e}")
            print(f"   Используем ручное склеивание...")

        # Ручное склеивание как fallback
        return self._manual_merge_lora()

    def _manual_merge_lora(self) -> dict:
        """Ручное склеивание LORA весов"""
        state_dict = self.state_dict()
        merged_state = {}

        # Группируем LORA слои
        lora_layers = {}

        for name, param in state_dict.items():
            if 'base_model.model.' in name:
                # Извлекаем правильное имя слоя, убирая base_model.model
                # Из: qwen3_model.base_model.model.layers.0.self_attn.q_proj.base_layer.weight
                # В:  qwen3_model.layers.0.self_attn.q_proj
                clean_name = name.replace('.base_model.model.', '.')
                layer_name = clean_name.split('.base_layer.')[
                                              0] if '.base_layer.' in clean_name else None

                if layer_name:
                    if layer_name not in lora_layers:
                        lora_layers[layer_name] = {}

                    if '.base_layer.' in name:
                        lora_layers[layer_name]['base'] = param
                    elif '.lora_A.default.' in name:
                        lora_layers[layer_name]['lora_A'] = param
                    elif '.lora_B.default.' in name:
                        lora_layers[layer_name]['lora_B'] = param
            else:
                # Обычные параметры остаются как есть
                merged_state[name] = param

        # Склеиваем LORA слои
        lora_alpha = getattr(self, 'lora_alpha', 16)
        lora_rank = getattr(self, 'lora_rank', 8)
        scaling = lora_alpha / lora_rank

        for layer_name, lora_parts in lora_layers.items():
            if all(key in lora_parts for key in ['base', 'lora_A', 'lora_B']):
                # Склеиваем: W_final = W_base + lora_B @ lora_A * scaling
                base_weight = lora_parts['base']
                lora_A = lora_parts['lora_A']
                lora_B = lora_parts['lora_B']

                # Вычисляем дельту LORA
                delta = torch.mm(lora_B, lora_A) * scaling

                # Финальный вес
                final_weight = base_weight + delta

                # Сохраняем под обычным именем (без .base_layer)
                final_name = layer_name + '.weight'
                merged_state[final_name] = final_weight

                print(f"   Склеен слой: {layer_name}")
            else:
                # Если не все части LORA найдены, сохраняем базовый вес
                if 'base' in lora_parts:
                    final_name = layer_name + '.weight'
                    merged_state[final_name] = lora_parts['base']

        print(f"   ✅ Ручное склеивание завершено")
        return merged_state

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "auto",
        **kwargs
    ):
        """
        Загрузка мета-модели из сохраненного состояния

        Args:
            model_path: путь к директории с сохраненной моделью
            device: устройство для вычислений
            **kwargs: дополнительные аргументы для конструктора

        Returns:
            meta_model: экземпляр Qwen3CrossAttentionMetaModel
        """
        model_path = Path(model_path)

        # Определяем устройство
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Проверяем наличие файла с весами
        model_pt_path = model_path / "model.pt"
        config_path = model_path / "config.json"

        if model_pt_path.exists():
            # Новый формат - загружаем полную модель
            print(f"Загружаем полную модель из {model_pt_path}")
            model_state = torch.load(model_pt_path, map_location=device)

            # Сначала пытаемся загрузить CrossAttentionModel из папки (если
            # есть)
            cross_attention_path = model_path / "cross_attention_model"
            cross_attention_model = None

            if cross_attention_path.exists():
                try:
                    from cross_attention_model import CrossAttentionModel
                    cross_attention_model = CrossAttentionModel.from_pretrained(
                        str(cross_attention_path))
                    print(f"✅ CrossAttentionModel загружен из папки")
                except Exception as e:
                    print(
                        f"⚠️ Не удалось загрузить CrossAttentionModel из папки: {e}")

            # Создаем мета-модель с теми же параметрами
            meta_model = cls(
    cross_attention_model=cross_attention_model,
    qwen3_model_name=model_state["qwen3_model_name"],
    max_length=model_state["max_length"],
    embedding_batch_size=model_state["embedding_batch_size"],
    freeze_qwen3=model_state["freeze_qwen3"],
    embedding_pooling=model_state.get(
        "embedding_pooling",
        "last_token_norm"),
        device=device,
         **kwargs)

            # Восстанавливаем информацию о стратегии заморозки (для справки)
            if 'freeze_strategy' in model_state:
                meta_model.freeze_strategy = model_state['freeze_strategy']
                meta_model.lora_rank = model_state.get('lora_rank')
                meta_model.lora_alpha = model_state.get('lora_alpha')

                if model_state.get('has_lora', False):
                    print(
                        f"✅ Модель была обучена с LORA (rank={meta_model.lora_rank}, alpha={meta_model.lora_alpha})")
                    print(f"   LORA адаптеры уже склеены в цельные веса при сохранении")

            # Загружаем склеенные веса (должны точно совпадать)
            try:
                meta_model.load_state_dict(model_state['model_state_dict'])
                print(f"✅ Веса модели загружены успешно")
            except Exception as e:
                print(f"⚠️ Ошибка при загрузке весов: {e}")
                print(f"   Возможно, версии модели несовместимы")
                # Попробуем загрузить только совместимые веса
                current_state = meta_model.state_dict()
                compatible_state = {}
                for key, value in model_state['model_state_dict'].items():
                    if key in current_state and current_state[key].shape == value.shape:
                        compatible_state[key] = value
                    else:
                        print(f"   Пропускаем несовместимый слой: {key}")

                meta_model.load_state_dict(compatible_state, strict=False)
                print(f"✅ Совместимые веса загружены")

            # Восстанавливаем f1_threshold
            if model_state.get("f1_threshold") is not None:
                meta_model.f1_threshold = model_state["f1_threshold"]
                print(
                    f"Восстановлен F1 threshold: {meta_model.f1_threshold:.4f}")

            print(f"✅ Мета-модель загружена из {model_path} (полная модель)")
            return meta_model

        elif config_path.exists():
            # Старый формат - загружаем по частям (для совместимости)
            print(f"Загружаем модель в старом формате из {config_path}")
            with open(config_path, 'r') as f:
                meta_config = json.load(f)

            # Загружаем CrossAttentionModel
            cross_attention_path = model_path / "cross_attention_model"
            if not cross_attention_path.exists():
                raise FileNotFoundError(
                    f"cross_attention_model не найден в {model_path}")

            cross_attention_model = CrossAttentionModel.from_pretrained(
                str(cross_attention_path))

            # Создаем мета-модель
            meta_model = cls(
                cross_attention_model=cross_attention_model,
                qwen3_model_name=meta_config["qwen3_model_name"],
                max_length=meta_config["max_length"],
                embedding_batch_size=meta_config["embedding_batch_size"],
                freeze_qwen3=meta_config["freeze_qwen3"],
                embedding_pooling=meta_config.get("embedding_pooling", "last_token_norm"),
                device=device,
                **kwargs
            )

            # Восстанавливаем f1_threshold если есть
            if meta_config.get("f1_threshold") is not None:
                meta_model.f1_threshold = meta_config["f1_threshold"]
                print(
                    f"Восстановлен F1 threshold: {meta_model.f1_threshold:.4f}")

            print(f"⚠️  Мета-модель загружена из {model_path} (старый формат)")
            print(f"   Внимание: веса Qwen3 могут быть не обучены!")
            return meta_model

        else:
            raise FileNotFoundError(
                f"Не найден ни model.pt, ни config.json в {model_path}")

    @classmethod
    def from_cross_attention_checkpoint(
        cls,
        cross_attention_results_dir: str,
        qwen3_model_name: str = "Qwen/Qwen3-Embedding-4B",
        device: str = "auto",
        **kwargs
    ):
        """
        Создание мета-модели из директории с результатами обучения CrossAttentionModel

        Args:
            cross_attention_results_dir: директория с результатами обучения
            qwen3_model_name: имя модели Qwen3
            device: устройство для вычислений
            **kwargs: дополнительные аргументы для конструктора

        Returns:
            meta_model: экземпляр Qwen3CrossAttentionMetaModel
        """
        # Загружаем обученную CrossAttentionModel
        from inference_cross_attention import load_trained_model

        cross_attention_model, best_results, f1_threshold = load_trained_model(
            cross_attention_results_dir
        )

        # Создаем мета-модель
        meta_model = cls(
            cross_attention_model=cross_attention_model,
            qwen3_model_name=qwen3_model_name,
            device=device,
            **kwargs
        )

        # Сохраняем информацию о лучших результатах
        meta_model.best_results = best_results
        meta_model.f1_threshold = f1_threshold

        print(f"Мета-модель создана из {cross_attention_results_dir}")
        print(f"Лучший F1 порог: {f1_threshold:.4f}")

        return meta_model

    def get_model_info(self) -> Dict[str, Any]:
        """
        Получение информации о мета-модели

        Returns:
            info: словарь с информацией о модели
        """
        cross_attention_info = self.cross_attention_model.get_model_info()

        qwen3_params = sum(p.numel() for p in self.qwen3_model.parameters())
        cross_attention_params = cross_attention_info['num_trainable_params']

        info = {
            'model_type': 'Qwen3CrossAttentionMetaModel',
            'qwen3_model_name': self.qwen3_model_name,
            'embedding_dim': self.embedding_dim,
            'max_length': self.max_length,
            'embedding_batch_size': self.embedding_batch_size,
            'freeze_qwen3': self.freeze_qwen3,
            'embedding_pooling': self.embedding_pooling,
            'device': self.device,
            'f1_threshold': getattr(self, 'f1_threshold', None),
            'qwen3_params': qwen3_params,
            'cross_attention_params': cross_attention_params,
            'total_params': qwen3_params + cross_attention_params,
            'cross_attention_model_info': cross_attention_info
        }

        return info

    def __repr__(self):
        info = self.get_model_info()
        threshold_str = f", threshold={info['f1_threshold']:.4f}" if info['f1_threshold'] is not None else ""
        return f"Qwen3CrossAttentionMetaModel(" \
               f"qwen3={info['qwen3_model_name']}, " \
               f"embedding_dim={info['embedding_dim']}, " \
               f"pooling={info['embedding_pooling']}, " \
               f"freeze_qwen3={info['freeze_qwen3']}" \
               f"{threshold_str}, " \
               f"total_params={info['total_params']:,})"


def create_meta_model_from_checkpoint(
    cross_attention_results_dir: str,
    qwen3_model_name: str = "Qwen/Qwen3-Embedding-4B",
    device: str = "auto",
    **kwargs
) -> Qwen3CrossAttentionMetaModel:
    """
    Удобная функция для создания мета-модели из чекпоинта

    Args:
        cross_attention_results_dir: директория с результатами обучения
        qwen3_model_name: имя модели Qwen3
        device: устройство для вычислений
        **kwargs: дополнительные аргументы

    Returns:
        meta_model: экземпляр Qwen3CrossAttentionMetaModel
    """
    return Qwen3CrossAttentionMetaModel.from_cross_attention_checkpoint(
        cross_attention_results_dir=cross_attention_results_dir,
        qwen3_model_name=qwen3_model_name,
        device=device,
        **kwargs
    )


def create_meta_model_from_scratch(
    qwen3_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    num_attention_heads: int = 8,
    num_key_value_heads: int = 8,
    layer_idx: int = 0,
    device: str = "auto",
    init_from_qwen3: bool = True,
    embedding_pooling: str = "mean",
    **kwargs
) -> Qwen3CrossAttentionMetaModel:
    """
    Создание мета-модели с нуля на основе выбранной модели Qwen3

    Args:
        qwen3_model_name: имя модели Qwen3 для эмбеддингов
        num_attention_heads: количество голов внимания
        num_key_value_heads: количество голов для key-value (GQA)
        layer_idx: индекс слоя для инициализации (если init_from_qwen3=True)
        device: устройство для вычислений
        init_from_qwen3: инициализировать ли CrossAttentionModel из весов Qwen3
        embedding_pooling: метод pooling для эмбеддингов ("mean", "last_token", "last_token_norm")
        **kwargs: дополнительные аргументы для мета-модели

    Returns:
        meta_model: экземпляр Qwen3CrossAttentionMetaModel
    """
    from transformers import AutoModel, AutoConfig

    print(f"🚀 Создание мета-модели с нуля на основе {qwen3_model_name}")

    # Загружаем конфигурацию Qwen3 для получения размерности
    print("📁 Загрузка конфигурации Qwen3...")
    qwen3_config = AutoConfig.from_pretrained(qwen3_model_name)
    hidden_size = qwen3_config.hidden_size

    print(f"   - Размерность эмбеддингов: {hidden_size}")
    print(f"   - Количество голов внимания: {num_attention_heads}")
    print(f"   - Количество KV голов: {num_key_value_heads}")

    # Создаем конфигурацию для CrossAttentionModel
    cross_attention_config = {
        'hidden_size': hidden_size,
        'num_attention_heads': num_attention_heads,
        'num_key_value_heads': num_key_value_heads,
        'rms_norm_eps': getattr(qwen3_config, 'rms_norm_eps', 1e-6),
        'attention_bias': getattr(qwen3_config, 'attention_bias', False),
        'head_dim': hidden_size // num_attention_heads
    }

    # Создаем CrossAttentionModel
    print("🔧 Создание CrossAttentionModel...")
    cross_attention_model = CrossAttentionModel(
    cross_attention_config, layer_idx=layer_idx)

    # Инициализируем из весов Qwen3 если нужно
    if init_from_qwen3:
        print("⚡ Инициализация из весов Qwen3...")
        try:
            # Загружаем модель Qwen3 для инициализации весов
            qwen3_model = AutoModel.from_pretrained(qwen3_model_name)

            # Получаем слой для инициализации
            if hasattr(
    qwen3_model, 'layers') and len(
        qwen3_model.layers) > layer_idx:
                qwen3_layer = qwen3_model.layers[layer_idx]

                # Копируем веса attention
                if hasattr(qwen3_layer, 'self_attn'):
                    qwen3_attn = qwen3_layer.self_attn

                    # Копируем q_proj и k_proj
                    if hasattr(qwen3_attn, 'q_proj'):
                        cross_attention_model.q_proj.weight.data.copy_(
                            qwen3_attn.q_proj.weight.data)
                        if cross_attention_model.q_proj.bias is not None and qwen3_attn.q_proj.bias is not None:
                            cross_attention_model.q_proj.bias.data.copy_(
                                qwen3_attn.q_proj.bias.data)

                    if hasattr(qwen3_attn, 'k_proj'):
                        cross_attention_model.k_proj.weight.data.copy_(
                            qwen3_attn.k_proj.weight.data)
                        if cross_attention_model.k_proj.bias is not None and qwen3_attn.k_proj.bias is not None:
                            cross_attention_model.k_proj.bias.data.copy_(
                                qwen3_attn.k_proj.bias.data)

                    print("   ✅ Веса q_proj и k_proj скопированы")
                    cross_attention_model.initialized_from_qwen3 = True
                else:
                    print("   ⚠️ self_attn не найден в слое Qwen3")
            else:
                print(f"   ⚠️ Слой {layer_idx} не найден в модели Qwen3")

            # Освобождаем память
            del qwen3_model
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"   ⚠️ Ошибка при инициализации из Qwen3: {e}")
            print("   Используется случайная инициализация")

    # Создаем мета-модель
    print("🏗️ Создание мета-модели...")
    meta_model = Qwen3CrossAttentionMetaModel(
        cross_attention_model=cross_attention_model,
        qwen3_model_name=qwen3_model_name,
        device=device,
        embedding_pooling=embedding_pooling,
        **kwargs
    )

    # Примечание: f1_threshold остается None для моделей, созданных с нуля
    # Будет установлен после обучения или можно задать вручную

    print("✅ Мета-модель создана успешно!")
    print(
        f"   - Общее количество параметров: {meta_model.get_model_info()['total_params']:,}")
    print(
        f"   - F1 threshold: {meta_model.f1_threshold} (будет установлен после обучения)")

    return meta_model


if __name__ == "__main__":
    # Пример использования
    print("🚀 Тестирование Qwen3CrossAttentionMetaModel...")

    # Создаем тестовую CrossAttentionModel
    test_config = {
        'hidden_size': 2560,  # Размерность Qwen3-Embedding-4B
        'num_attention_heads': 8,
        'num_key_value_heads': 8,
        'rms_norm_eps': 1e-6,
        'attention_bias': False
    }

    cross_attention_model = CrossAttentionModel(test_config)

    # Создаем мета-модель
    meta_model = Qwen3CrossAttentionMetaModel(
        cross_attention_model=cross_attention_model,
        embedding_batch_size=2,  # Маленький батч для тестирования
        device="cpu"  # Для тестирования
    )

    print(f"Создана мета-модель: {meta_model}")

    # Тестовые данные
    phrases_1 = ["machine learning", "deep learning", "neural networks"]
    phrases_2 = ["artificial intelligence", "computer science", "technology"]

    print(f"\nТестовые данные:")
    print(f"  phrases_1: {phrases_1}")
    print(f"  phrases_2: {phrases_2}")

    # Тестируем forward pass
    with torch.no_grad():
        attention_matrix = meta_model(phrases_1, phrases_2)

    print(f"\nРезультаты:")
    print(f"  Матрица внимания: {attention_matrix.shape}")
    print(f"  Матрица внимания:\n{attention_matrix}")

    # Тестируем предсказание отношений (используется автоматический threshold)
    relationships = meta_model.predict_relationships(phrases_1, phrases_2)
    current_threshold = getattr(meta_model, 'f1_threshold', 0.5)
    print(f"\nПредсказанные отношения (threshold={current_threshold}):")
    for rel in relationships:
        print(
            f"  {rel['child']} -> {rel['parent']} (confidence: {rel['confidence']:.4f})")

    # Также тестируем с явным threshold
    relationships_custom = meta_model.predict_relationships(
        phrases_1, phrases_2, threshold=0.3)
    print(f"\nПредсказанные отношения (явный threshold=0.3):")
    for rel in relationships_custom:
        print(
            f"  {rel['child']} -> {rel['parent']} (confidence: {rel['confidence']:.4f})")

    # Информация о модели
    info = meta_model.get_model_info()
    print(f"\nИнформация о модели:")
    for key, value in info.items():
        if key != 'cross_attention_model_info':
            print(f"  {key}: {value}")

    print("\n✅ Тестирование завершено успешно!")
