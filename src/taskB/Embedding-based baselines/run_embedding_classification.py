#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Простой скрипт для запуска классификации терминов с эмбеддингами
Для запуска в conda окружении (rah_11_cu12.4_torch)
"""

import os
import json
from term_classification_with_embeddings import EmbeddingTermClassifier

def load_json_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def get_domain_files(domain, base_path):
    """Получение путей к файлам для конкретного домена"""
    domain_lower = domain.lower()
    train_file = os.path.join(base_path, domain, "train", "term_typing_train_data.json")
    test_file = os.path.join(base_path, domain, "test", f"{domain_lower}_term_typing_test_data.json")
    return train_file, test_file

def main():
    # Пути к данным
    base_path = "../../../2025/TaskB-TermTyping"
    domains = ["MatOnto", "OBI", "SWEET"]
    
    for domain in domains:
        print(f"\n🔄 Обработка домена {domain}")
        
        # Получение путей к файлам данных
        train_file, test_file = get_domain_files(domain, base_path)
        
        # Загрузка данных
        train_data = load_json_data(train_file)
        test_data = load_json_data(test_file)
        
        print(f"📊 Размер обучающей выборки: {len(train_data)}")
        print(f"📊 Размер тестовой выборки: {len(test_data)}")
        
        # Создание директории для сохранения результатов
        save_dir = f"predictions_embedding_{domain}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Инициализация классификатора
        classifier = EmbeddingTermClassifier()
        
        # Подготовка данных
        X_train_embeddings, y_train, X_test_embeddings, test_terms = classifier.prepare_data(train_data, test_data)
        
        # Обучение и оценка классификаторов
        clf, clf_with_graph, G = classifier.train_and_evaluate(domain, save_dir)
        
        # Предсказание для тестовых данных
        predictions, predictions_with_graph = classifier.predict(
            clf, clf_with_graph, G, X_test_embeddings, test_terms, test_data, save_dir
        )
        
        print(f"\n✅ Обработка домена {domain} завершена")
        print(f"📁 Результаты сохранены в директории: {save_dir}")

if __name__ == "__main__":
    main() 