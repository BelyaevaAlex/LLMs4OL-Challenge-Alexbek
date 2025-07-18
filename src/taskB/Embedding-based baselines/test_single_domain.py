#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестовый скрипт для проверки правильного формата данных на одном домене
"""

import sys
import os
import json

# Добавляем путь к проекту
project_root = '../../../'
sys.path.append(project_root)

from src.taskB.method_v1_1.term_classification_with_embeddings import EmbeddingTermClassifier, load_taskb_data

def test_single_domain(domain_name="MatOnto"):
    """Тестирование одного домена с проверкой формата"""
    print(f"🧪 Тестирование домена: {domain_name}")
    
    # Загрузка данных
    train_data, test_data, domains = load_taskb_data()
    
    if domain_name not in train_data or domain_name not in test_data:
        print(f"❌ Домен {domain_name} не найден")
        return
    
    # Показываем примеры данных
    print(f"\n📊 Примеры {domain_name} данных:")
    print("Train пример:")
    print(json.dumps(train_data[domain_name][0], indent=2, ensure_ascii=False))
    print("\nTest пример:")
    print(json.dumps(test_data[domain_name][0], indent=2, ensure_ascii=False))
    
    # Создание классификатора (только с быстрыми алгоритмами для теста)
    classifier = EmbeddingTermClassifier(
        model_name="Qwen/Qwen3-Embedding-4B",
        max_length=8192
    )
    
    # Обучение только на логистической регрессии для быстроты
    classifier.classifiers = {}  # Очищаем список классификаторов
    
    # Подготовка данных
    train_texts, test_texts, y, test_terms, df_train = classifier.prepare_data(
        train_data[domain_name][:10],  # Только первые 10 для теста
        test_data[domain_name][:5]     # Только первые 5 для теста
    )
    
    print(f"\n🔢 Размеры данных:")
    print(f"Train: {len(train_texts)} текстов")
    print(f"Test: {len(test_texts)} текстов")
    print(f"Классы: {y.shape[1]} уникальных типов")
    
    # Получение эмбеддингов (маленький размер для теста)
    print("\n🧮 Получение эмбеддингов...")
    X_train = classifier.get_embeddings(train_texts, batch_size=4)
    X_test = classifier.get_embeddings(test_texts, batch_size=4)
    
    print(f"Train embeddings: {X_train.shape}")
    print(f"Test embeddings: {X_test.shape}")
    
    # Быстрое обучение только логистической регрессии
    print("\n🤖 Обучение классификатора...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    
    lr_model = OneVsRestClassifier(LogisticRegression(max_iter=100))
    lr_model.fit(X_train.numpy(), y)
    classifier.classifiers['logistic_regression'] = lr_model
    
    # Предсказание
    print("\n🔮 Получение предсказаний...")
    predictions = classifier.predict(X_test, test_terms, test_data[domain_name][:5])
    
    # Показываем результаты
    print(f"\n✅ Результаты для {domain_name}:")
    for name, pred_data in predictions.items():
        print(f"\n{name.upper()}:")
        for i, pred in enumerate(pred_data):
            print(f"  {pred}")
            if i >= 2:  # Показываем только первые 3
                print("  ...")
                break
    
    # Проверяем формат
    print(f"\n🔍 Проверка формата:")
    sample_pred = predictions['logistic_regression'][0]
    required_keys = ['id', 'types']
    
    for key in required_keys:
        if key in sample_pred:
            print(f"  ✅ {key}: {type(sample_pred[key])}")
        else:
            print(f"  ❌ {key}: отсутствует!")
    
    # Проверяем что id корректный
    if 'id' in sample_pred:
        original_id = test_data[domain_name][0]['id']
        predicted_id = sample_pred['id']
        if original_id == predicted_id:
            print(f"  ✅ ID соответствует: {original_id}")
        else:
            print(f"  ❌ ID не соответствует: {original_id} != {predicted_id}")
    
    print(f"\n🎉 Тест {domain_name} завершен!")

if __name__ == "__main__":
    test_single_domain("MatOnto") 