# Скрипт для дообучения DistilBERT на датасете IMDB

import os
import time
import torch
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

def check_model_exists():
    """Проверка существования полной обученной модели"""
    model_path = "./fine_tuned_model"
    
    # Список необходимых файлов для полной модели
    required_files = [
        "config.json",           # Конфигурация модели
        "pytorch_model.bin",     # Веса модели
        "special_tokens_map.json",        # Токенизатор
        "tokenizer_config.json", # Конфигурация токенизатора
        "vocab.txt"              # Словарь
    ]
    
    # Проверяем существование директории
    if not os.path.exists(model_path):
        print(f"Директория модели {model_path} не существует")
        return False
    
    # Проверяем наличие всех необходимых файлов
    missing_files = []
    for file_name in required_files:
        file_path = os.path.join(model_path, file_name)
        if not os.path.exists(file_path):
            missing_files.append(file_name)
        elif os.path.getsize(file_path) == 0:
            # Проверяем, что файл не пустой
            missing_files.append(f"{file_name} (пустой)")
    
    if missing_files:
        print(f"Отсутствуют или повреждены файлы модели: {missing_files}")
        return False
    
    # Попытка загрузить модель для проверки целостности
    try:
        print("Проверка целостности существующей модели...")
        model = DistilBertForSequenceClassification.from_pretrained(
            model_path, 
            local_files_only=True
        )
        tokenizer = DistilBertTokenizer.from_pretrained(
            model_path, 
            local_files_only=True
        )
        print("✓ Существующая модель прошла проверку целостности")
        return True
        
    except Exception as e:
        print(f"Модель повреждена или неполная: {e}")
        return False

def safe_clean_incomplete_model():
    """Безопасная очистка неполной или поврежденной модели"""
    model_path = "./fine_tuned_model"
    if os.path.exists(model_path):
        print("Безопасное удаление неполной модели...")
        try:
            # Удаляем файлы по одному, чтобы избежать конфликтов с volume
            for root, dirs, files in os.walk(model_path, topdown=False):
                for file in files:
                    try:
                        os.remove(os.path.join(root, file))
                    except OSError as e:
                        print(f"Предупреждение: не удалось удалить файл {file}: {e}")
                for dir in dirs:
                    try:
                        os.rmdir(os.path.join(root, dir))
                    except OSError as e:
                        print(f"Предупреждение: не удалось удалить директорию {dir}: {e}")
            
            # Попытка удалить основную директорию
            try:
                os.rmdir(model_path)
                print("✓ Неполная модель успешно удалена")
            except OSError as e:
                print(f"Предупреждение: не удалось удалить основную директорию: {e}")
                print("Продолжаем с существующей директорией...")
                
        except Exception as e:
            print(f"Предупреждение при очистке: {e}")
            print("Продолжаем обучение...")

def create_training_marker():
    """Создает маркер начала обучения"""
    marker_path = "./training_in_progress.marker"
    with open(marker_path, 'w') as f:
        f.write(f"Training started at {time.time()}")
    print("✓ Маркер обучения создан")

def remove_training_marker():
    """Удаляет маркер обучения"""
    marker_path = "./training_in_progress.marker"
    if os.path.exists(marker_path):
        os.remove(marker_path)
        print("✓ Маркер обучения удален")

def main():
    print("Начинается проверка и обучение модели...")
    
    # Задержка для ожидания готовности базы данных
    time.sleep(10)
    
    # Проверка существования полной модели
    if check_model_exists():
        print("Полная обученная модель уже существует, обучение пропущено...")
        return
    else:
        print("Модель отсутствует или неполная, начинается обучение...")
        # Создаем маркер начала обучения
        create_training_marker()
        # Безопасно очищаем неполную модель если она есть
        safe_clean_incomplete_model()
    
    # Загрузка датасета IMDB
    print("Загрузка датасета IMDB...")
    try:
        print("Подключение к Hugging Face для загрузки датасета IMDB...")
        dataset = load_dataset("imdb")
        
        # Использование подмножества для ускорения обучения
        train_dataset = dataset["train"].select(range(1000))  # 1000 примеров для обучения
        eval_dataset = dataset["test"].select(range(200))     # 200 примеров для валидации
        
        print(f"Загружено {len(train_dataset)} обучающих и {len(eval_dataset)} тестовых примеров")
        
    except Exception as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить датасет IMDB: {e}")
        print("Обучение невозможно без настоящего датасета IMDB.")
        remove_training_marker()
        raise e
    
    # Инициализация токенизатора
    print("🔧 Инициализация токенизатора...")
    try:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        print("Токенизатор успешно загружен")
    except Exception as e:
        print(f"Ошибка загрузки токенизатора: {e}")
        remove_training_marker()
        raise e
    
    # Функция токенизации
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)
    
    # Токенизация датасетов
    print("Токенизация датасетов...")
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    eval_tokenized = eval_dataset.map(tokenize_function, batched=True)
    
    # Переименование столбца label в labels для совместимости с Trainer
    train_tokenized = train_tokenized.rename_column("label", "labels")
    eval_tokenized = eval_tokenized.rename_column("label", "labels")
    
    # Установка формата датасетов
    train_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    eval_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    print("Данные успешно подготовлены для обучения")
    
    # Загрузка модели для классификации
    print("Загрузка предобученной модели DistilBERT...")
    try:
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        print("Модель DistilBERT успешно загружена")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        remove_training_marker()
        raise e
    
    # Параметры обучения (согласно заданию: 1-2 эпохи)
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=2,  # 1-2 эпохи как указано в задании
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        save_strategy="no",
        evaluation_strategy="steps",
        eval_steps=200,
        report_to=[],
        push_to_hub=False,
        dataloader_pin_memory=False,
        remove_unused_columns=True,
        load_best_model_at_end=False,
        disable_tqdm=False,
    )
    
    # Инициализация тренера
    print("Инициализация тренера...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
    )
    
    # Дообучение модели на IMDB датасете
    print("Начинается дообучение DistilBERT на датасете IMDB...")
    print("Это займет 10-20 минут в зависимости от производительности системы...")
    
    try:
        trainer.train()
        print("Обучение успешно завершено!")
    except Exception as e:
        print(f"Ошибка во время обучения: {e}")
        remove_training_marker()
        raise e
    
    # Сохранение дообученной модели
    print("Сохранение дообученной модели...")
    try:
        os.makedirs("./fine_tuned_model", exist_ok=True)
        model.save_pretrained("./fine_tuned_model")
        tokenizer.save_pretrained("./fine_tuned_model")
        print("Модель и токенизатор сохранены")
    except Exception as e:
        print(f"Ошибка сохранения модели: {e}")
        remove_training_marker()
        raise e
    
    # Удаляем маркер обучения
    remove_training_marker()
    
    # Финальная проверка сохраненной модели
    if check_model_exists():
        print("Дообучение модели успешно завершено!")
        print("Модель корректно сохранена в директории ./fine_tuned_model")
        print("Система готова к работе!")
    else:
        print("ОШИБКА: Модель не была корректно сохранена!")
        raise Exception("Модель не была корректно сохранена")

if __name__ == "__main__":
    main()