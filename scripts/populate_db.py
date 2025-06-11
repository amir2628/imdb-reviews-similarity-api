import os
import time
import torch
from datasets import load_dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from app.models.review import Review
from app.dependencies import SessionLocal

def wait_for_model():
    """Умное ожидание готовности обученной модели без жестких таймаутов"""
    model_path = './fine_tuned_model'
    config_file = os.path.join(model_path, 'config.json')
    model_file = os.path.join(model_path, 'pytorch_model.bin')
    tokenizer_file = os.path.join(model_path, 'special_tokens_map.json')
    
    waited = 0
    check_interval = 15  # Проверяем каждые 15 секунд
    
    print("Ожидание завершения обучения модели...")
    print("Это может занять 10-30 минут в зависимости от производительности системы.")
    print("Прогресс обучения можно отслеживать в логах train сервиса.")
    
    while True:
        # Проверяем существование директории модели
        if os.path.exists(model_path):
            print(f"Найдена директория модели: {model_path}")
            
            # Проверяем наличие всех необходимых файлов модели
            required_files = [config_file, model_file, tokenizer_file]
            existing_files = [f for f in required_files if os.path.exists(f)]
            
            print(f"Найдено файлов модели: {len(existing_files)}/{len(required_files)}")
            
            if len(existing_files) == len(required_files):
                # Проверяем, что файлы не пустые и полностью записаны
                try:
                    # Попытка загрузить модель для проверки целостности
                    print("Проверка целостности модели...")
                    model = DistilBertForSequenceClassification.from_pretrained(
                        model_path, 
                        local_files_only=True
                    )
                    tokenizer = DistilBertTokenizer.from_pretrained(
                        model_path, 
                        local_files_only=True
                    )
                    print("Модель успешно загружена и проверена!")
                    return model, tokenizer
                    
                except Exception as e:
                    print(f"Модель еще не готова (ошибка загрузки): {e}")
                    print("Продолжаем ожидание...")
            else:
                missing_files = [f for f in required_files if not os.path.exists(f)]
                print(f"Ожидаем файлы: {[os.path.basename(f) for f in missing_files]}")
        else:
            print(f"Ожидание создания директории модели... ({waited} секунд)")
        
        # Ожидание перед следующей проверкой
        time.sleep(check_interval)
        waited += check_interval
        
        # Информативный вывод каждую минуту
        if waited % 60 == 0:
            minutes = waited // 60
            print(f"Прошло {minutes} мин. Продолжаем ожидание завершения обучения...")

def get_imdb_reviews():
    """Получение 1000 отзывов из датасета IMDB"""
    try:
        print("Загрузка датасета IMDB для извлечения векторных представлений...")
        dataset = load_dataset("imdb")
        
        # Берем первые 1000 отзывов из тестового набора
        reviews = dataset["test"][:1000]["text"]
        print(f"Загружено {len(reviews)} отзывов из датасета IMDB")
        return reviews
        
    except Exception as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить датасет IMDB: {e}")
        print("Извлечение векторных представлений невозможно без настоящих IMDB отзывов.")
        raise e

def main():
    print("Начинается заполнение базы данных векторными представлениями...")
    
    # Проверка наличия данных в базе
    try:
        with SessionLocal() as session:
            existing_count = session.query(Review).count()
            if existing_count > 0:
                print(f"База данных уже содержит {existing_count} отзывов. Пропуск заполнения...")
                return
    except Exception as e:
        print(f"Предупреждение: Не удалось проверить состояние базы данных: {e}")
        print("Продолжаем с заполнением...")
    
    # Умное ожидание готовности модели
    try:
        model, tokenizer = wait_for_model()
    except KeyboardInterrupt:
        print("\n Процесс прерван пользователем")
        return
    except Exception as e:
        print(f"Неожиданная ошибка при ожидании модели: {e}")
        return
    
    # Установка модели в режим оценки
    model.eval()
    print("Модель переведена в режим оценки")
    
    # Получение 1000 IMDB отзывов
    try:
        reviews = get_imdb_reviews()
    except Exception as e:
        print(f"Ошибка загрузки датасета: {e}")
        return
    
    print(f"Извлечение векторных представлений для {len(reviews)} отзывов...")
    print("Это займет несколько минут...")
    
    # Генерация векторов и сохранение в базу данных
    successful_inserts = 0
    failed_inserts = 0
    
    with torch.no_grad():
        for i, review in enumerate(reviews):
            # Прогресс каждые 100 отзывов
            if (i + 1) % 100 == 0:
                print(f"Обработано {i+1}/{len(reviews)} отзывов... "
                      f"(Успешно: {successful_inserts}, Ошибок: {failed_inserts})")
            
            try:
                # Токенизация и генерация эмбеддинга
                inputs = tokenizer(review, return_tensors="pt", truncation=True, max_length=512)
                outputs = model.distilbert(**inputs)
                
                # Извлечение последнего скрытого слоя (CLS токен)
                embedding = outputs.last_hidden_state[:, 0, :].detach().numpy().tolist()[0]
                
                # Сохранение в базу данных
                with SessionLocal() as session:
                    db_review = Review(text=review, vector=embedding)
                    session.add(db_review)
                    session.commit()
                    successful_inserts += 1
                    
            except Exception as e:
                print(f"Ошибка обработки отзыва {i}: {e}")
                failed_inserts += 1
                continue
    
    print(f"Заполнение базы данных успешно завершено!")
    print(f"Статистика:")
    print(f"   - Всего отзывов: {len(reviews)}")
    print(f"   - Успешно сохранено: {successful_inserts}")
    print(f"   - Ошибок: {failed_inserts}")

if __name__ == "__main__":
    main()