# Задачи для асинхронной обработки

import os
from app.tasks.celery_app import celery_app
from app.models.review import Review
from app.dependencies import SessionLocal
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from sqlalchemy import text as sql_text
import torch

# Глобальные переменные для модели и токенизатора
model = None
tokenizer = None

def load_model():
    """Загрузка модели и токенизатора, если они еще не загружены"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        model_path = './fine_tuned_model'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Дообученная модель не найдена по пути {model_path}. Пожалуйста, сначала обучите модель.")
        
        print("Загрузка модели и токенизатора...")
        model = DistilBertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        tokenizer = DistilBertTokenizer.from_pretrained(model_path, local_files_only=True)
        model.eval()
        print("Модель и токенизатор успешно загружены!")

@celery_app.task
def find_similar_reviews(input_text: str):
    """Поиск похожих отзывов по входному тексту"""
    try:
        # Загрузка модели, если она еще не загружена
        load_model()
        
        # Генерация вектора для входного текста
        with torch.no_grad():
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
            outputs = model.distilbert(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].detach().numpy().tolist()[0]
        
        # Поиск похожих отзывов в базе данных
        with SessionLocal() as session:
            # Использование правильного синтаксиса для pgvector
            # Преобразуем список в строку формата PostgreSQL array
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'
            
            query = session.query(Review).order_by(
                sql_text('vector <=> :embedding')
            ).params(embedding=embedding_str).limit(3)
            
            similar_reviews = query.all()
        
        # Возврат списка текстов похожих отзывов
        return [review.text for review in similar_reviews]
        
    except Exception as e:
        print(f"Ошибка в задаче find_similar_reviews: {str(e)}")
        # Возвращаем пустой список в случае ошибки, чтобы соответствовать схеме
        return []