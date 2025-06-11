# Основной файл FastAPI

import os
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.dependencies import get_db
from app.schemas.review import ReviewCreate, ReviewResponse, FindSimilarRequest, TaskResponse, StatusResponse
from app.models.review import Review
from app.tasks.tasks import find_similar_reviews
from celery.result import AsyncResult
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

app = FastAPI(
    title="IMDB Reviews Similarity API", 
    description="API для поиска похожих отзывов на фильмы",
    version="1.0.0"
)

# Глобальные переменные для модели и токенизатора
model = None
tokenizer = None

def check_model_ready():
    """Проверка готовности модели"""
    model_path = './fine_tuned_model'
    required_files = [
        'config.json',
        'pytorch_model.bin', 
        'special_tokens_map.json',
        'tokenizer_config.json',
        'vocab.txt'
    ]
    
    if not os.path.exists(model_path):
        return False
        
    for file_name in required_files:
        file_path = os.path.join(model_path, file_name)
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return False
    
    return True

def load_model():
    """Загрузка модели и токенизатора, если они еще не загружены"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        model_path = './fine_tuned_model'
        
        if not check_model_ready():
            raise FileNotFoundError(f"Дообученная модель не готова по пути {model_path}. Модель все еще обучается или произошла ошибка.")
        
        print("🤖 Загрузка модели и токенизатора...")
        try:
            model = DistilBertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
            tokenizer = DistilBertTokenizer.from_pretrained(model_path, local_files_only=True)
            model.eval()
            print("Модель и токенизатор успешно загружены!")
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            raise

@app.on_event("startup")
async def startup_event():
    """Проверка готовности при запуске (без принудительной загрузки)"""
    print("Запуск IMDB Reviews Similarity API...")
    
    # Проверяем, идет ли обучение
    if os.path.exists("./training_in_progress.marker"):
        print("Обнаружен маркер обучения - модель все еще обучается")
        print("API будет доступен после завершения обучения")
        return
    
    # Проверяем готовность модели без загрузки
    if check_model_ready():
        print("Обученная модель найдена и готова к использованию")
        try:
            load_model()
            print("🎉 API полностью готов к работе!")
        except Exception as e:
            print(f"Модель найдена, но возникла ошибка загрузки: {e}")
            print("Модель будет загружена при первом запросе")
    else:
        print("Обученная модель не найдена")
        print("Модель будет загружена после завершения обучения")

@app.get("/health")
async def health_check():
    """Проверка состояния сервиса"""
    status = {
        "status": "healthy",
        "training_in_progress": os.path.exists("./training_in_progress.marker"),
        "model_ready": check_model_ready(),
        "model_loaded": model is not None and tokenizer is not None
    }
    return status

@app.post("/add_review", response_model=ReviewResponse)
async def add_review(review: ReviewCreate, db: AsyncSession = Depends(get_db)):
    """Добавление нового отзыва в базу данных с генерацией его векторного представления"""
    # Проверяем, идет ли обучение
    if os.path.exists("./training_in_progress.marker"):
        raise HTTPException(
            status_code=503, 
            detail="Модель все еще обучается. Пожалуйста, подождите завершения обучения."
        )
    
    # Загрузка модели, если она еще не загружена
    if model is None or tokenizer is None:
        try:
            load_model()
        except FileNotFoundError:
            raise HTTPException(
                status_code=503, 
                detail="Модель недоступна или все еще обучается. Пожалуйста, проверьте статус через /health"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка загрузки модели: {str(e)}"
            )
    
    # Генерация вектора для отзыва
    try:
        with torch.no_grad():
            inputs = tokenizer(review.text, return_tensors="pt", truncation=True, max_length=512)
            outputs = model.distilbert(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].detach().numpy().tolist()[0]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка генерации векторного представления: {str(e)}"
        )
    
    # Сохранение в базу данных
    try:
        db_review = Review(text=review.text, vector=embedding)
        db.add(db_review)
        await db.commit()
        await db.refresh(db_review)
        return db_review
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка сохранения в базу данных: {str(e)}"
        )

@app.post("/find_similar", response_model=TaskResponse)
async def find_similar(request: FindSimilarRequest):
    """Поиск похожих отзывов через асинхронную обработку в Celery"""
    # Проверяем, идет ли обучение
    if os.path.exists("./training_in_progress.marker"):
        raise HTTPException(
            status_code=503, 
            detail="Модель все еще обучается. Пожалуйста, подождите завершения обучения."
        )
    
    # Проверяем готовность модели
    if not check_model_ready():
        raise HTTPException(
            status_code=503,
            detail="Модель недоступна. Пожалуйста, проверьте статус через /health"
        )
    
    # Создание асинхронной задачи в Celery
    try:
        task = find_similar_reviews.delay(request.text)
        return {"task_id": task.id}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка создания задачи: {str(e)}"
        )

@app.get("/status/{task_id}", response_model=StatusResponse)
async def get_status(task_id: str):
    """Проверка статуса выполнения задачи Celery"""
    try:
        # Проверка статуса задачи
        task_result = AsyncResult(task_id)
        
        if task_result.ready():
            if task_result.successful():
                result = task_result.get()
                return {"status": "completed", "result": result}
            else:
                # В случае ошибки возвращаем информацию об ошибке
                error_info = str(task_result.info) if task_result.info else "Неизвестная ошибка"
                return {"status": "failed", "result": error_info}
        else:
            return {"status": "pending", "result": None}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка проверки статуса задачи: {str(e)}"
        )