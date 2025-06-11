# 🎬 IMDB Reviews Similarity API 
**English version at the end**

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-FFD21E?style=for-the-badge&logoColor=black)
![Celery](https://img.shields.io/badge/Celery-37B24D?style=for-the-badge&logo=celery&logoColor=white)

</div>


## 🇷🇺 Русская версия

### 📝 Описание

Сервис на FastAPI для поиска похожих отзывов на фильмы с использованием машинного обучения. Система дообучает модель DistilBERT на датасете IMDB, сохраняет векторные представления текстов в PostgreSQL и предоставляет API для поиска семантически похожих отзывов.

### ⚠️ Обратите внимание!

Если вы получаете результаты, которые семантически не похожи на ваш отзыв, из-за использования ограниченного набора данных, вам необходимо добавить несколько семантически похожих отзывов к вашему фактическому Отзыву. Если ваш отзыв положительный, а вы получаете отрицательные результаты, то сначала вам нужно добавить несколько положительных отзывов в базу данных (и наоборот).

### 🏗️ Архитектура

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    FastAPI      │    │     Celery      │    │   PostgreSQL    │
│   (Web API)     │◄──►│   (ML Tasks)    │◄──►│   + pgvector    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                     ┌─────────────────┐
                     │      Redis      │
                     │   (Message      │
                     │    Broker)      │
                     └─────────────────┘
```

### 🛠️ Технологический стек

- **🐍 Python 3.11+** - Основной язык программирования
- **⚡ FastAPI** - Веб-фреймворк для создания API
- **🤖 DistilBERT** - Предобученная модель для обработки текста
- **🗄️ PostgreSQL + pgvector** - База данных с поддержкой векторного поиска
- **🔄 Celery + Redis** - Асинхронная обработка задач
- **📦 Docker Compose** - Оркестрация контейнеров
- **🧠 PyTorch + Transformers** - Машинное обучение

### 🚀 Быстрый старт

#### Предварительные требования

- Docker и Docker Compose
- Git

#### Установка и запуск

1. **Клонирование репозитория:**
```bash
git clone <repository-url>
cd imdb-reviews-similarity
```

2. **Запуск через Docker Compose:**
```bash
docker-compose up --build
```

3. **Ожидание инициализации:**
   - 🗄️ Инициализация PostgreSQL
   - 🤖 Дообучение модели DistilBERT
   - 📊 Заполнение базы данных векторами
   - 🌐 Запуск веб-сервера

4. **Проверка готовности:**
```bash
curl http://localhost:8000/docs
```

### 📡 API Endpoints

#### 1. Добавление отзыва
```http
POST /add_review
Content-Type: application/json

{
    "text": "This movie was absolutely amazing! Great acting and storyline."
}
```

**Ответ:**
```json
{
    "id": 1,
    "text": "This movie was absolutely amazing! Great acting and storyline."
}
```

#### 2. Поиск похожих отзывов
```http
POST /find_similar
Content-Type: application/json

{
    "text": "I loved this film!"
}
```

**Ответ:**
```json
{
    "task_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

#### 3. Проверка статуса задачи
```http
GET /status/{task_id}
```

**Ответ (в процессе):**
```json
{
    "status": "pending",
    "result": null
}
```

**Ответ (завершено):**
```json
{
    "status": "completed",
    "result": [
        "I really enjoyed this movie! The plot was engaging.",
        "Great film with excellent character development.",
        "Loved every minute of this masterpiece!"
    ]
}
```

### 🧪 Тестирование API

#### Использование curl

```bash
# Добавление отзыва
curl -X POST "http://localhost:8000/add_review" \
     -H "Content-Type: application/json" \
     -d '{"text":"This movie was incredible! Best film I have seen this year."}'

# Поиск похожих отзывов
curl -X POST "http://localhost:8000/find_similar" \
     -H "Content-Type: application/json" \
     -d '{"text":"Amazing movie with great acting!"}'

# Проверка статуса (замените task_id на полученный)
curl -X GET "http://localhost:8000/status/YOUR_TASK_ID"
```

#### Использование Python

```python
import requests
import time

# Добавление отзыва
response = requests.post(
    "http://localhost:8000/add_review",
    json={"text": "Fantastic movie with amazing visuals!"}
)
print("Добавлен отзыв:", response.json())

# Поиск похожих отзывов
response = requests.post(
    "http://localhost:8000/find_similar",
    json={"text": "Great film with excellent story!"}
)
task_id = response.json()["task_id"]
print("ID задачи:", task_id)

# Проверка статуса
while True:
    response = requests.get(f"http://localhost:8000/status/{task_id}")
    status_data = response.json()
    
    if status_data["status"] == "completed":
        print("Похожие отзывы:", status_data["result"])
        break
    elif status_data["status"] == "failed":
        print("Ошибка:", status_data["result"])
        break
    
    print("Обработка...")
    time.sleep(2)
```

### 🔧 Конфигурация

#### Переменные окружения

| Переменная | Описание | Значение по умолчанию |
|-----------|----------|---------------------|
| `DATABASE_URL` | URL подключения к PostgreSQL | `postgresql://user:password@db/imdb_reviews` |
| `REDIS_URL` | URL подключения к Redis | `redis://redis:6379/0` |
| `PYTHONPATH` | Путь Python | `/app` |

#### Docker Compose сервисы

- **db** - PostgreSQL с расширением pgvector
- **redis** - Redis для очереди задач
- **init-db** - Инициализация таблиц базы данных
- **train** - Дообучение модели DistilBERT
- **populate** - Заполнение базы данных векторами
- **web** - FastAPI веб-сервер
- **celery** - Celery worker для обработки задач

### 🧠 Машинное обучение

#### Дообучение модели

1. **Загрузка датасета:** IMDB отзывы на фильмы (50,000 отзывов)
2. **Предобученная модель:** `distilbert-base-uncased`
3. **Тип задачи:** Классификация тональности (положительная/отрицательная)
4. **Параметры обучения:**
   - Эпохи: 2
   - Размер батча: 8 (обучение), 16 (валидация)
   - Максимальная длина: 256 токенов
   - Оптимизатор: AdamW

#### Векторизация текстов

- **Извлечение:** Последний скрытый слой DistilBERT
- **Размерность:** 768 (CLS токен)
- **Сохранение:** PostgreSQL с pgvector
- **Поиск:** Косинусная близость

### 📁 Структура проекта

```
imdb-reviews-similarity/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI приложение
│   ├── dependencies.py         # Подключения к БД
│   ├── models/
│   │   ├── __init__.py
│   │   └── review.py          # SQLAlchemy модели
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── review.py          # Pydantic схемы
│   └── tasks/
│       ├── __init__.py
│       ├── celery_app.py      # Конфигурация Celery
│       └── tasks.py           # Celery задачи
├── ml/
│   └── train.py               # Скрипт обучения модели
├── scripts/
│   ├── init_db.py            # Инициализация БД
│   └── populate_db.py        # Заполнение БД
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── init-pgvector.sql
└── README.md
```

### 🐛 Устранение неполадок

#### Проблемы с памятью
```bash
# Увеличение лимита памяти для обучения
docker-compose up --build train --memory=6g
```

#### Проблемы с подключением к БД
```bash
# Проверка статуса БД
docker-compose ps db
docker-compose logs db
```

#### Проблемы с Celery
```bash
# Логи Celery worker
docker-compose logs celery

# Перезапуск Celery
docker-compose restart celery
```

#### Проблемы с моделью
```bash
# Пересоздание модели
docker-compose down
docker volume rm imdb-reviews-similarity_postgres_data
docker-compose up --build
```

### 📊 Мониторинг

#### Логи сервисов
```bash
# Все сервисы
docker-compose logs -f

# Конкретный сервис
docker-compose logs -f web
docker-compose logs -f celery
docker-compose logs -f train
```

#### Статус задач Celery
```bash
# Подключение к контейнеру Celery
docker-compose exec celery bash

# Просмотр активных задач
celery -A app.tasks.celery_app inspect active
```

### 🔐 Безопасность

⚠️ **Важно:** Данная реализация предназначена для разработки и тестирования. Для продуктивного использования необходимо:

- Изменить пароли по умолчанию
- Настроить SSL/TLS
- Добавить аутентификацию и авторизацию
- Ограничить доступ к портам

### 🤝 Вклад в проект

1. Fork репозитория
2. Создайте feature branch (`git checkout -b feature/amazing-feature`)
3. Commit изменения (`git commit -m 'Add amazing feature'`)
4. Push в branch (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

---

## 🇺🇸 English Version

### 📝 Description

A FastAPI service for finding similar movie reviews using machine learning. The system fine-tunes a DistilBERT model on the IMDB dataset, stores vector representations of texts in PostgreSQL, and provides an API for finding semantically similar reviews.

### ⚠️ Note!

If you get results that are not semantically similar to your review, due to using a limited dataset, you have to add some semantically similar reviews to your actual review. If you review is positive, and you are getting negative results, then you have to add some positive reviews to the database first (and vise-versa).

### 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    FastAPI      │    │     Celery      │    │   PostgreSQL    │
│   (Web API)     │◄──►│   (ML Tasks)    │◄──►│   + pgvector    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                     ┌─────────────────┐
                     │      Redis      │
                     │   (Message      │
                     │    Broker)      │
                     └─────────────────┘
```

### 🛠️ Tech Stack

- **🐍 Python 3.11+** - Main programming language
- **⚡ FastAPI** - Web framework for API creation
- **🤖 DistilBERT** - Pre-trained model for text processing
- **🗄️ PostgreSQL + pgvector** - Database with vector search support
- **🔄 Celery + Redis** - Asynchronous task processing
- **📦 Docker Compose** - Container orchestration
- **🧠 PyTorch + Transformers** - Machine learning

### 🚀 Quick Start

#### Prerequisites

- Docker and Docker Compose
- Git

#### Installation and Running

1. **Clone the repository:**
```bash
git clone <repository-url>
cd imdb-reviews-similarity
```

2. **Run with Docker Compose:**
```bash
docker-compose up --build
```

3. **Wait for initialization:**
   - 🗄️ PostgreSQL initialization
   - 🤖 DistilBERT model fine-tuning
   - 📊 Database population with vectors
   - 🌐 Web server startup

4. **Check readiness:**
```bash
curl http://localhost:8000/docs
```

### 📡 API Endpoints

#### 1. Add Review
```http
POST /add_review
Content-Type: application/json

{
    "text": "This movie was absolutely amazing! Great acting and storyline."
}
```

**Response:**
```json
{
    "id": 1,
    "text": "This movie was absolutely amazing! Great acting and storyline."
}
```

#### 2. Find Similar Reviews
```http
POST /find_similar
Content-Type: application/json

{
    "text": "I loved this film!"
}
```

**Response:**
```json
{
    "task_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

#### 3. Check Task Status
```http
GET /status/{task_id}
```

**Response (pending):**
```json
{
    "status": "pending",
    "result": null
}
```

**Response (completed):**
```json
{
    "status": "completed",
    "result": [
        "I really enjoyed this movie! The plot was engaging.",
        "Great film with excellent character development.",
        "Loved every minute of this masterpiece!"
    ]
}
```

### 🧪 API Testing

#### Using curl

```bash
# Add review
curl -X POST "http://localhost:8000/add_review" \
     -H "Content-Type: application/json" \
     -d '{"text":"This movie was incredible! Best film I have seen this year."}'

# Find similar reviews
curl -X POST "http://localhost:8000/find_similar" \
     -H "Content-Type: application/json" \
     -d '{"text":"Amazing movie with great acting!"}'

# Check status (replace task_id with received one)
curl -X GET "http://localhost:8000/status/YOUR_TASK_ID"
```

#### Using Python

```python
import requests
import time

# Add review
response = requests.post(
    "http://localhost:8000/add_review",
    json={"text": "Fantastic movie with amazing visuals!"}
)
print("Added review:", response.json())

# Find similar reviews
response = requests.post(
    "http://localhost:8000/find_similar",
    json={"text": "Great film with excellent story!"}
)
task_id = response.json()["task_id"]
print("Task ID:", task_id)

# Check status
while True:
    response = requests.get(f"http://localhost:8000/status/{task_id}")
    status_data = response.json()
    
    if status_data["status"] == "completed":
        print("Similar reviews:", status_data["result"])
        break
    elif status_data["status"] == "failed":
        print("Error:", status_data["result"])
        break
    
    print("Processing...")
    time.sleep(2)
```

### 🔧 Configuration

#### Environment Variables

| Variable | Description | Default Value |
|----------|-------------|---------------|
| `DATABASE_URL` | PostgreSQL connection URL | `postgresql://user:password@db/imdb_reviews` |
| `REDIS_URL` | Redis connection URL | `redis://redis:6379/0` |
| `PYTHONPATH` | Python path | `/app` |

#### Docker Compose Services

- **db** - PostgreSQL with pgvector extension
- **redis** - Redis for task queue
- **init-db** - Database tables initialization
- **train** - DistilBERT model fine-tuning
- **populate** - Database population with vectors
- **web** - FastAPI web server
- **celery** - Celery worker for task processing

### 🧠 Machine Learning

#### Model Fine-tuning

1. **Dataset Loading:** IMDB movie reviews (50,000 reviews)
2. **Pre-trained Model:** `distilbert-base-uncased`
3. **Task Type:** Sentiment classification (positive/negative)
4. **Training Parameters:**
   - Epochs: 2
   - Batch size: 8 (training), 16 (validation)
   - Max length: 256 tokens
   - Optimizer: AdamW

#### Text Vectorization

- **Extraction:** Last hidden layer of DistilBERT
- **Dimensionality:** 768 (CLS token)
- **Storage:** PostgreSQL with pgvector
- **Search:** Cosine similarity

### 📁 Project Structure

```
imdb-reviews-similarity/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── dependencies.py         # Database connections
│   ├── models/
│   │   ├── __init__.py
│   │   └── review.py          # SQLAlchemy models
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── review.py          # Pydantic schemas
│   └── tasks/
│       ├── __init__.py
│       ├── celery_app.py      # Celery configuration
│       └── tasks.py           # Celery tasks
├── ml/
│   └── train.py               # Model training script
├── scripts/
│   ├── init_db.py            # Database initialization
│   └── populate_db.py        # Database population
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── init-pgvector.sql
└── README.md
```

### 🐛 Troubleshooting

#### Memory Issues
```bash
# Increase memory limit for training
docker-compose up --build train --memory=6g
```

#### Database Connection Issues
```bash
# Check database status
docker-compose ps db
docker-compose logs db
```

#### Celery Issues
```bash
# Celery worker logs
docker-compose logs celery

# Restart Celery
docker-compose restart celery
```

#### Model Issues
```bash
# Recreate model
docker-compose down
docker volume rm imdb-reviews-similarity_postgres_data
docker-compose up --build
```

### 📊 Monitoring

#### Service Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f web
docker-compose logs -f celery
docker-compose logs -f train
```

#### Celery Task Status
```bash
# Connect to Celery container
docker-compose exec celery bash

# View active tasks
celery -A app.tasks.celery_app inspect active
```

### 🔐 Security

⚠️ **Important:** This implementation is intended for development and testing. For production use, you need to:

- Change default passwords
- Configure SSL/TLS
- Add authentication and authorization
- Restrict port access

### 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License.

## 👥 Authors

- **Amir Bahrami** - [My GitHub](https://github.com/amir2628)

## 🙏 Acknowledgments

- Hugging Face for the Transformers library
- IMDB for the dataset
- The open-source community