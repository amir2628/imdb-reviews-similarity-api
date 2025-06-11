# Конфигурация Celery
import os
from celery import Celery

# Получение URL Redis из переменной окружения или использование значения по умолчанию
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

# Настройка Celery с Redis в качестве брокера
celery_app = Celery(
    'tasks',
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=['app.tasks.tasks']
)

# Обновление настроек Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)