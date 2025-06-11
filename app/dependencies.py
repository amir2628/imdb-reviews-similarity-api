# Настройка подключения к базе данных
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Получение URL базы данных из переменной окружения
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/imdb_reviews")
ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

# Синхронный движок для Celery
sync_engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=sync_engine)

# Асинхронный движок для FastAPI
async_engine = create_async_engine(ASYNC_DATABASE_URL)
async_session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

# Зависимость для получения асинхронной сессии
async def get_db():
    async with async_session() as session:
        yield session