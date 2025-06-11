import asyncio
import time
from sqlalchemy import text
from app.models.review import Base
from app.dependencies import async_engine

async def create_tables():
    """Создание всех таблиц базы данных с проверкой готовности pgvector"""
    print("Инициализация базы данных...")
    
    # Ожидание полной готовности PostgreSQL
    max_retries = 30
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            async with async_engine.begin() as conn:
                # Проверяем доступность pgvector расширения
                result = await conn.execute(
                    text("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")
                )
                if result.fetchone() is None:
                    print("Расширение pgvector недоступно. Повторная попытка через 2 секунды...")
                    time.sleep(2)
                    retry_count += 1
                    continue
                
                print("Расширение pgvector доступно")
                
                # Создаем расширение pgvector
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                print("Расширение pgvector создано")
                
                # Создаем таблицы
                await conn.run_sync(Base.metadata.create_all)
                print("Таблицы базы данных успешно созданы!")
                return
                
        except Exception as e:
            print(f"Попытка {retry_count + 1}/{max_retries} не удалась: {e}")
            retry_count += 1
            if retry_count < max_retries:
                print("Повторная попытка через 2 секунды...")
                time.sleep(2)
            else:
                print("Не удалось инициализировать базу данных после всех попыток")
                raise e

if __name__ == "__main__":
    asyncio.run(create_tables())