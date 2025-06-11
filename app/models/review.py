# Модель для таблицы отзывов

from sqlalchemy import Column, Integer, Text
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class Review(Base):
    __tablename__ = 'reviews'
    
    id = Column(Integer, primary_key=True)
    text = Column(Text, nullable=False)
    vector = Column(Vector(768))  # Размерность вектора DistilBERT - 768