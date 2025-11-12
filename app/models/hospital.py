from sqlalchemy import Column, String, Integer, Float, DateTime
from sqlalchemy.sql import func
from app.database import Base


class Hospital(Base):
    __tablename__ = "hospitals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, index=True)
    address = Column(String, nullable=True)
    city = Column(String, nullable=True, index=True)
    state = Column(String, nullable=True, index=True)
    country = Column(String, nullable=True, index=True)
    zip_code = Column(String, nullable=True)
    
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    
    phone_number = Column(String, nullable=True)
    email = Column(String, nullable=True)
    website = Column(String, nullable=True)
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
