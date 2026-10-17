from datetime import datetime
from pathlib import Path
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from ..config import settings

Base = declarative_base()

class Analysis(Base):  # non-identifying data only
    __tablename__ = "analyses"
    id = Column(Integer, primary_key=True)
    created = Column(DateTime, default=datetime.utcnow)
    sha256 = Column(String); modality = Column(String)
    width = Column(Integer); height = Column(Integer); quality = Column(Integer)
    model_version = Column(String); prediction = Column(String)
    confidence = Column(Float); total_ms = Column(Float)

if settings.db_url.startswith("sqlite:///"):
    Path(settings.db_url[10:]).parent.mkdir(parents=True, exist_ok=True)
engine = create_engine(settings.db_url)
Session = sessionmaker(engine)
Base.metadata.create_all(engine)
