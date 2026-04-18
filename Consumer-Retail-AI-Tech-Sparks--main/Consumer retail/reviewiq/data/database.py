"""
ReviewIQ Database Models - SQLAlchemy ORM for Pre-Compute Strategy
"""

import os
from datetime import datetime
from typing import Generator

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
from sqlalchemy.pool import QueuePool
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{os.path.join(os.path.dirname(__file__), 'database', 'reviewiq.db')}"
)

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Review(Base):
    """Raw review data with trust scoring."""
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)
    product_name = Column(String(255), nullable=False, index=True)
    original_text = Column(Text, nullable=False)
    clean_text = Column(Text, nullable=True)
    language = Column(String(10), nullable=True, index=True)
    trust_score = Column(Float, nullable=True)
    is_suspicious = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    features = relationship("FeatureExtraction", back_populates="review", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_reviews_product_created", "product_name", "created_at"),
    )


class FeatureExtraction(Base):
    """Structured insights from LLM extraction per review."""
    __tablename__ = "feature_extractions"

    id = Column(Integer, primary_key=True, index=True)
    review_id = Column(Integer, ForeignKey("reviews.id", ondelete="CASCADE"), nullable=False, index=True)
    feature_name = Column(String(255), nullable=False, index=True)
    sentiment = Column(String(20), nullable=False)  # positive, negative, neutral, mixed
    intensity = Column(Float, nullable=False)  # 0.0 to 1.0
    confidence = Column(Float, nullable=False)  # LLM confidence score
    ambiguity_flag = Column(Boolean, default=False)
    sarcasm_flag = Column(Boolean, default=False)
    evidence = Column(Text, nullable=True)  # Supporting text snippets

    review = relationship("Review", back_populates="features")

    __table_args__ = (
        Index("idx_features_review_feature", "review_id", "feature_name"),
        Index("idx_features_sentiment", "sentiment", "confidence"),
    )


class TrendWindow(Base):
    """Aggregated complaint/praise rates per feature per time window."""
    __tablename__ = "trend_windows"

    id = Column(Integer, primary_key=True, index=True)
    product_name = Column(String(255), nullable=False, index=True)
    feature_name = Column(String(255), nullable=False, index=True)
    window_label = Column(String(50), nullable=False)  # e.g., "2024-W01", "2024-01"
    complaint_rate = Column(Float, nullable=False)
    praise_rate = Column(Float, nullable=False)
    z_score = Column(Float, nullable=True)  # Statistical significance
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (
        Index("idx_trends_product_feature_window", "product_name", "feature_name", "window_label"),
        Index("idx_trends_zscore", "z_score", "complaint_rate"),
    )


class Escalation(Base):
    """Priority-ranked issues requiring immediate attention."""
    __tablename__ = "escalations"

    id = Column(Integer, primary_key=True, index=True)
    product_name = Column(String(255), nullable=False, index=True)
    feature_name = Column(String(255), nullable=False, index=True)
    severity_score = Column(Float, nullable=False)  # Computed severity 0-1
    lifecycle_stage = Column(String(50), nullable=False)  # emerging, active, fading, chronic
    trend_direction = Column(String(20), nullable=False)  # improving, stable, worsening
    priority_rank = Column(Integer, nullable=False, index=True)

    __table_args__ = (
        Index("idx_escalations_product_priority", "product_name", "priority_rank"),
        Index("idx_escalations_lifecycle", "lifecycle_stage", "severity_score"),
    )


class ActionBrief(Base):
    """Generated PDF action briefs with metadata."""
    __tablename__ = "action_briefs"

    id = Column(Integer, primary_key=True, index=True)
    product_name = Column(String(255), nullable=False, index=True)
    generated_text = Column(Text, nullable=False)
    pdf_path = Column(String(512), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (
        Index("idx_briefs_product_created", "product_name", "created_at"),
    )


def get_db() -> Generator[Session, None, None]:
    """Dependency for FastAPI routes to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize database - create all tables."""
    Base.metadata.create_all(bind=engine)


def clear_db() -> None:
    """Clear all data - drop and recreate all tables."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")
