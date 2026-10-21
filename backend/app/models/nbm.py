"""Modelos del modulo NBM (Normal Behavior Model).

Persisten los experimentos de comparacion de algoritmos y los datasets
generados/extraidos para trazabilidad y reproducibilidad.
"""
import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, Index, Integer, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Experimento(Base):
    """Cada corrida del runner NBM registra aqui sus metricas."""

    __tablename__ = "experimentos"
    __table_args__ = (
        Index("idx_nbm_exp_algoritmo", "algoritmo"),
        Index("idx_nbm_exp_fecha", "fecha"),
        {"schema": "nbm"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    fecha: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    algoritmo: Mapped[str] = mapped_column(String(50), nullable=False)
    hiperparametros: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    precision: Mapped[Decimal | None] = mapped_column(Numeric(6, 4))
    recall: Mapped[Decimal | None] = mapped_column(Numeric(6, 4))
    f1_score: Mapped[Decimal | None] = mapped_column(Numeric(6, 4))
    fpr: Mapped[Decimal | None] = mapped_column(Numeric(6, 4))
    auc_roc: Mapped[Decimal | None] = mapped_column(Numeric(6, 4))

    tp: Mapped[int | None] = mapped_column(Integer)
    tn: Mapped[int | None] = mapped_column(Integer)
    fp: Mapped[int | None] = mapped_column(Integer)
    fn: Mapped[int | None] = mapped_column(Integer)

    tiempo_entrenamiento_s: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    tiempo_prediccion_s: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    n_train: Mapped[int | None] = mapped_column(Integer)
    n_test: Mapped[int | None] = mapped_column(Integer)

    dataset_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    notas: Mapped[str | None] = mapped_column(Text)


class DatasetGenerado(Base):
    """Registro del dataset usado (sintetico o extraido de security_logs)."""

    __tablename__ = "datasets_generados"
    __table_args__ = ({"schema": "nbm"},)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    fecha: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    fuente: Mapped[str] = mapped_column(String(50), nullable=False)
    n_registros: Mapped[int] = mapped_column(Integer, nullable=False)
    n_anomalias: Mapped[int] = mapped_column(Integer, nullable=False)
    n_features: Mapped[int] = mapped_column(Integer, nullable=False)
    hash_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    ruta_csv: Mapped[str | None] = mapped_column(Text)
    seed: Mapped[int | None] = mapped_column(Integer)
    descripcion: Mapped[str | None] = mapped_column(Text)