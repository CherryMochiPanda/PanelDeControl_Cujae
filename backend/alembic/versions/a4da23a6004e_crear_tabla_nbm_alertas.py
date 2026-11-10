"""crear tabla nbm.alertas

Revision ID: a4da23a6004e
Revises: f75e8f883d57
Create Date: 2026-11-09 21:07:54.618763
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = 'a4da23a6004e'
down_revision = 'f75e8f883d57'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "alertas",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("fecha", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("usuario_hash", sa.String(64), nullable=False),
        sa.Column("componente", sa.String(50), nullable=False),
        sa.Column("score", sa.Numeric(10, 6), nullable=False),
        sa.Column("severidad", sa.String(20), nullable=False),
        sa.Column("tipo_anomalia", sa.String(50), nullable=False),
        sa.Column("estado", sa.String(20), nullable=False, server_default="nueva"),
        sa.Column("detalles", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("dataset_id", postgresql.UUID(as_uuid=True), nullable=True),
        schema="nbm",
    )
    op.create_index("idx_nbm_alertas_fecha", "alertas", ["fecha"], schema="nbm")
    op.create_index("idx_nbm_alertas_severidad", "alertas", ["severidad"], schema="nbm")
    op.create_index("idx_nbm_alertas_estado", "alertas", ["estado"], schema="nbm")


def downgrade() -> None:
    op.drop_table("alertas", schema="nbm")
