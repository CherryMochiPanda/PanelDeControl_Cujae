"""schema experimentos y datasets

Revision ID: f75e8f883d57
Revises: d4e5f6a7b8c9
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "f75e8f883d57"
down_revision = "d4e5f6a7b8c9"  
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS nbm")

    op.create_table(
        "experimentos",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("fecha", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("algoritmo", sa.String(50), nullable=False),
        sa.Column("hiperparametros", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("precision", sa.Numeric(6, 4), nullable=True),
        sa.Column("recall", sa.Numeric(6, 4), nullable=True),
        sa.Column("f1_score", sa.Numeric(6, 4), nullable=True),
        sa.Column("fpr", sa.Numeric(6, 4), nullable=True),
        sa.Column("auc_roc", sa.Numeric(6, 4), nullable=True),
        sa.Column("tp", sa.Integer, nullable=True),
        sa.Column("tn", sa.Integer, nullable=True),
        sa.Column("fp", sa.Integer, nullable=True),
        sa.Column("fn", sa.Integer, nullable=True),
        sa.Column("tiempo_entrenamiento_s", sa.Numeric(10, 4), nullable=True),
        sa.Column("tiempo_prediccion_s", sa.Numeric(10, 4), nullable=True),
        sa.Column("n_train", sa.Integer, nullable=True),
        sa.Column("n_test", sa.Integer, nullable=True),
        sa.Column("dataset_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("notas", sa.Text, nullable=True),
        schema="nbm",
    )
    op.create_index("idx_nbm_exp_algoritmo", "experimentos", ["algoritmo"], schema="nbm")
    op.create_index("idx_nbm_exp_fecha", "experimentos", ["fecha"], schema="nbm")

    op.create_table(
        "datasets_generados",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("fecha", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("fuente", sa.String(50), nullable=False),  # "sintetico" | "security_logs"
        sa.Column("n_registros", sa.Integer, nullable=False),
        sa.Column("n_anomalias", sa.Integer, nullable=False),
        sa.Column("n_features", sa.Integer, nullable=False),
        sa.Column("hash_sha256", sa.String(64), nullable=False),
        sa.Column("ruta_csv", sa.Text, nullable=True),
        sa.Column("seed", sa.Integer, nullable=True),
        sa.Column("descripcion", sa.Text, nullable=True),
        schema="nbm",
    )


def downgrade() -> None:
    op.drop_table("datasets_generados", schema="nbm")
    op.drop_table("experimentos", schema="nbm")
    op.execute("DROP SCHEMA IF EXISTS nbm")