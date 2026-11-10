"""Generacion de reportes Excel de anomalias (caso de uso: Generar Reportes).

Consulta la tabla nbm.alertas en un rango de fechas, calcula estadisticas
(resumen ejecutivo) y genera un .xlsx con 3 hojas:
1. Resumen: totales, distribucion por severidad, componente, usuario.
2. Incidentes: listado detallado de alertas.
3. Detalles: JSON con los features del evento que disparo cada alerta.
"""
from __future__ import annotations

import io
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.nbm import Alerta

log = logging.getLogger("nbm.report")


SEVERIDADES = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]


async def generar_reporte_anomalias(
    session: AsyncSession,
    *,
    desde: datetime | None = None,
    hasta: datetime | None = None,
) -> bytes | None:
    """Genera el Excel del reporte. Devuelve None si no hay alertas.

    Args:
        session: sesion async.
        desde: fecha inicio (default: hace 7 dias).
        hasta: fecha fin (default: ahora).

    Returns:
        Bytes del .xlsx o None si no hay alertas en el rango.
    """
    if hasta is None:
        hasta = datetime.now(timezone.utc)
    if desde is None:
        desde = hasta - timedelta(days=7)

    stmt = (
        select(Alerta)
        .where(Alerta.fecha >= desde, Alerta.fecha <= hasta)
        .order_by(Alerta.fecha.desc())
    )
    res = await session.execute(stmt)
    alertas = list(res.scalars().all())

    if not alertas:
        return None

    df = pd.DataFrame([
        {
            "Fecha": a.fecha.strftime("%Y-%m-%d %H:%M:%S"),
            "Usuario (hash)": a.usuario_hash[:16] + "...",
            "Componente": a.componente,
            "Score": float(a.score),
            "Severidad": a.severidad,
            "Tipo de anomalía": a.tipo_anomalia,
            "Estado": a.estado,
            "Detalles": str(a.detalles),
        }
        for a in alertas
    ])

    # --- Hoja 1: Resumen ejecutivo ---
    total = len(df)
    por_sev = df["Severidad"].value_counts().reindex(SEVERIDADES, fill_value=0)
    por_comp = df["Componente"].value_counts()
    por_tipo = df["Tipo de anomalía"].value_counts()
    por_usuario = df["Usuario (hash)"].value_counts().head(10)

    resumen = pd.DataFrame({
        "Métrica": [
            "Período analizado",
            "Total de anomalías detectadas",
            "",
            "Distribución por severidad",
            "  CRITICAL", "  HIGH", "  MEDIUM", "  LOW",
            "",
            "Componente más afectado",
            "Tipo de anomalía más frecuente",
            "Usuario más afectado (hash)",
            "",
            "Estado de las alertas",
        ],
        "Valor": [
            f"{desde.strftime('%Y-%m-%d')} a {hasta.strftime('%Y-%m-%d')}",
            total,
            "",
            "",
            int(por_sev.get("CRITICAL", 0)),
            int(por_sev.get("HIGH", 0)),
            int(por_sev.get("MEDIUM", 0)),
            int(por_sev.get("LOW", 0)),
            "",
            f"{por_comp.index[0]} ({por_comp.iloc[0]})" if len(por_comp) else "N/A",
            f"{por_tipo.index[0]} ({por_tipo.iloc[0]})" if len(por_tipo) else "N/A",
            f"{por_usuario.index[0]} ({por_usuario.iloc[0]})" if len(por_usuario) else "N/A",
            "",
            ", ".join(f"{k}={v}" for k, v in df["Estado"].value_counts().items()),
        ],
    })

    # --- Escribir Excel ---
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        resumen.to_excel(writer, sheet_name="Resumen", index=False)
        df.to_excel(writer, sheet_name="Incidentes", index=False)

        # Formato de encabezados y anchos de columna
        for sheet_name in ["Resumen", "Incidentes"]:
            ws = writer.sheets[sheet_name]
            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="4472C4")
                cell.alignment = Alignment(horizontal="center")
            for i, col in enumerate(ws.columns, 1):
                max_len = max(len(str(c.value)) if c.value else 0 for c in col)
                ws.column_dimensions[get_column_letter(i)].width = min(max_len + 2, 60)

    buf.seek(0)
    return buf.read()