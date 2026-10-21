"""Modelos SQLAlchemy. Importar aqui cada modelo para que Alembic lo detecte."""
from app.database import Base  # noqa: F401
from app.models.auditoria import (  # noqa: F401
    Auditoria,
    LogAiAnalysis,
    Notificacion,
    SecurityLog,
)
from app.models.catalogos import (  # noqa: F401
    ConfiguracionSistema,
    Curso,
    MetricaProceso,
    ObjetivoEstrategico,
    Proceso,
    UnidadOrganizativa,
)
from app.models.importacion import RegistroImportado  # noqa: F401
from app.models.indicadores import (  # noqa: F401
    FuenteDatos,
    HistorialSnapshot,
    Indicador,
    MetaIndicador,
    NarrativaIA,
    ValorIndicador,
    VariableIndicador,
)
from app.models.usuarios import ROLES_VALIDOS, Usuario  # noqa: F401
from app.models.nbm import DatasetGenerado, Experimento  # noqa: F401
