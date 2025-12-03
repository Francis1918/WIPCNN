# -*- coding: utf-8 -*-
"""
Bot package initialization with automatic dependency configuration
"""

# Configurar dependencias automáticamente antes de cualquier import
import sys
from pathlib import Path

# Agregar el directorio padre al path para poder importar utils
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Configurar dependencias silenciosamente
try:
    from utils import setup_quartopy
    setup_quartopy.setup(silent=True)
except Exception:
    pass
