# -*- coding: utf-8 -*-
"""
Inicialización automática de dependencias para hierarchical-SAE
Este archivo se ejecuta automáticamente cuando se importa cualquier módulo del proyecto
"""

# Configurar dependencias automáticamente
try:
    from utils import setup_quartopy
    setup_quartopy.setup(silent=True)
    # No mostrar mensajes aquí para no alterar la salida original
except Exception:
    # Si falla, se manejará cuando se intente importar quartopy
    pass
