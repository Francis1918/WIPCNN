# -*- coding: utf-8 -*-
"""
Inicialización automática de dependencias para hierarchical-SAE
Este archivo se ejecuta automáticamente cuando se importa cualquier módulo del proyecto
"""

# Configurar dependencias automáticamente
try:
    import setup_dependencies
    # No mostrar mensajes aquí para no alterar la salida original
except Exception:
    # Si falla, se manejará cuando se intente importar quartopy
    pass
