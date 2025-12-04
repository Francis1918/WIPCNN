"""
Utilidades para el sistema de duelo.
"""
import os
from pathlib import Path
from datetime import datetime
import json

# Rutas fijas
RUTA_AGENTES = Path(r"C:\Users\bravo\Documents\Metodos Numericos Pycharm\Mech Interp\Comparacion entre agentes\Agentes")
RUTA_RESULTADOS = Path(r"C:\Users\bravo\Documents\Metodos Numericos Pycharm\Mech Interp\Comparacion entre agentes\Resultados")


def asegurar_directorios():
    """Crea los directorios si no existen."""
    RUTA_AGENTES.mkdir(parents=True, exist_ok=True)
    RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)


def listar_agentes():
    """Lista todos los agentes disponibles en la carpeta de agentes."""
    extensiones = ['.pt', '.pth']
    agentes = []

    if RUTA_AGENTES.exists():
        for archivo in RUTA_AGENTES.iterdir():
            if archivo.suffix.lower() in extensiones:
                agentes.append(archivo)

    return agentes


def generar_nombre_resultado():
    """Genera nombre único para archivo de resultados."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"duelo_{timestamp}.json"


def guardar_resultado(resultado: dict, nombre_archivo: str = None):
    """Guarda el resultado del duelo en JSON."""
    asegurar_directorios()

    if nombre_archivo is None:
        nombre_archivo = generar_nombre_resultado()

    ruta_completa = RUTA_RESULTADOS / nombre_archivo

    with open(ruta_completa, 'w', encoding='utf-8') as f:
        json.dump(resultado, f, indent=4, ensure_ascii=False, default=str)

    print(f"Resultado guardado en: {ruta_completa}")
    return ruta_completa