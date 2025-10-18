#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuración automática de dependencias externas para el proyecto hierarchical-SAE
"""

import os
import sys
from pathlib import Path

def setup_quartopy(silent=True):
    """Configura automáticamente la ruta del proyecto quartopy"""
    
    # Cargar variables de entorno desde .env si existe
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
            if not silent:
                print(f"✅ Cargado archivo .env desde: {env_file}")
        except ImportError:
            # Si no tiene python-dotenv, instalar automáticamente
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            from dotenv import load_dotenv
            load_dotenv(env_file)
            if not silent:
                print(f"✅ Instalado python-dotenv y cargado .env desde: {env_file}")
    else:
        if not silent:
            print(f"⚠️ No se encontró el archivo .env en: {env_file}")
    
    # Obtener la ruta principal de quartopy desde las variables de entorno
    quartopy_path = os.getenv("QUARTOPY_PATH")
    
    # Obtener rutas alternativas desde las variables de entorno
    fallback_paths_env = os.getenv("QUARTOPY_FALLBACK_PATHS", "")
    env_fallback_paths = fallback_paths_env.split(",") if fallback_paths_env else []
    
    # Lista de rutas posibles donde buscar quartopy
    fallback_paths = [
        quartopy_path,
        *env_fallback_paths,
        "../quartopy",
        "../Quartopy", 
        "../../quartopy",
        "../../Quartopy",
        os.path.expanduser("~/Documents/GitHub/Quartopy"),
        os.path.expanduser("~/Documents/quartopy"),
        "C:/quartopy",
        "C:/Quartopy",
        "C:/Users/bravo/Documents/quartopy",
        "C:/Users/bravo/Documents/Quartopy",
        "C:/Users/bravo/Documents/GitHub/quartopy",
        "C:/Users/bravo/Documents/GitHub/Quartopy",
        "C:/Users/bravo/Documents/Metodos Numericos Pycharm/quartopy",
        "C:/Users/bravo/Documents/Metodos Numericos Pycharm/Quartopy",
        "C:/Users/bravo/Documents/Metodos Numericos Pycharm/Mech Interp/Quartopy"
    ]
    
    # Buscar una ruta válida
    for path_str in fallback_paths:
        if not path_str:
            continue
            
        path = Path(path_str).resolve()
        
        # Verificar si la ruta existe y contiene quartopy
        if path.exists():
            # Buscar indicadores de que es el proyecto quartopy correcto
            indicators = [
                path / "__init__.py",
                path / "quartopy" / "__init__.py", 
                path / "setup.py",
                path / "pyproject.toml"
            ]
            
            if any(indicator.exists() for indicator in indicators):
                # Agregar al sys.path si no está presente
                path_str = str(path)
                if path_str not in sys.path:
                    sys.path.insert(0, path_str)
                    if not silent:
                        print(f"✅ Configurado quartopy desde: {path_str}")
                    return True
    
    # Si no encontró quartopy, mostrar mensaje de ayuda solo si no es silencioso
    if not silent:
        print("❌ No se pudo encontrar el proyecto quartopy.")
        print("\n📋 Para resolver este problema:")
        print("1. Clona o descarga el proyecto quartopy")
        print("2. Colócalo en alguna de estas ubicaciones:")
        for path in fallback_paths[1:6]:  # Mostrar solo las rutas más comunes
            if path:
                print(f"   - {path}")
        print("3. O edita el archivo .env y establece QUARTOPY_PATH con la ruta correcta")
        print("\n💡 Ejemplo: QUARTOPY_PATH=C:/ruta/a/tu/proyecto/quartopy")

    return False

# Configurar automáticamente al importar este módulo de forma silenciosa
if __name__ != "__main__":
    try:
        setup_quartopy(silent=True)
    except Exception as e:
        # Si falla silenciosamente, no hacer nada pero podemos loggear el error
        import traceback
        print(f"Error al configurar quartopy: {e}")
        print(traceback.format_exc())
        pass