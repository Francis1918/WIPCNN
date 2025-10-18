#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script simple para consolidar todos los READMEs en uno solo
"""

from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).parent

# Archivos de documentación a consolidar
doc_files_to_merge = [
    ("readme.md", "README Principal"),
    ("README_DETALLADO.md", "Documentacion Detallada"),
    ("INSTALAR_CUDA_PYTORCH.md", "Instalacion CUDA y PyTorch"),
    ("INSTRUCCIONES_PYCHARM.md", "Configuracion PyCharm"),
    ("OPTIMIZACIONES_GPU_100.md", "Optimizaciones GPU"),
    ("README_OPTIMIZACIONES_96GB.md", "Optimizaciones para 96GB RAM"),
    ("SOLUCION_CUDA_ERROR_LARGO_PLAZO.md", "Soluciones a Errores CUDA"),
    ("TORNEO_MASIVO_README.md", "Torneos Masivos"),
    ("TORNEO_TODOS_CONTRA_TODOS_CUDA.md", "Torneos CUDA"),
    ("ANALISIS_TOURNAMENT_CUDA.md", "Analisis de Torneos"),
]

# Eliminar duplicados manteniendo el orden
seen = set()
doc_files_to_merge = [(f, t) for f, t in doc_files_to_merge if not (f in seen or seen.add(f))]

consolidated_content = []
consolidated_content.append("# Hierarchical-SAE - Documentacion Consolidada")
consolidated_content.append(f"\n**Generado automaticamente:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
consolidated_content.append("\n**Nota:** Esta documentacion consolida todos los archivos README del proyecto.")
consolidated_content.append("\n---\n")

print("Consolidando documentacion...")
print("=" * 60)

for filename, section_title in doc_files_to_merge:
    filepath = PROJECT_DIR / filename
    if filepath.exists():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            consolidated_content.append(f"\n# {section_title}")
            consolidated_content.append(f"\n*Fuente: {filename}*\n")
            consolidated_content.append(content)
            consolidated_content.append("\n---\n")
            
            print(f"[OK] Incorporado: {filename}")
        except Exception as e:
            print(f"[ERROR] Error leyendo {filename}: {e}")
    else:
        print(f"[SKIP] No existe: {filename}")

# Guardar documentación consolidada
consolidated_file = PROJECT_DIR / "README.md"
try:
    with open(consolidated_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(consolidated_content))
    print(f"\n[OK] Documentacion consolidada guardada en: {consolidated_file.name}")
    print(f"[OK] Total de secciones: {len([f for f, t in doc_files_to_merge if (PROJECT_DIR / f).exists()])}")
except Exception as e:
    print(f"[ERROR] Error guardando documentacion consolidada: {e}")

print("=" * 60)
print("Consolidacion completada!")