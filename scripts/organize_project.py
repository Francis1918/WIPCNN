#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para organizar archivos del proyecto en carpetas apropiadas.
Excluye trainRL.py y trainRL_resume.py de la reorganización.
"""

import os
import sys
import shutil
import platform
from pathlib import Path
from typing import Dict, List

# Configurar encoding para Windows
if platform.system() == 'Windows':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

class ProjectOrganizer:
    """Clase para organizar archivos del proyecto."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.moves = []
        
        # Definir estructura de carpetas
        self.folder_structure = {
            'scripts': {
                'description': 'Scripts de utilidad y herramientas',
                'files': [
                    'auto_checkpoint_monitor.py',
                    'check_cuda.py',
                    'cleanup_project.py',
                    'comprehensive_cleanup.py',
                    'consolidate_readme.py',
                    'diagnostico_proyecto.py',
                    'run_checkpoint_monitor.py',
                    'setup_dependencies.py',
                    'organize_project.py'
                ]
            },
            'tournaments': {
                'description': 'Scripts de torneos y competencias',
                'files': [
                    'compare_agents.py',
                    'torneomasivo.py',
                    'tournament_bracket.py',
                    'tournament_parallel_CUDA.py',
                    'tournament_parallel_massive.py',
                    'tournament_parallel.py',
                    'tournament.py'
                ]
            },
            'monitoring': {
                'description': 'Scripts de monitoreo',
                'files': [
                    'epoch_group_monitor.py'
                ]
            },
            'install': {
                'description': 'Sistema de instalación de dependencias',
                'files': [
                    'install_requirements.py',
                    'install.bat',
                    'install.sh',
                    'INSTALL_REQUIREMENTS_README.md',
                    'INSTALL_SYSTEM_README.md',
                    'QUICK_START.md'
                ]
            },
            'docs': {
                'description': 'Documentación del proyecto',
                'files': [
                    'README_CONSOLIDATED.md',
                    'README_DETALLADO.md',
                    'REPORTE_ANALISIS_ARCHIVOS_INNECESARIOS.md',
                    'DIAGNOSTICO_PROYECTO.json'
                ]
            }
        }
        
        # Archivos que NO se deben mover
        self.exclude_files = {
            'trainRL.py',
            'trainRL_resume.py',
            '__init__.py',
            '.gitignore',
            '.gitmodules',
            '.env',
            'LICENSE',
            'readme.md',
            'requirements.txt',
            '# Activar entorno Python automática.txt'
        }
    
    def print_plan(self):
        """Imprime el plan de reorganización."""
        is_windows = platform.system() == 'Windows'
        
        print("=" * 70)
        print("PLAN DE REORGANIZACION DEL PROYECTO")
        print("=" * 70)
        print()
        
        for folder, info in self.folder_structure.items():
            folder_icon = "[DIR]" if is_windows else "📁"
            print(f"{folder_icon} {folder}/ - {info['description']}")
            for file in info['files']:
                file_path = self.project_root / file
                check_icon = "[OK]" if is_windows else "✓"
                cross_icon = "[X]" if is_windows else "✗"
                if file_path.exists():
                    print(f"   {check_icon} {file}")
                else:
                    print(f"   {cross_icon} {file} (no existe)")
            print()
        
        pin_icon = "[PIN]" if is_windows else "📌"
        print(f"{pin_icon} Archivos que permaneceran en la raiz:")
        for file in sorted(self.exclude_files):
            file_path = self.project_root / file
            if file_path.exists():
                print(f"   * {file}")
        print()
        
        folder_icon = "[DIRS]" if is_windows else "📂"
        print(f"{folder_icon} Carpetas existentes que se mantendran:")
        existing_dirs = ['.idea', 'analysis', 'bot', 'chat', 'checkpoint_monitor', 
                        'models', 'QuartoRL', 'tests', 'tools', 'utils', 'venv']
        for dir_name in existing_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists() and dir_path.is_dir():
                print(f"   • {dir_name}/")
        print()
    
    def create_folders(self):
        """Crea las carpetas necesarias."""
        is_windows = platform.system() == 'Windows'
        check_icon = "[OK]" if is_windows else "✓"
        dot_icon = "*" if is_windows else "•"
        
        print("Creando carpetas...")
        for folder in self.folder_structure.keys():
            folder_path = self.project_root / folder
            if not folder_path.exists():
                folder_path.mkdir(parents=True, exist_ok=True)
                print(f"{check_icon} Creada: {folder}/")
            else:
                print(f"{dot_icon} Ya existe: {folder}/")
        print()
    
    def move_files(self, dry_run: bool = True):
        """Mueve los archivos a sus carpetas correspondientes."""
        is_windows = platform.system() == 'Windows'
        warn_icon = "[WARN]" if is_windows else "⚠️"
        file_icon = "[FILE]" if is_windows else "📄"
        check_icon = "[OK]" if is_windows else "✓"
        error_icon = "[ERROR]" if is_windows else "❌"
        
        if dry_run:
            print("MODO SIMULACION - No se moveran archivos realmente")
            print("-" * 70)
        else:
            print("MOVIENDO ARCHIVOS...")
            print("-" * 70)
        
        moved_count = 0
        skipped_count = 0
        
        for folder, info in self.folder_structure.items():
            folder_path = self.project_root / folder
            
            for file in info['files']:
                source = self.project_root / file
                destination = folder_path / file
                
                if not source.exists():
                    print(f"{warn_icon}  {file} no existe, omitiendo...")
                    skipped_count += 1
                    continue
                
                if source.is_dir():
                    print(f"{warn_icon}  {file} es un directorio, omitiendo...")
                    skipped_count += 1
                    continue
                
                if dry_run:
                    print(f"{file_icon} {file} -> {folder}/{file}")
                    self.moves.append((source, destination))
                else:
                    try:
                        shutil.move(str(source), str(destination))
                        print(f"{check_icon} Movido: {file} -> {folder}/{file}")
                        moved_count += 1
                    except Exception as e:
                        print(f"{error_icon} Error moviendo {file}: {e}")
                        skipped_count += 1
        
        print()
        if dry_run:
            print(f"Total de archivos a mover: {len(self.moves)}")
        else:
            print(f"Archivos movidos: {moved_count}")
            print(f"Archivos omitidos: {skipped_count}")
        print()
    
    def create_readme_files(self):
        """Crea archivos README en cada carpeta."""
        is_windows = platform.system() == 'Windows'
        check_icon = "[OK]" if is_windows else "✓"
        dot_icon = "*" if is_windows else "•"
        
        print("Creando archivos README en carpetas...")
        
        for folder, info in self.folder_structure.items():
            folder_path = self.project_root / folder
            readme_path = folder_path / "README.md"
            
            if not readme_path.exists():
                content = f"# {folder.capitalize()}\n\n{info['description']}\n\n## Archivos\n\n"
                for file in info['files']:
                    if (folder_path / file).exists():
                        content += f"- [`{file}`]({file})\n"
                
                readme_path.write_text(content, encoding='utf-8')
                print(f"{check_icon} Creado: {folder}/README.md")
            else:
                print(f"{dot_icon} Ya existe: {folder}/README.md")
        print()
    
    def update_main_readme(self):
        """Actualiza el README principal con la nueva estructura."""
        is_windows = platform.system() == 'Windows'
        note_icon = "[NOTE]" if is_windows else "📝"
        check_icon = "[OK]" if is_windows else "✓"
        
        readme_path = self.project_root / "readme.md"
        
        if readme_path.exists():
            print(f"{note_icon} Actualizando readme.md con nueva estructura...")
            
            structure_section = "\n## 📁 Estructura del Proyecto\n\n"
            structure_section += "```\n"
            structure_section += "hierarchical-SAE/\n"
            structure_section += "├── trainRL.py              # Script principal de entrenamiento\n"
            structure_section += "├── trainRL_resume.py       # Reanudar entrenamiento\n"
            structure_section += "├── requirements.txt        # Dependencias del proyecto\n"
            structure_section += "├── readme.md              # Este archivo\n"
            structure_section += "│\n"
            
            for folder, info in self.folder_structure.items():
                structure_section += f"├── {folder}/              # {info['description']}\n"
            
            structure_section += "│\n"
            structure_section += "├── bot/                   # Implementaciones de bots\n"
            structure_section += "├── models/                # Modelos de redes neuronales\n"
            structure_section += "├── QuartoRL/              # Lógica del juego Quarto\n"
            structure_section += "├── utils/                 # Utilidades generales\n"
            structure_section += "├── tests/                 # Tests y notebooks\n"
            structure_section += "├── tools/                 # Herramientas adicionales\n"
            structure_section += "├── analysis/              # Análisis de resultados\n"
            structure_section += "└── checkpoint_monitor/    # Monitoreo de checkpoints\n"
            structure_section += "```\n"
            
            print(f"{check_icon} Seccion de estructura preparada")
            print("  (Agregar manualmente al readme.md si es necesario)")
            print()
    
    def organize(self, dry_run: bool = True):
        """Ejecuta el proceso completo de organización."""
        print()
        self.print_plan()
        
        if dry_run:
            print("=" * 70)
            print("EJECUTANDO EN MODO SIMULACIÓN")
            print("=" * 70)
            print()
        
        self.create_folders()
        self.move_files(dry_run=dry_run)
        
        if not dry_run:
            self.create_readme_files()
            self.update_main_readme()
        
        print("=" * 70)
        if dry_run:
            print("SIMULACIÓN COMPLETADA")
            print("Para ejecutar los cambios reales, usa: --execute")
        else:
            print("REORGANIZACIÓN COMPLETADA")
        print("=" * 70)
        print()


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Organizar archivos del proyecto en carpetas apropiadas"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Ejecutar los cambios (por defecto solo simula)"
    )
    
    args = parser.parse_args()
    
    project_root = Path.cwd()
    organizer = ProjectOrganizer(project_root)
    
    organizer.organize(dry_run=not args.execute)


if __name__ == "__main__":
    main()