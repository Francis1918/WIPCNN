#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de diagnóstico completo del proyecto hierarchical-SAE
Busca errores, problemas y áreas de mejora
"""

import os
import sys
from pathlib import Path
import ast
import json

PROJECT_DIR = Path(__file__).parent

class ProjectDiagnostic:
    def __init__(self):
        self.issues = {
            'critical': [],
            'warning': [],
            'info': []
        }
        
    def add_issue(self, level, category, file, message):
        """Agrega un problema encontrado"""
        self.issues[level].append({
            'category': category,
            'file': str(file),
            'message': message
        })
    
    def check_python_syntax(self):
        """Verifica sintaxis de archivos Python"""
        print("\n[1/8] Verificando sintaxis de archivos Python...")
        
        py_files = list(PROJECT_DIR.rglob("*.py"))
        checked = 0
        errors = 0
        
        for py_file in py_files:
            # Ignorar archivos en directorios especiales
            if any(part in py_file.parts for part in ['__pycache__', '.venv', 'venv', '.git']):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    code = f.read()
                ast.parse(code)
                checked += 1
            except SyntaxError as e:
                self.add_issue('critical', 'Syntax Error', py_file.relative_to(PROJECT_DIR),
                             f"Error de sintaxis en linea {e.lineno}: {e.msg}")
                errors += 1
            except Exception as e:
                self.add_issue('warning', 'Parse Error', py_file.relative_to(PROJECT_DIR),
                             f"No se pudo analizar: {str(e)}")
        
        print(f"   Archivos verificados: {checked}")
        print(f"   Errores encontrados: {errors}")
    
    def check_imports(self):
        """Verifica imports problemáticos"""
        print("\n[2/8] Verificando imports...")
        
        py_files = list(PROJECT_DIR.rglob("*.py"))
        checked = 0
        
        for py_file in py_files:
            if any(part in py_file.parts for part in ['__pycache__', '.venv', 'venv', '.git']):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            # Verificar imports comunes problemáticos
                            if alias.name in ['quartopy'] and not (PROJECT_DIR.parent / 'quartopy').exists():
                                self.add_issue('warning', 'Missing Dependency', 
                                             py_file.relative_to(PROJECT_DIR),
                                             f"Import '{alias.name}' podria no estar disponible")
                    
                    elif isinstance(node, ast.ImportFrom):
                        if node.module and node.module.startswith('.'):
                            # Import relativo - verificar que el módulo existe
                            pass  # Simplificado por ahora
                
                checked += 1
            except Exception as e:
                pass  # Ya capturado en check_syntax
        
        print(f"   Archivos verificados: {checked}")
    
    def check_file_structure(self):
        """Verifica estructura de archivos"""
        print("\n[3/8] Verificando estructura de archivos...")
        
        # Verificar archivos esenciales
        essential_files = [
            'requirements.txt',
            'README.md',
            'LICENSE',
        ]
        
        for file in essential_files:
            if not (PROJECT_DIR / file).exists():
                self.add_issue('warning', 'Missing File', file,
                             f"Archivo esencial no encontrado")
        
        # Verificar directorios esenciales
        essential_dirs = [
            'bot',
            'models',
            'QuartoRL',
            'utils',
        ]
        
        for dir_name in essential_dirs:
            if not (PROJECT_DIR / dir_name).exists():
                self.add_issue('critical', 'Missing Directory', dir_name,
                             f"Directorio esencial no encontrado")
            elif not (PROJECT_DIR / dir_name / '__init__.py').exists():
                self.add_issue('info', 'Missing __init__.py', 
                             f"{dir_name}/__init__.py",
                             "Falta archivo __init__.py (puede ser intencional)")
        
        print(f"   Estructura verificada")
    
    def check_duplicates(self):
        """Busca archivos duplicados o redundantes"""
        print("\n[4/8] Buscando archivos duplicados...")
        
        # Buscar múltiples READMEs
        readme_files = list(PROJECT_DIR.glob("README*.md")) + list(PROJECT_DIR.glob("readme*.md"))
        if len(readme_files) > 1:
            self.add_issue('info', 'Duplicate Files', 'Multiple READMEs',
                         f"Se encontraron {len(readme_files)} archivos README: {[f.name for f in readme_files]}")
        
        # Buscar scripts de instalación duplicados
        install_scripts = list(PROJECT_DIR.glob("install*.py")) + list(PROJECT_DIR.glob("install*.bat"))
        if len(install_scripts) > 2:
            self.add_issue('info', 'Duplicate Files', 'Multiple Install Scripts',
                         f"Se encontraron {len(install_scripts)} scripts de instalacion")
        
        print(f"   Verificacion completada")
    
    def check_requirements(self):
        """Verifica requirements.txt"""
        print("\n[5/8] Verificando requirements.txt...")
        
        req_file = PROJECT_DIR / 'requirements.txt'
        if not req_file.exists():
            self.add_issue('critical', 'Missing File', 'requirements.txt',
                         "Archivo requirements.txt no encontrado")
            return
        
        try:
            with open(req_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            packages = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
            
            # Buscar duplicados
            seen = set()
            for pkg in packages:
                pkg_name = pkg.split('==')[0].split('>=')[0].split('<=')[0].strip()
                if pkg_name in seen:
                    self.add_issue('warning', 'Duplicate Dependency', 'requirements.txt',
                                 f"Paquete duplicado: {pkg_name}")
                seen.add(pkg_name)
            
            print(f"   Paquetes encontrados: {len(packages)}")
            print(f"   Duplicados: {len(packages) - len(seen)}")
            
        except Exception as e:
            self.add_issue('warning', 'Parse Error', 'requirements.txt',
                         f"Error leyendo archivo: {str(e)}")
    
    def check_large_files(self):
        """Busca archivos muy grandes"""
        print("\n[6/8] Buscando archivos grandes...")
        
        large_files = []
        total_size = 0
        
        for file in PROJECT_DIR.rglob("*"):
            if file.is_file():
                # Ignorar directorios especiales
                if any(part in file.parts for part in ['__pycache__', '.venv', 'venv', '.git', 'node_modules']):
                    continue
                
                try:
                    size = file.stat().st_size
                    total_size += size
                    
                    # Archivos mayores a 100MB
                    if size > 100 * 1024 * 1024:
                        size_mb = size / (1024 * 1024)
                        self.add_issue('warning', 'Large File', file.relative_to(PROJECT_DIR),
                                     f"Archivo muy grande: {size_mb:.2f} MB")
                        large_files.append((file, size))
                    
                    # Archivos .pt mayores a 50MB
                    elif file.suffix == '.pt' and size > 50 * 1024 * 1024:
                        size_mb = size / (1024 * 1024)
                        self.add_issue('info', 'Large Checkpoint', file.relative_to(PROJECT_DIR),
                                     f"Checkpoint grande: {size_mb:.2f} MB")
                
                except Exception:
                    pass
        
        total_gb = total_size / (1024 * 1024 * 1024)
        print(f"   Tamano total del proyecto: {total_gb:.2f} GB")
        print(f"   Archivos grandes encontrados: {len(large_files)}")
    
    def check_empty_files(self):
        """Busca archivos vacíos"""
        print("\n[7/8] Buscando archivos vacios...")
        
        empty_count = 0
        
        for file in PROJECT_DIR.rglob("*.py"):
            if any(part in file.parts for part in ['__pycache__', '.venv', 'venv', '.git']):
                continue
            
            try:
                if file.stat().st_size == 0:
                    self.add_issue('warning', 'Empty File', file.relative_to(PROJECT_DIR),
                                 "Archivo Python vacio")
                    empty_count += 1
                elif file.stat().st_size < 50:  # Muy pequeño
                    with open(file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    if not content or content == '':
                        self.add_issue('warning', 'Empty File', file.relative_to(PROJECT_DIR),
                                     "Archivo Python practicamente vacio")
                        empty_count += 1
            except Exception:
                pass
        
        print(f"   Archivos vacios encontrados: {empty_count}")
    
    def check_code_quality(self):
        """Verifica calidad básica del código"""
        print("\n[8/8] Verificando calidad de codigo...")
        
        issues_found = 0
        
        for py_file in PROJECT_DIR.rglob("*.py"):
            if any(part in py_file.parts for part in ['__pycache__', '.venv', 'venv', '.git']):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Buscar prints de debug
                if 'print(' in content and 'debug' in content.lower():
                    self.add_issue('info', 'Debug Code', py_file.relative_to(PROJECT_DIR),
                                 "Posible codigo de debug (prints)")
                    issues_found += 1
                
                # Buscar TODOs
                if 'TODO' in content or 'FIXME' in content:
                    self.add_issue('info', 'TODO/FIXME', py_file.relative_to(PROJECT_DIR),
                                 "Contiene comentarios TODO o FIXME")
                    issues_found += 1
                
            except Exception:
                pass
        
        print(f"   Problemas de calidad encontrados: {issues_found}")
    
    def generate_report(self):
        """Genera reporte final"""
        print("\n" + "=" * 70)
        print("REPORTE DE DIAGNOSTICO")
        print("=" * 70)
        
        total_issues = sum(len(issues) for issues in self.issues.values())
        
        print(f"\nTotal de problemas encontrados: {total_issues}")
        print(f"  - Criticos: {len(self.issues['critical'])}")
        print(f"  - Advertencias: {len(self.issues['warning'])}")
        print(f"  - Informativos: {len(self.issues['info'])}")
        
        # Mostrar problemas críticos
        if self.issues['critical']:
            print("\n" + "=" * 70)
            print("PROBLEMAS CRITICOS")
            print("=" * 70)
            for issue in self.issues['critical']:
                print(f"\n[{issue['category']}] {issue['file']}")
                print(f"  {issue['message']}")
        
        # Mostrar advertencias
        if self.issues['warning']:
            print("\n" + "=" * 70)
            print("ADVERTENCIAS")
            print("=" * 70)
            for issue in self.issues['warning'][:10]:  # Mostrar solo las primeras 10
                print(f"\n[{issue['category']}] {issue['file']}")
                print(f"  {issue['message']}")
            if len(self.issues['warning']) > 10:
                print(f"\n... y {len(self.issues['warning']) - 10} advertencias mas")
        
        # Mostrar información
        if self.issues['info']:
            print("\n" + "=" * 70)
            print("INFORMACION")
            print("=" * 70)
            for issue in self.issues['info'][:5]:  # Mostrar solo las primeras 5
                print(f"\n[{issue['category']}] {issue['file']}")
                print(f"  {issue['message']}")
            if len(self.issues['info']) > 5:
                print(f"\n... y {len(self.issues['info']) - 5} items informativos mas")
        
        # Guardar reporte completo
        report_file = PROJECT_DIR / "DIAGNOSTICO_PROYECTO.json"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(self.issues, f, indent=2, ensure_ascii=False)
            print(f"\n[OK] Reporte completo guardado en: {report_file.name}")
        except Exception as e:
            print(f"\n[ERROR] No se pudo guardar el reporte: {e}")
        
        print("\n" + "=" * 70)
        print("DIAGNOSTICO COMPLETADO")
        print("=" * 70)
    
    def run(self):
        """Ejecuta diagnóstico completo"""
        print("=" * 70)
        print("DIAGNOSTICO COMPLETO DEL PROYECTO")
        print("=" * 70)
        print(f"Proyecto: {PROJECT_DIR}")
        print("=" * 70)
        
        self.check_python_syntax()
        self.check_imports()
        self.check_file_structure()
        self.check_duplicates()
        self.check_requirements()
        self.check_large_files()
        self.check_empty_files()
        self.check_code_quality()
        
        self.generate_report()


if __name__ == "__main__":
    diagnostic = ProjectDiagnostic()
    diagnostic.run()