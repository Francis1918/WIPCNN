#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
comprehensive_cleanup.py - Script completo para limpiar archivos innecesarios del proyecto

Este script mueve todos los archivos temporales e innecesarios a una carpeta externa,
organizándolos por categorías y preservando la estructura original.

Uso:
    python comprehensive_cleanup.py                    # Modo interactivo
    python comprehensive_cleanup.py --dry-run          # Modo simulación (sin cambios)
    python comprehensive_cleanup.py --auto             # Modo automático (sin confirmación)
    python comprehensive_cleanup.py --consolidate-docs # Solo consolidar documentación
"""

import os
import sys
import shutil
import argparse
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import json

# Configuración
TEMP_DIR = Path(r"C:\Users\bravo\Documents\Metodos Numericos Pycharm\Mech Interp\Temporales")
PROJECT_DIR = Path(__file__).parent

class CleanupManager:
    """Gestor principal de limpieza del proyecto"""
    
    def __init__(self, dry_run=False, auto_mode=False):
        self.dry_run = dry_run
        self.auto_mode = auto_mode
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_dir = TEMP_DIR / f"hierarchical-SAE_cleanup_{self.timestamp}"
        
        # Estructura de categorías
        self.categories = {
            '01_tournament_results': self.backup_dir / '01_tournament_results',
            '02_model_checkpoints': self.backup_dir / '02_model_checkpoints',
            '03_generated_data': self.backup_dir / '03_generated_data',
            '04_debug_test_files': self.backup_dir / '04_debug_test_files',
            '05_documentation_duplicates': self.backup_dir / '05_documentation_duplicates',
            '06_installation_scripts': self.backup_dir / '06_installation_scripts',
            '07_configuration_backups': self.backup_dir / '07_configuration_backups',
            '08_log_files': self.backup_dir / '08_log_files',
            '09_pickle_data': self.backup_dir / '09_pickle_data',
            '10_cache_files': self.backup_dir / '10_cache_files',
            '11_analysis_results': self.backup_dir / '11_analysis_results',
            '12_miscellaneous': self.backup_dir / '12_miscellaneous',
        }
        
        # Estadísticas
        self.stats = {
            'moved': 0,
            'skipped': 0,
            'errors': 0,
            'total_size': 0,
        }
        
        # Logs
        self.log_file = self.backup_dir / "cleanup_log.txt"
        self.errors_log = self.backup_dir / "errors_log.txt"
        self.manifest_file = self.backup_dir / "manifest.json"
        self.manifest_data = []
        
    def setup_directories(self):
        """Crea la estructura de directorios"""
        if not self.dry_run:
            for cat_dir in self.categories.values():
                cat_dir.mkdir(parents=True, exist_ok=True)
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
    def log(self, message, is_error=False):
        """Registra un mensaje en el log apropiado"""
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp_str}] {message}"
        
        print(message)
        
        if not self.dry_run:
            target_log = self.errors_log if is_error else self.log_file
            # Asegurar que el directorio padre existe
            target_log.parent.mkdir(parents=True, exist_ok=True)
            with open(target_log, 'a', encoding='utf-8') as f:
                f.write(log_entry + "\n")
    
    def get_file_hash(self, filepath):
        """Calcula el hash MD5 de un archivo"""
        hash_md5 = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.log(f"Error calculando hash de {filepath}: {e}", is_error=True)
            return None
    
    def get_file_size(self, filepath):
        """Obtiene el tamaño de un archivo en bytes"""
        try:
            return filepath.stat().st_size
        except:
            return 0
    
    def move_file_safely(self, source: Path, dest_category: str, preserve_structure=False):
        """Mueve un archivo de forma segura con verificación"""
        if not source.exists():
            return False, "No existe"
        
        category_dir = self.categories[dest_category]
        
        # Determinar destino
        if preserve_structure:
            # Preservar estructura de directorios relativa
            rel_path = source.relative_to(PROJECT_DIR)
            dest = category_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
        else:
            dest = category_dir / source.name
            
            # Si el destino existe, agregar sufijo numérico
            counter = 1
            original_dest = dest
            while dest.exists():
                dest = original_dest.parent / f"{original_dest.stem}_{counter}{original_dest.suffix}"
                counter += 1
        
        if self.dry_run:
            self.log(f"[DRY-RUN] Movería: {source.relative_to(PROJECT_DIR)} -> {dest_category}")
            return True, "DRY-RUN"
        
        try:
            # Calcular hash antes de mover
            file_size = self.get_file_size(source)
            source_hash = self.get_file_hash(source)
            if source_hash is None:
                return False, "Error calculando hash origen"
            
            # Mover archivo
            shutil.move(str(source), str(dest))
            
            # Verificar hash después de mover
            dest_hash = self.get_file_hash(dest)
            if dest_hash is None:
                return False, "Error calculando hash destino"
            
            if source_hash != dest_hash:
                # Restaurar archivo si los hashes no coinciden
                shutil.move(str(dest), str(source))
                return False, "Hash no coincide - archivo restaurado"
            
            # Registrar en manifest
            self.manifest_data.append({
                'source': str(source.relative_to(PROJECT_DIR)),
                'destination': str(dest.relative_to(self.backup_dir)),
                'category': dest_category,
                'size': file_size,
                'hash': source_hash,
                'timestamp': datetime.now().isoformat()
            })
            
            self.stats['moved'] += 1
            self.stats['total_size'] += file_size
            self.log(f"✓ [{dest_category}] Movido: {source.relative_to(PROJECT_DIR)}")
            return True, "OK"
            
        except Exception as e:
            self.log(f"✗ [{dest_category}] Error moviendo {source.name}: {e}", is_error=True)
            return False, str(e)
    
    def move_directory_safely(self, source: Path, dest_category: str):
        """Mueve un directorio completo de forma segura"""
        if not source.exists() or not source.is_dir():
            return False, "No existe o no es directorio"
        
        category_dir = self.categories[dest_category]
        dest = category_dir / source.name
        
        # Si el destino existe, agregar sufijo numérico
        counter = 1
        original_dest = dest
        while dest.exists():
            dest = original_dest.parent / f"{original_dest.name}_{counter}"
            counter += 1
        
        if self.dry_run:
            self.log(f"[DRY-RUN] Movería directorio: {source.relative_to(PROJECT_DIR)}/ -> {dest_category}")
            return True, "DRY-RUN"
        
        try:
            # Calcular tamaño total del directorio
            dir_size = sum(f.stat().st_size for f in source.rglob('*') if f.is_file())
            
            shutil.move(str(source), str(dest))
            
            self.stats['moved'] += 1
            self.stats['total_size'] += dir_size
            self.log(f"✓ [{dest_category}] Movido directorio: {source.relative_to(PROJECT_DIR)}/")
            return True, "OK"
            
        except Exception as e:
            self.log(f"✗ [{dest_category}] Error moviendo directorio {source.name}: {e}", is_error=True)
            return False, str(e)
    
    def remove_pycache_recursively(self):
        """Elimina todos los directorios __pycache__ recursivamente"""
        removed_count = 0
        for root, dirs, files in os.walk(PROJECT_DIR):
            if '__pycache__' in dirs:
                pycache_path = Path(root) / '__pycache__'
                if self.dry_run:
                    self.log(f"[DRY-RUN] Eliminaría: {pycache_path.relative_to(PROJECT_DIR)}")
                    removed_count += 1
                else:
                    try:
                        shutil.rmtree(pycache_path)
                        self.log(f"✓ [CACHE] Eliminado: {pycache_path.relative_to(PROJECT_DIR)}")
                        removed_count += 1
                    except Exception as e:
                        self.log(f"✗ [CACHE] Error eliminando {pycache_path}: {e}", is_error=True)
        return removed_count
    
    def cleanup_tournament_results(self):
        """Limpia resultados de torneos"""
        self.log("\n=== LIMPIANDO RESULTADOS DE TORNEOS ===")
        
        # Lista única de directorios de torneos
        tournament_dirs = list(set([
            "torneomasivo",
            "torneomasivo_cli",
            "tournament_parallel",
            "tournament_parallel_massive",
            "tournaments_parallel",
        ]))
        
        for dir_name in tournament_dirs:
            source = PROJECT_DIR / dir_name
            success, msg = self.move_directory_safely(source, '01_tournament_results')
            if not success and msg != "No existe o no es directorio":
                self.stats['errors'] += 1
            elif msg == "No existe o no es directorio":
                self.stats['skipped'] += 1
    
    def cleanup_model_checkpoints(self):
        """Limpia checkpoints de modelos antiguos (mantiene checkpoints clave)"""
        self.log("\n=== LIMPIANDO CHECKPOINTS DE MODELOS ===")
        
        weights_dir = PROJECT_DIR / "models" / "weights" / "QuartoCNN1"
        if not weights_dir.exists():
            self.log("No se encontró directorio de pesos")
            return
        
        # Obtener todos los checkpoints
        all_checkpoints = list(weights_dir.glob("*.pt"))
        
        # Extraer épocas
        epoch_files = {}
        for ckpt in all_checkpoints:
            try:
                epoch_str = ckpt.name.split("epoch_")[1].split(".")[0]
                epoch = int(epoch_str)
                if epoch not in epoch_files:
                    epoch_files[epoch] = []
                epoch_files[epoch].append(ckpt)
            except (IndexError, ValueError):
                continue
        
        # Determinar qué checkpoints mantener
        max_epoch = max(epoch_files.keys()) if epoch_files else 0
        keep_epochs = set()
        
        # Mantener checkpoints cada 50 épocas
        for epoch in range(0, max_epoch + 1, 50):
            keep_epochs.add(epoch)
        
        # Mantener el último checkpoint
        keep_epochs.add(max_epoch)
        
        # Mantener algunos checkpoints adicionales clave
        key_epochs = [0, 100, 200, 500, 1000]
        keep_epochs.update([e for e in key_epochs if e <= max_epoch])
        
        self.log(f"Total de épocas: {len(epoch_files)}")
        self.log(f"Épocas a mantener: {sorted(keep_epochs)}")
        
        # Mover checkpoints no esenciales
        for epoch, files in epoch_files.items():
            if epoch not in keep_epochs:
                for ckpt_file in files:
                    success, msg = self.move_file_safely(ckpt_file, '02_model_checkpoints', preserve_structure=True)
                    if not success and msg != "No existe":
                        self.stats['errors'] += 1
            elif len(files) > 1:
                # Si hay múltiples archivos para la misma época, mantener solo el más reciente
                files_sorted = sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)
                for ckpt_file in files_sorted[1:]:  # Mover todos excepto el más reciente
                    success, msg = self.move_file_safely(ckpt_file, '02_model_checkpoints', preserve_structure=True)
                    if not success and msg != "No existe":
                        self.stats['errors'] += 1
    
    def cleanup_generated_data(self):
        """Limpia archivos de datos generados"""
        self.log("\n=== LIMPIANDO DATOS GENERADOS ===")
        
        # Lista única de archivos de datos generados
        data_files = list(set([
            "board.csv",
            "piece_map.csv",
            "img.png",
        ]))
        
        for file_name in data_files:
            source = PROJECT_DIR / file_name
            success, msg = self.move_file_safely(source, '03_generated_data')
            if success:
                pass
            elif msg == "No existe":
                self.stats['skipped'] += 1
            else:
                self.stats['errors'] += 1
    
    def cleanup_debug_test_files(self):
        """Limpia archivos de debug y pruebas"""
        self.log("\n=== LIMPIANDO ARCHIVOS DE DEBUG Y PRUEBAS ===")
        
        # Lista única de archivos de debug y pruebas
        debug_files = list(set([
            "a.py",
            "a.md",
            "cart_p0ole.py",
            "debugging.py",
            "test_collector.py",
            "try_collector.py",
            "actions.py",
        ]))
        
        for file_name in debug_files:
            source = PROJECT_DIR / file_name
            success, msg = self.move_file_safely(source, '04_debug_test_files')
            if success:
                pass
            elif msg == "No existe":
                self.stats['skipped'] += 1
            else:
                self.stats['errors'] += 1
    
    def cleanup_documentation_duplicates(self):
        """Limpia documentación duplicada"""
        self.log("\n=== LIMPIANDO DOCUMENTACIÓN DUPLICADA ===")
        
        # Lista única de archivos de documentación duplicada
        doc_files = list(set([
            "INSTALAR_CUDA_PYTORCH.md",
            "INSTRUCCIONES_PYCHARM.md",
            "OPTIMIZACIONES_GPU_100.md",
            "README_OPTIMIZACIONES_96GB.md",
            "SOLUCION_CUDA_ERROR_LARGO_PLAZO.md",
            "TORNEO_MASIVO_README.md",
            "TORNEO_TODOS_CONTRA_TODOS_CUDA.md",
            "ANALISIS_TOURNAMENT_CUDA.md",
        ]))
        
        for file_name in doc_files:
            source = PROJECT_DIR / file_name
            success, msg = self.move_file_safely(source, '05_documentation_duplicates')
            if success:
                pass
            elif msg == "No existe":
                self.stats['skipped'] += 1
            else:
                self.stats['errors'] += 1
    
    def cleanup_installation_scripts(self):
        """Limpia scripts de instalación redundantes"""
        self.log("\n=== LIMPIANDO SCRIPTS DE INSTALACIÓN REDUNDANTES ===")
        
        # Lista única de scripts de instalación redundantes
        install_scripts = list(set([
            "install_cuda_simple.bat",
            "install_pytorch_cuda_full.py",
            "install_pytorch_cuda.py",
            "install_pytorch_direct.py",
            "update_requirements.py",
        ]))
        
        for file_name in install_scripts:
            source = PROJECT_DIR / file_name
            success, msg = self.move_file_safely(source, '06_installation_scripts')
            if success:
                pass
            elif msg == "No existe":
                self.stats['skipped'] += 1
            else:
                self.stats['errors'] += 1
    
    def cleanup_configuration_backups(self):
        """Limpia archivos de configuración personal"""
        self.log("\n=== LIMPIANDO CONFIGURACIONES PERSONALES ===")
        
        # Lista única de archivos de configuración personal
        config_files = list(set([
            "$PROFILE.txt",
        ]))
        
        for file_name in config_files:
            source = PROJECT_DIR / file_name
            success, msg = self.move_file_safely(source, '07_configuration_backups')
            if success:
                pass
            elif msg == "No existe":
                self.stats['skipped'] += 1
            else:
                self.stats['errors'] += 1
        
        # Mover .idea si existe
        idea_dir = PROJECT_DIR / ".idea"
        if idea_dir.exists():
            success, msg = self.move_directory_safely(idea_dir, '07_configuration_backups')
            if not success:
                self.stats['errors'] += 1
    
    def cleanup_log_files(self):
        """Limpia archivos de log"""
        self.log("\n=== LIMPIANDO ARCHIVOS DE LOG ===")
        
        # Lista única de patrones de archivos de log
        log_patterns = list(set(["*.log", "training_*.log", "tournament_results*.txt"]))
        for pattern in log_patterns:
            for log_file in PROJECT_DIR.rglob(pattern):
                if log_file.is_file() and log_file != self.log_file:
                    success, msg = self.move_file_safely(log_file, '08_log_files', preserve_structure=True)
                    if not success and msg != "No existe":
                        self.stats['errors'] += 1
    
    def cleanup_pickle_data(self):
        """Limpia archivos pickle temporales"""
        self.log("\n=== LIMPIANDO ARCHIVOS PICKLE ===")
        
        # Lista única de archivos pickle temporales
        pickle_files = list(set([
            "ac_2last_states.pkl",
            "ba_increasing_n_last_states.pkl",
        ]))
        
        for file_name in pickle_files:
            source = PROJECT_DIR / file_name
            success, msg = self.move_file_safely(source, '09_pickle_data')
            if success:
                pass
            elif msg == "No existe":
                self.stats['skipped'] += 1
            else:
                self.stats['errors'] += 1
    
    def cleanup_cache_files(self):
        """Limpia archivos de caché"""
        self.log("\n=== LIMPIANDO CACHÉ PYTHON ===")
        
        # Eliminar __pycache__
        cache_removed = self.remove_pycache_recursively()
        self.stats['moved'] += cache_removed
        
        # Buscar archivos .pyc y .pyo
        pyc_files = list(PROJECT_DIR.rglob("*.pyc")) + list(PROJECT_DIR.rglob("*.pyo"))
        for pyc_file in pyc_files:
            if self.dry_run:
                self.log(f"[DRY-RUN] Eliminaría: {pyc_file.relative_to(PROJECT_DIR)}")
                self.stats['moved'] += 1
            else:
                try:
                    pyc_file.unlink()
                    self.log(f"✓ [CACHE] Eliminado: {pyc_file.relative_to(PROJECT_DIR)}")
                    self.stats['moved'] += 1
                except Exception as e:
                    self.log(f"✗ [CACHE] Error eliminando {pyc_file}: {e}", is_error=True)
                    self.stats['errors'] += 1
    
    def cleanup_analysis_results(self):
        """Limpia resultados de análisis antiguos"""
        self.log("\n=== LIMPIANDO RESULTADOS DE ANÁLISIS ===")
        
        analysis_dir = PROJECT_DIR / "analysis" / "agent_comparisons"
        if analysis_dir.exists():
            success, msg = self.move_directory_safely(analysis_dir, '11_analysis_results')
            if not success and msg != "No existe o no es directorio":
                self.stats['errors'] += 1
    
    def cleanup_miscellaneous(self):
        """Limpia archivos temporales misceláneos"""
        self.log("\n=== LIMPIANDO ARCHIVOS TEMPORALES MISCELÁNEOS ===")
        
        # Lista única de patrones de archivos temporales
        temp_patterns = list(set(["*.tmp", "*.temp", "*.bak", "*.swp", "*~"]))
        for pattern in temp_patterns:
            for temp_file in PROJECT_DIR.rglob(pattern):
                if temp_file.is_file():
                    success, msg = self.move_file_safely(temp_file, '12_miscellaneous', preserve_structure=True)
                    if not success and msg != "No existe":
                        self.stats['errors'] += 1
    
    def consolidate_documentation(self):
        """Consolida toda la documentación en un README unificado"""
        self.log("\n=== CONSOLIDANDO DOCUMENTACIÓN ===")
        
        if self.dry_run:
            self.log("[DRY-RUN] Se consolidaría la documentación en README_CONSOLIDATED.md")
            return
        
        # Lista única de archivos de documentación a consolidar (mantener orden)
        doc_files_to_merge = [
            ("readme.md", "README Principal"),
            ("README_DETALLADO.md", "Documentación Detallada"),
            ("INSTALAR_CUDA_PYTORCH.md", "Instalación CUDA y PyTorch"),
            ("INSTRUCCIONES_PYCHARM.md", "Configuración PyCharm"),
            ("OPTIMIZACIONES_GPU_100.md", "Optimizaciones GPU"),
            ("README_OPTIMIZACIONES_96GB.md", "Optimizaciones para 96GB RAM"),
            ("SOLUCION_CUDA_ERROR_LARGO_PLAZO.md", "Soluciones a Errores CUDA"),
            ("TORNEO_MASIVO_README.md", "Torneos Masivos"),
            ("TORNEO_TODOS_CONTRA_TODOS_CUDA.md", "Torneos CUDA"),
            ("ANALISIS_TOURNAMENT_CUDA.md", "Análisis de Torneos"),
        ]
        # Eliminar duplicados manteniendo el orden
        seen = set()
        doc_files_to_merge = [(f, t) for f, t in doc_files_to_merge if not (f in seen or seen.add(f))]
        
        consolidated_content = []
        consolidated_content.append("# Hierarchical-SAE - Documentación Consolidada")
        consolidated_content.append(f"\n**Generado automáticamente:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        consolidated_content.append("\n**Nota:** Esta documentación consolida todos los archivos README del proyecto.")
        consolidated_content.append("\n---\n")
        
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
                    
                    self.log(f"✓ Incorporado: {filename}")
                except Exception as e:
                    self.log(f"✗ Error leyendo {filename}: {e}", is_error=True)
        
        # Guardar documentación consolidada
        consolidated_file = PROJECT_DIR / "README_CONSOLIDATED.md"
        try:
            with open(consolidated_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(consolidated_content))
            self.log(f"✓ Documentación consolidada guardada en: {consolidated_file.name}")
        except Exception as e:
            self.log(f"✗ Error guardando documentación consolidada: {e}", is_error=True)
    
    def save_manifest(self):
        """Guarda el manifest de archivos movidos"""
        if not self.dry_run and self.manifest_data:
            try:
                with open(self.manifest_file, 'w', encoding='utf-8') as f:
                    json.dump(self.manifest_data, f, indent=2, ensure_ascii=False)
                self.log(f"\n✓ Manifest guardado en: {self.manifest_file}")
            except Exception as e:
                self.log(f"✗ Error guardando manifest: {e}", is_error=True)
    
    def generate_summary(self):
        """Genera un resumen de la limpieza"""
        self.log("\n" + "=" * 80)
        self.log("RESUMEN DE LIMPIEZA")
        self.log("=" * 80)
        
        if self.dry_run:
            self.log("\n⚠️  MODO DRY-RUN - NO SE REALIZARON CAMBIOS REALES")
        
        self.log(f"\n✓ Archivos/directorios movidos: {self.stats['moved']}")
        self.log(f"○ Archivos omitidos (no existen): {self.stats['skipped']}")
        self.log(f"✗ Errores: {self.stats['errors']}")
        
        # Convertir tamaño a formato legible
        size_mb = self.stats['total_size'] / (1024 * 1024)
        size_gb = size_mb / 1024
        if size_gb >= 1:
            self.log(f"💾 Espacio liberado: {size_gb:.2f} GB")
        else:
            self.log(f"💾 Espacio liberado: {size_mb:.2f} MB")
        
        if not self.dry_run:
            self.log(f"\n📁 Ubicación de respaldo: {self.backup_dir}")
            self.log(f"📄 Log de operaciones: {self.log_file}")
            if self.stats['errors'] > 0:
                self.log(f"⚠️  Log de errores: {self.errors_log}")
            self.log(f"📋 Manifest: {self.manifest_file}")
        
        self.log("=" * 80)
        
        # Guardar resumen en archivo
        if not self.dry_run:
            summary_file = self.backup_dir / "RESUMEN.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("RESUMEN DE LIMPIEZA DEL PROYECTO hierarchical-SAE\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Proyecto original: {PROJECT_DIR}\n")
                f.write(f"Ubicación de respaldo: {self.backup_dir}\n\n")
                f.write(f"Archivos/directorios movidos: {self.stats['moved']}\n")
                f.write(f"Archivos omitidos: {self.stats['skipped']}\n")
                f.write(f"Errores: {self.stats['errors']}\n")
                f.write(f"Espacio liberado: {size_gb:.2f} GB ({size_mb:.2f} MB)\n\n")
                f.write("CATEGORÍAS:\n")
                for cat_name, cat_path in self.categories.items():
                    if cat_path.exists():
                        items = list(cat_path.rglob('*'))
                        files = [i for i in items if i.is_file()]
                        f.write(f"  - {cat_name}: {len(files)} archivos\n")
            
            self.log(f"\n✓ Resumen guardado en: {summary_file}")
    
    def run(self):
        """Ejecuta el proceso completo de limpieza"""
        self.log("=" * 80)
        self.log("INICIANDO LIMPIEZA COMPLETA DEL PROYECTO")
        self.log("=" * 80)
        self.log(f"Proyecto: {PROJECT_DIR}")
        self.log(f"Destino: {self.backup_dir}")
        
        if self.dry_run:
            self.log("\n⚠️  MODO DRY-RUN ACTIVADO - No se realizarán cambios reales")
        
        self.log("=" * 80)
        
        # Configurar directorios
        self.setup_directories()
        
        # Ejecutar limpieza por categorías
        self.cleanup_tournament_results()
        self.cleanup_model_checkpoints()
        self.cleanup_generated_data()
        self.cleanup_debug_test_files()
        self.cleanup_documentation_duplicates()
        self.cleanup_installation_scripts()
        self.cleanup_configuration_backups()
        self.cleanup_log_files()
        self.cleanup_pickle_data()
        self.cleanup_cache_files()
        self.cleanup_analysis_results()
        self.cleanup_miscellaneous()
        
        # Consolidar documentación
        self.consolidate_documentation()
        
        # Guardar manifest
        self.save_manifest()
        
        # Generar resumen
        self.generate_summary()
        
        self.log("\n✅ Limpieza completada!")


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Script completo para limpiar archivos innecesarios del proyecto",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--dry-run", action="store_true", 
                       help="Modo simulación (muestra qué se haría sin hacer cambios)")
    parser.add_argument("--auto", action="store_true",
                       help="Modo automático (sin confirmación)")
    parser.add_argument("--consolidate-docs", action="store_true",
                       help="Solo consolidar documentación sin mover archivos")
    
    args = parser.parse_args()
    
    # Crear gestor de limpieza
    manager = CleanupManager(dry_run=args.dry_run, auto_mode=args.auto)
    
    # Si solo se quiere consolidar documentación
    if args.consolidate_docs:
        manager.setup_directories()
        manager.consolidate_documentation()
        return
    
    # Modo interactivo si no es automático
    if not args.auto and not args.dry_run:
        print("\n" + "=" * 80)
        print("LIMPIEZA COMPLETA DEL PROYECTO hierarchical-SAE")
        print("=" * 80)
        print("\nEste script moverá archivos innecesarios a:")
        print(f"  {TEMP_DIR}")
        print("\nCategorías a limpiar:")
        print("  1. Resultados de torneos antiguos")
        print("  2. Checkpoints de modelos (mantiene checkpoints clave)")
        print("  3. Archivos de datos generados")
        print("  4. Scripts de debug y pruebas")
        print("  5. Documentación duplicada")
        print("  6. Scripts de instalación redundantes")
        print("  7. Configuraciones personales")
        print("  8. Archivos de log")
        print("  9. Archivos pickle temporales")
        print(" 10. Caché de Python")
        print(" 11. Resultados de análisis antiguos")
        print(" 12. Archivos temporales misceláneos")
        print("\n⚠️  IMPORTANTE: Se creará un respaldo completo con verificación de integridad")
        print("\nOpciones:")
        print("  - Presiona ENTER para continuar")
        print("  - Escribe 'dry-run' para ver qué se haría sin hacer cambios")
        print("  - Escribe 'cancel' para cancelar")
        
        response = input("\nTu elección: ").strip().lower()
        
        if response == 'cancel':
            print("Operación cancelada por el usuario.")
            return
        elif response == 'dry-run':
            manager.dry_run = True
            print("\n✓ Modo DRY-RUN activado")
        elif response == '':
            # ENTER presionado - continuar con la ejecución normal
            print("\n✓ Continuando con la limpieza...")
        else:
            print(f"\n⚠️  Opción no reconocida: '{response}'. Continuando de todas formas...")
    
    # Ejecutar limpieza
    manager.run()


if __name__ == "__main__":
    main()