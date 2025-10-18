"""
Script para mover archivos temporales y generados del proyecto
Mueve TODOS los archivos temporales a: C:/Users/bravo/Documents/Metodos Numericos Pycharm/Mech Interp/Temporales
Organiza por tipo y verifica cada operación
"""
import os
import shutil
from pathlib import Path
from datetime import datetime
import hashlib

# Directorio de destino para archivos temporales
TEMP_DIR = Path(r"C:\Users\bravo\Documents\Metodos Numericos Pycharm\Mech Interp\Temporales")
PROJECT_DIR = Path(__file__).parent

# Crear subdirectorio con timestamp para esta limpieza
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_dir = TEMP_DIR / f"hierarchical-SAE_temporales_{timestamp}"

# Crear estructura de carpetas por tipo
categories = {
    'pkl_files': backup_dir / '01_pickle_files',
    'csv_files': backup_dir / '02_csv_files',
    'images': backup_dir / '03_images',
    'logs': backup_dir / '04_logs',
    'cache': backup_dir / '05_cache',
    'tournament_results': backup_dir / '06_tournament_results',
    'tournament_dirs': backup_dir / '07_tournament_directories',
    'analysis': backup_dir / '08_analysis_results',
    'checkpoints': backup_dir / '09_old_checkpoints',
    'other': backup_dir / '10_other_temp_files',
}

# Crear todas las carpetas
for cat_dir in categories.values():
    cat_dir.mkdir(parents=True, exist_ok=True)

# Log de operaciones
log_file = backup_dir / "cleanup_operations.log"
errors_log = backup_dir / "errors.log"

def log_message(message, is_error=False):
    """Registra un mensaje en el log apropiado"""
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp_str}] {message}\n"
    
    print(message)
    
    target_log = errors_log if is_error else log_file
    with open(target_log, 'a', encoding='utf-8') as f:
        f.write(log_entry)

def get_file_hash(filepath):
    """Calcula el hash MD5 de un archivo para verificación"""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        log_message(f"Error calculando hash de {filepath}: {e}", is_error=True)
        return None

def move_file_safely(source, dest_dir, category_name):
    """Mueve un archivo de forma segura con verificación"""
    if not source.exists():
        return False, "No existe"
    
    try:
        # Calcular hash antes de mover
        source_hash = get_file_hash(source)
        if source_hash is None:
            return False, "Error calculando hash origen"
        
        # Mover archivo
        dest = dest_dir / source.name
        
        # Si el destino existe, agregar sufijo numérico
        counter = 1
        original_dest = dest
        while dest.exists():
            dest = original_dest.parent / f"{original_dest.stem}_{counter}{original_dest.suffix}"
            counter += 1
        
        shutil.move(str(source), str(dest))
        
        # Verificar hash después de mover
        dest_hash = get_file_hash(dest)
        if dest_hash is None:
            return False, "Error calculando hash destino"
        
        if source_hash != dest_hash:
            # Restaurar archivo si los hashes no coinciden
            shutil.move(str(dest), str(source))
            return False, "Hash no coincide - archivo restaurado"
        
        log_message(f"✓ [{category_name}] Movido: {source.name}")
        return True, "OK"
        
    except Exception as e:
        log_message(f"✗ [{category_name}] Error moviendo {source.name}: {e}", is_error=True)
        return False, str(e)

def move_directory_safely(source, dest_dir, category_name):
    """Mueve un directorio completo de forma segura"""
    if not source.exists() or not source.is_dir():
        return False, "No existe o no es directorio"
    
    try:
        dest = dest_dir / source.name
        
        # Si el destino existe, agregar sufijo numérico
        counter = 1
        original_dest = dest
        while dest.exists():
            dest = original_dest.parent / f"{original_dest.stem}_{counter}"
            counter += 1
        
        shutil.move(str(source), str(dest))
        log_message(f"✓ [{category_name}] Movido directorio: {source.name}/")
        return True, "OK"
        
    except Exception as e:
        log_message(f"✗ [{category_name}] Error moviendo directorio {source.name}: {e}", is_error=True)
        return False, str(e)

def remove_pycache_recursively(directory):
    """Elimina todos los directorios __pycache__ recursivamente"""
    removed_count = 0
    for root, dirs, files in os.walk(directory):
        if '__pycache__' in dirs:
            pycache_path = Path(root) / '__pycache__'
            try:
                shutil.rmtree(pycache_path)
                log_message(f"✓ [CACHE] Eliminado: {pycache_path.relative_to(PROJECT_DIR)}")
                removed_count += 1
            except Exception as e:
                log_message(f"✗ [CACHE] Error eliminando {pycache_path}: {e}", is_error=True)
    return removed_count

# Iniciar limpieza
log_message("="*70)
log_message(f"INICIANDO LIMPIEZA DE ARCHIVOS TEMPORALES")
log_message(f"Proyecto: {PROJECT_DIR}")
log_message(f"Destino: {backup_dir}")
log_message("="*70)

stats = {
    'moved': 0,
    'skipped': 0,
    'errors': 0,
}

# 1. Archivos pickle de estados de entrenamiento
log_message("\n=== 1. ARCHIVOS PICKLE ===")
pickle_files = [
    "ba_increasing_n_last_states.pkl",
]
for pkl_file in pickle_files:
    source = PROJECT_DIR / pkl_file
    success, msg = move_file_safely(source, categories['pkl_files'], "PKL")
    if success:
        stats['moved'] += 1
    elif msg == "No existe":
        stats['skipped'] += 1
    else:
        stats['errors'] += 1

# 2. Archivos CSV generados
log_message("\n=== 2. ARCHIVOS CSV GENERADOS ===")
csv_files = [
    "board.csv",
    "piece_map.csv",
]
for csv_file in csv_files:
    source = PROJECT_DIR / csv_file
    success, msg = move_file_safely(source, categories['csv_files'], "CSV")
    if success:
        stats['moved'] += 1
    elif msg == "No existe":
        stats['skipped'] += 1
    else:
        stats['errors'] += 1

# 3. Imágenes temporales
log_message("\n=== 3. IMÁGENES TEMPORALES ===")
image_files = [
    "img.png",
]
for img_file in image_files:
    source = PROJECT_DIR / img_file
    success, msg = move_file_safely(source, categories['images'], "IMG")
    if success:
        stats['moved'] += 1
    elif msg == "No existe":
        stats['skipped'] += 1
    else:
        stats['errors'] += 1

# 4. Archivos de log
log_message("\n=== 4. ARCHIVOS DE LOG ===")
log_patterns = ["training.log", "training_*.log", "tournament_results.txt", "tournament_results_*.txt"]
for pattern in log_patterns:
    for log_path in PROJECT_DIR.glob(pattern):
        if log_path.is_file():
            success, msg = move_file_safely(log_path, categories['logs'], "LOG")
            if success:
                stats['moved'] += 1
            else:
                stats['errors'] += 1

# 5. Directorios de resultados de torneos
log_message("\n=== 5. DIRECTORIOS DE TORNEOS ===")
tournament_dirs = [
    "torneomasivo",
    "torneomasivo_cli",
    "tournament_parallel",
    "tournament_parallel_massive",
    "tournaments_parallel",
]
for tour_dir in tournament_dirs:
    source = PROJECT_DIR / tour_dir
    success, msg = move_directory_safely(source, categories['tournament_dirs'], "TORNEO")
    if success:
        stats['moved'] += 1
    elif msg == "No existe o no es directorio":
        stats['skipped'] += 1
    else:
        stats['errors'] += 1

# 6. Resultados de análisis antiguos
log_message("\n=== 6. RESULTADOS DE ANÁLISIS ===")
analysis_subdirs = [
    "analysis/agent_comparisons",
]
for subdir in analysis_subdirs:
    source = PROJECT_DIR / subdir
    if source.exists() and source.is_dir():
        success, msg = move_directory_safely(source, categories['analysis'], "ANALYSIS")
        if success:
            stats['moved'] += 1
        else:
            stats['errors'] += 1
    else:
        stats['skipped'] += 1

# 7. Archivos de caché Python
log_message("\n=== 7. CACHÉ PYTHON ===")
cache_removed = remove_pycache_recursively(PROJECT_DIR)
stats['moved'] += cache_removed

# Buscar archivos .pyc y .pyo
pyc_files = list(PROJECT_DIR.rglob("*.pyc")) + list(PROJECT_DIR.rglob("*.pyo"))
for pyc_file in pyc_files:
    try:
        pyc_file.unlink()
        log_message(f"✓ [CACHE] Eliminado: {pyc_file.relative_to(PROJECT_DIR)}")
        stats['moved'] += 1
    except Exception as e:
        log_message(f"✗ [CACHE] Error eliminando {pyc_file}: {e}", is_error=True)
        stats['errors'] += 1

# 8. Checkpoints antiguos (opcional - solo si hay muchos)
log_message("\n=== 8. CHECKPOINTS ANTIGUOS (OPCIONAL) ===")
log_message("Nota: Los checkpoints NO se mueven automáticamente.")
log_message("Si deseas mover checkpoints antiguos, hazlo manualmente.")

# 9. Otros archivos temporales comunes
log_message("\n=== 9. OTROS ARCHIVOS TEMPORALES ===")
temp_patterns = ["*.tmp", "*.temp", "*.bak", "*.swp", "*~"]
for pattern in temp_patterns:
    for temp_file in PROJECT_DIR.rglob(pattern):
        if temp_file.is_file():
            success, msg = move_file_safely(temp_file, categories['other'], "TEMP")
            if success:
                stats['moved'] += 1
            else:
                stats['errors'] += 1

# Resumen final
log_message("\n" + "="*70)
log_message("RESUMEN DE LIMPIEZA:")
log_message(f"  ✓ Archivos/directorios movidos: {stats['moved']}")
log_message(f"  ○ Archivos omitidos (no existen): {stats['skipped']}")
log_message(f"  ✗ Errores: {stats['errors']}")
log_message(f"\nUbicación de respaldo: {backup_dir}")
log_message(f"Log de operaciones: {log_file}")
if stats['errors'] > 0:
    log_message(f"Log de errores: {errors_log}")
log_message("="*70)

# Crear resumen en archivo de texto
summary_file = backup_dir / "RESUMEN.txt"
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("RESUMEN DE LIMPIEZA DEL PROYECTO hierarchical-SAE\n")
    f.write("="*70 + "\n\n")
    f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Proyecto original: {PROJECT_DIR}\n")
    f.write(f"Ubicación de respaldo: {backup_dir}\n\n")
    f.write(f"Archivos/directorios movidos: {stats['moved']}\n")
    f.write(f"Archivos omitidos: {stats['skipped']}\n")
    f.write(f"Errores: {stats['errors']}\n\n")
    f.write("CATEGORÍAS:\n")
    for cat_name, cat_path in categories.items():
        items = list(cat_path.iterdir())
        f.write(f"  - {cat_name}: {len(items)} items\n")

log_message(f"\nResumen guardado en: {summary_file}")
log_message("\n¡Limpieza completada!")