"""
Script para ejecutar el monitor de checkpoints por grupos de épocas.

Este script divide los checkpoints en grupos de N épocas (por defecto 10) y
ejecuta el monitor para cada grupo, seleccionando el mejor modelo de cada grupo.
"""

import os
import sys
import subprocess
import argparse
import re
import datetime
import glob
from collections import defaultdict

def extract_epoch_number(checkpoint_path):
    """Extrae el número de época del nombre del archivo checkpoint."""
    filename = os.path.basename(checkpoint_path)

    # Buscar un patrón como "epoch_0123" o "_epoch_123" en el nombre
    match = re.search(r'(?:_)?epoch_?(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Patrón alternativo: buscar simplemente números al final del nombre
    match = re.search(r'(\d+)(?:\.|$)', filename)
    if match:
        return int(match.group(1))

    return -1  # No se pudo extraer un número de época

def group_checkpoints_by_epoch(checkpoints_dir, group_size=10):
    """Agrupa los checkpoints por grupos de épocas."""
    # Obtener todos los archivos de checkpoint
    checkpoint_files = []
    for ext in ['.pt', '.pth']:
        checkpoint_files.extend(glob.glob(os.path.join(checkpoints_dir, f'*{ext}')))

    # Extraer el número de época de cada archivo y organizarlos
    epoch_to_file = {}
    for file_path in checkpoint_files:
        epoch = extract_epoch_number(file_path)
        if epoch >= 0:  # Solo considerar archivos con número de época válido
            epoch_to_file[epoch] = file_path

    # Ordenar las épocas
    sorted_epochs = sorted(epoch_to_file.keys())

    # Agrupar en grupos de 'group_size'
    groups = defaultdict(list)
    for epoch in sorted_epochs:
        group_id = epoch // group_size
        groups[group_id].append(epoch_to_file[epoch])

    return groups

def main():
    parser = argparse.ArgumentParser(description="Ejecutar monitor de checkpoints por grupos de épocas")
    parser.add_argument("--weights-dir", default="models/weights/QuartoCNN1",
                      help="Directorio donde se generan los checkpoints originales")
    parser.add_argument("--group-size", type=int, default=10,
                      help="Tamaño de cada grupo de épocas")
    parser.add_argument("--games", type=int, default=5,
                      help="Número de juegos para evaluar cada modelo")
    parser.add_argument("--output-dir", default="models/best_models_by_group",
                      help="Directorio base donde se guardarán los mejores modelos por grupo")
    parser.add_argument("--log", action="store_true",
                      help="Guardar log de la ejecución")

    args = parser.parse_args()

    # Crear directorio de salida si no existe
    os.makedirs(args.output_dir, exist_ok=True)

    # Crear directorio de logs si es necesario y la opción está activada
    log_dir = "checkpoint_monitor/logs/groups"
    if args.log:
        os.makedirs(log_dir, exist_ok=True)

    # Generar timestamp para esta ejecución
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Agrupar checkpoints
    print(f"Agrupando checkpoints en grupos de {args.group_size} épocas...")
    groups = group_checkpoints_by_epoch(args.weights_dir, args.group_size)

    print(f"Se encontraron {len(groups)} grupos de épocas.")

    # Procesar cada grupo
    for group_id, checkpoint_files in groups.items():
        # Crear subdirectorio para este grupo
        start_epoch = group_id * args.group_size
        end_epoch = (group_id + 1) * args.group_size - 1
        group_dir = os.path.join(args.output_dir, f"epoch_{start_epoch}_to_{end_epoch}")
        os.makedirs(group_dir, exist_ok=True)

        # Nombre de archivo de log para este grupo
        group_log_file = os.path.join(log_dir, f"group_{start_epoch}_to_{end_epoch}_{timestamp}.log")

        print(f"\nProcesando grupo {group_id}: épocas {start_epoch}-{end_epoch}")
        print(f"Se encontraron {len(checkpoint_files)} checkpoints en este grupo.")

        # Guardar lista de checkpoints a evaluar en un archivo temporal
        temp_list_file = os.path.join(args.output_dir, f"temp_list_{group_id}.txt")
        with open(temp_list_file, 'w') as f:
            for checkpoint in checkpoint_files:
                f.write(f"{checkpoint}\n")

        # Construir comando para este grupo
        cmd = [
            sys.executable,
            "run_checkpoint_monitor.py",
            "--source-list", temp_list_file,  # Usar la lista de checkpoints específica
            "--best-dir", group_dir,  # Guardar en el directorio del grupo
            "--games", str(args.games),
            "--evaluate-all"  # Evaluar todos los checkpoints del grupo
        ]

        print(f"Ejecutando evaluación para el grupo {group_id}...")

        # Ejecutar el comando
        if args.log:
            print(f"Guardando log en: {group_log_file}")
            with open(group_log_file, 'w') as f:
                f.write(f"=== Evaluación del grupo {group_id} (épocas {start_epoch}-{end_epoch}) ===\n")
                f.write(f"Fecha y hora: {datetime.datetime.now()}\n")
                f.write(f"Comando: {' '.join(cmd)}\n\n")

                # Ejecutar con redirección de salida al archivo de log
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in process.stdout:
                    f.write(line)
                    print(line, end='')  # También mostrar en consola

                process.wait()

                f.write(f"\nProceso completado con código de salida: {process.returncode}\n")
        else:
            # Ejecutar sin guardar log
            subprocess.run(cmd)

        # Eliminar archivo temporal
        if os.path.exists(temp_list_file):
            os.remove(temp_list_file)

    print("\nProceso completo. Se han evaluado todos los grupos de épocas.")
    print(f"Los mejores modelos de cada grupo están disponibles en: {args.output_dir}")

if __name__ == "__main__":
    main()
