"""
Script para ejecutar automáticamente el monitor de checkpoints.

Este script está diseñado para ser ejecutado periódicamente mediante una tarea programada
o cron job, para mantener una monitorización constante de los checkpoints generados.
"""

import os
import sys
import subprocess
import datetime
import argparse

def main():
    parser = argparse.ArgumentParser(description="Ejecutar automáticamente el monitor de checkpoints")
    parser.add_argument("--interval", type=int, default=1,
                        help="Intervalo de ejecución en horas (para fines informativos)")
    parser.add_argument("--weights-dir", default="models/weights/QuartoCNN1",
                        help="Directorio donde se generan los checkpoints originales")
    parser.add_argument("--games", type=int, default=5,
                        help="Número de juegos para evaluar cada modelo")
    parser.add_argument("--latest", type=int, default=5,
                        help="Evaluar solo los N checkpoints más recientes")
    parser.add_argument("--log", action="store_true",
                        help="Guardar log de la ejecución")

    args = parser.parse_args()

    # Crear directorio de logs si es necesario
    log_dir = "checkpoint_monitor/logs"
    os.makedirs(log_dir, exist_ok=True)

    # Generar nombre de archivo de log con timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"auto_monitor_{timestamp}.log")

    # Construir comando para ejecutar el monitor
    cmd = [
        sys.executable,
        "run_checkpoint_monitor.py",
        "--source-dir", args.weights_dir,
        "--games", str(args.games),
        "--evaluate-latest", str(args.latest),
        "--new-folder"  # Siempre crear una nueva carpeta para los modelos
    ]

    print(f"=== Ejecución automática del Monitor de Checkpoints ===")
    print(f"Fecha y hora: {datetime.datetime.now()}")
    print(f"Intervalo configurado: Cada {args.interval} hora(s)")
    print(f"Directorio de checkpoints: {args.weights_dir}")
    print(f"Número de juegos: {args.games}")
    print(f"Checkpoints a evaluar: {args.latest} más recientes")

    # Ejecutar el comando
    if args.log:
        print(f"Guardando log en: {log_file}")
        with open(log_file, 'w') as f:
            f.write(f"=== Ejecución automática del Monitor de Checkpoints ===\n")
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

    print("Proceso de monitoreo automático completado.")

if __name__ == "__main__":
    main()
