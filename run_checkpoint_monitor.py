"""
Script para ejecutar el monitor de checkpoints con los parámetros correctos.

Este script configura y ejecuta el monitor de checkpoints para detectar, evaluar
y seleccionar automáticamente los mejores modelos entrenados.
"""

import os
import sys
import argparse
import datetime
import shutil

# Añadir el directorio raíz al path para poder importar los módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from checkpoint_monitor.checkpoint_manager import CheckpointManager
from checkpoint_monitor.model_evaluator import ModelEvaluator
from checkpoint_monitor.visualize import ModelVisualizer

# Añadir método get_timestamp al CheckpointManager
def get_timestamp():
    """Obtiene la marca de tiempo actual en formato ISO."""
    return datetime.datetime.now().isoformat()

# Crear una función para inicializar una nueva carpeta para mejores modelos
def setup_new_best_models_folder(base_dir="models/best_models_auto"):
    """
    Crea y configura una nueva carpeta para almacenar los mejores modelos.

    Args:
        base_dir: Directorio base para los mejores modelos

    Returns:
        Ruta a la nueva carpeta creada
    """
    # Crear nombre con timestamp para la nueva carpeta
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    new_folder = f"{base_dir}_{timestamp}"

    # Crear la carpeta
    os.makedirs(new_folder, exist_ok=True)

    # Crear archivo README para documentar
    readme_path = os.path.join(new_folder, "README.txt")
    with open(readme_path, "w") as f:
        f.write(f"Carpeta de mejores modelos creada automáticamente el {datetime.datetime.now()}\n")
        f.write("Esta carpeta contiene los mejores modelos seleccionados por el sistema de monitoreo de checkpoints.\n")
        f.write("Los modelos se seleccionan automáticamente basados en su rendimiento contra un bot aleatorio.\n\n")
        f.write("Formato de nombres de archivos:\n")
        f.write("best_[métrica]_[valor]_epoch_[época].pt\n\n")
        f.write("Ejemplo: best_win_rate_vs_random_0.8500_epoch_42.pt\n")

    print(f"Nueva carpeta para mejores modelos creada: {new_folder}")
    return new_folder

# Aplicar monkey patch al CheckpointManager
CheckpointManager._get_timestamp = get_timestamp

def main():
    parser = argparse.ArgumentParser(description="Ejecutar monitor de checkpoints")
    parser.add_argument("--source-dir", default="models/weights/QuartoCNN1",
                      help="Directorio donde se generan los checkpoints originales")
    parser.add_argument("--source-list",
                      help="Archivo con lista de checkpoints específicos a evaluar (un checkpoint por línea)")
    parser.add_argument("--target-dir", default="models/checkpoints_monitored",
                      help="Directorio donde se almacenarán los checkpoints monitoreados")
    parser.add_argument("--best-dir", default="models/best_models_auto",
                      help="Directorio donde se almacenarán los mejores modelos")
    parser.add_argument("--games", type=int, default=5,
                      help="Número de juegos para evaluar cada modelo")
    parser.add_argument("--evaluate-all", action="store_true",
                      help="Evaluar todos los checkpoints (puede tomar mucho tiempo)")
    parser.add_argument("--evaluate-latest", type=int, default=5,
                      help="Evaluar solo los N checkpoints más recientes")
    parser.add_argument("--new-folder", action="store_true",
                      help="Crear una nueva carpeta para almacenar los mejores modelos")

    args = parser.parse_args()

    # Si se solicita una nueva carpeta, crearla
    if args.new_folder:
        args.best_dir = setup_new_best_models_folder(args.best_dir)

    print(f"=== Monitor de Checkpoints para Quarto RL ===")
    print(f"Directorio de origen: {args.source_dir}")
    print(f"Directorio de destino: {args.target_dir}")
    print(f"Directorio de mejores modelos: {args.best_dir}")
    print(f"Número de juegos para evaluación: {args.games}")

    # Inicializar componentes
    checkpoint_manager = CheckpointManager(
        checkpoint_source_dir=args.source_dir,
        checkpoint_target_dir=args.target_dir,
        best_models_dir=args.best_dir
    )

    model_evaluator = ModelEvaluator(
        evaluation_results_dir="models/evaluation_results",
        num_evaluation_games=args.games
    )

    visualizer = ModelVisualizer(
        checkpoints_metrics_file=os.path.join(args.target_dir, "checkpoint_metrics.json"),
        evaluation_results_file=os.path.join("models/evaluation_results", "evaluation_results.json"),
        output_dir="checkpoint_monitor/visualizations"
    )

    # Obtener checkpoints
    print("Escaneando checkpoints existentes...")

    # Leer checkpoints desde un archivo de lista si se proporciona
    if args.source_list and os.path.exists(args.source_list):
        with open(args.source_list, 'r') as f:
            checkpoints = [line.strip() for line in f if line.strip()]
        print(f"Se cargaron {len(checkpoints)} checkpoints desde el archivo {args.source_list}.")
    else:
        # Usar el comportamiento normal de escanear el directorio
        checkpoints = checkpoint_manager.get_all_checkpoints()
        print(f"Se encontraron {len(checkpoints)} checkpoints en total.")

    # Copiar checkpoints al directorio monitoreado
    print("Copiando checkpoints al directorio monitoreado...")
    copied_checkpoints = []

    for checkpoint in checkpoints:
        checkpoint_name = os.path.basename(checkpoint)
        target_path = os.path.join(args.target_dir, checkpoint_name)

        # Si no existe en el directorio de destino, copiarlo
        if not os.path.exists(target_path):
            checkpoint_manager.copy_checkpoint(checkpoint)
            copied_checkpoints.append(target_path)

    print(f"Se copiaron {len(copied_checkpoints)} nuevos checkpoints.")

    # Seleccionar checkpoints para evaluar
    checkpoints_to_evaluate = []

    # Si tenemos una lista de checkpoints específica, usarla directamente
    if args.source_list and os.path.exists(args.source_list):
        # Ya tenemos los checkpoints cargados desde el archivo
        checkpoints_to_evaluate = [
            checkpoint if os.path.exists(checkpoint) else os.path.join(args.target_dir, os.path.basename(checkpoint))
            for checkpoint in checkpoints
        ]
        print(f"Se usará la lista específica de {len(checkpoints_to_evaluate)} checkpoints para evaluación.")
    else:
        # Seleccionar checkpoints del directorio monitoreado
        monitored_checkpoints_dir = args.target_dir
        all_monitored_checkpoints = [
            os.path.join(monitored_checkpoints_dir, f)
            for f in os.listdir(monitored_checkpoints_dir)
            if f.endswith('.pt')
        ]

        # Ordenar por fecha de modificación (más reciente primero)
        all_monitored_checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        if args.evaluate_all:
            checkpoints_to_evaluate = all_monitored_checkpoints
            print(f"Se evaluarán todos los {len(checkpoints_to_evaluate)} checkpoints.")
        else:
            # Tomar solo los N más recientes
            checkpoints_to_evaluate = all_monitored_checkpoints[:args.evaluate_latest]
            print(f"Se evaluarán los {len(checkpoints_to_evaluate)} checkpoints más recientes.")

    # Evaluar checkpoints seleccionados
    print("Evaluando checkpoints contra bot aleatorio...")
    for i, checkpoint in enumerate(checkpoints_to_evaluate):
        print(f"Evaluando checkpoint {i+1}/{len(checkpoints_to_evaluate)}: {os.path.basename(checkpoint)}")
        try:
            # Usamos model_path en lugar de model con temperature
            result = model_evaluator.evaluate_against_random_fixed(checkpoint)
            win_rate = result.get('win_rate', 0)
            print(f"  Win rate: {win_rate:.4f}")

            # Extraer época del nombre del archivo si es posible
            epoch = None
            checkpoint_name = os.path.basename(checkpoint)
            if "_epoch_" in checkpoint_name:
                try:
                    epoch = int(checkpoint_name.split("_epoch_")[1].split("_")[0].split(".")[0])
                except (IndexError, ValueError):
                    pass

            # Actualizar métricas en el checkpoint manager
            if checkpoint_name in checkpoint_manager.metrics_history:
                if 'metrics' not in checkpoint_manager.metrics_history[checkpoint_name]:
                    checkpoint_manager.metrics_history[checkpoint_name]['metrics'] = {}

                checkpoint_manager.metrics_history[checkpoint_name]['metrics']['win_rate_vs_random'] = win_rate
                checkpoint_manager.metrics_history[checkpoint_name]['epoch'] = epoch
            else:
                checkpoint_manager.metrics_history[checkpoint_name] = {
                    'epoch': epoch,
                    'metrics': {'win_rate_vs_random': win_rate},
                    'timestamp': get_timestamp(),
                    'source_path': checkpoint,
                    'target_path': checkpoint
                }

            checkpoint_manager._save_metrics_history()

        except Exception as e:
            print(f"  Error al evaluar: {e}")

    # Seleccionar el mejor modelo
    print("Seleccionando el mejor modelo...")
    best_model_path = checkpoint_manager.save_best_model(
        metric_name="win_rate_vs_random", higher_is_better=True
    )

    if best_model_path:
        print(f"Mejor modelo guardado en: {best_model_path}")
    else:
        print("No se pudo determinar el mejor modelo.")

    # Generar visualizaciones
    print("Generando visualizaciones...")
    generated_files = visualizer.generate_all_visualizations()
    print(f"Se generaron {len(generated_files)} visualizaciones.")

    for file in generated_files:
        print(f"  - {file}")

    print("Proceso completado.")

if __name__ == "__main__":
    main()
