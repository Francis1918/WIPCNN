"""
Monitor de entrenamiento para selección automática de mejores modelos.

Este script monitorea los checkpoints generados durante el entrenamiento y
selecciona automáticamente los mejores modelos basados en diferentes criterios.
"""

import os
import time
import glob
import argparse
import logging
from typing import Dict, List, Optional, Any
import threading
import json
from datetime import datetime

from .checkpoint_manager import CheckpointManager
from .model_evaluator import ModelEvaluator
from .visualize import ModelVisualizer

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("checkpoint_monitor/monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ModelMonitor")

class ModelMonitor:
    """
    Monitor de entrenamiento para selección automática de mejores modelos.

    Esta clase proporciona funcionalidades para monitorear los checkpoints generados
    durante el entrenamiento y seleccionar automáticamente los mejores modelos.
    """

    def __init__(self,
                checkpoint_source_dir: str = "models/checkpoints",
                checkpoint_target_dir: str = "models/checkpoints_monitored",
                best_models_dir: str = "models/best_models_auto",
                evaluation_results_dir: str = "models/evaluation_results",
                visualization_dir: str = "checkpoint_monitor/visualizations",
                check_interval: int = 300,
                num_evaluation_games: int = 10):
        """
        Inicializa el monitor de modelos.

        Args:
            checkpoint_source_dir: Directorio donde se generan los checkpoints originales
            checkpoint_target_dir: Directorio donde se almacenarán los checkpoints monitoreados
            best_models_dir: Directorio donde se almacenarán los mejores modelos
            evaluation_results_dir: Directorio donde se guardarán los resultados de evaluación
            visualization_dir: Directorio donde se guardarán las visualizaciones
            check_interval: Intervalo de verificación en segundos
            num_evaluation_games: Número de juegos para evaluar cada modelo
        """
        # Inicializar componentes
        self.checkpoint_manager = CheckpointManager(
            checkpoint_source_dir=checkpoint_source_dir,
            checkpoint_target_dir=checkpoint_target_dir,
            best_models_dir=best_models_dir
        )

        self.model_evaluator = ModelEvaluator(
            evaluation_results_dir=evaluation_results_dir,
            num_evaluation_games=num_evaluation_games
        )

        self.visualizer = ModelVisualizer(
            checkpoints_metrics_file=os.path.join(checkpoint_target_dir, "checkpoint_metrics.json"),
            evaluation_results_file=os.path.join(evaluation_results_dir, "evaluation_results.json"),
            output_dir=visualization_dir
        )

        self.check_interval = check_interval
        self.running = False
        self.monitor_thread = None
        self.last_check_time = 0

        # Archivos de estado
        self.status_file = os.path.join(checkpoint_target_dir, "monitor_status.json")
        self.status = self._load_status()

        logger.info(f"ModelMonitor inicializado. Intervalo de verificación: {check_interval} segundos")
        logger.info(f"Número de juegos para evaluación: {num_evaluation_games}")

    def _load_status(self) -> Dict[str, Any]:
        """Carga el estado del monitor desde el archivo JSON."""
        if os.path.exists(self.status_file):
            try:
                with open(self.status_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.warning(f"Error al cargar estado del monitor: {e}. Creando nuevo estado.")
                return {'last_check_time': 0, 'monitored_checkpoints': []}
        return {'last_check_time': 0, 'monitored_checkpoints': []}

    def _save_status(self) -> None:
        """Guarda el estado del monitor en el archivo JSON."""
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=4)

    def start(self) -> None:
        """Inicia el monitor en un hilo separado."""
        if self.running:
            logger.warning("El monitor ya está en ejecución")
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        logger.info("Monitor iniciado en segundo plano")

    def stop(self) -> None:
        """Detiene el monitor."""
        if not self.running:
            logger.warning("El monitor no está en ejecución")
            return

        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        logger.info("Monitor detenido")

    def _monitor_loop(self) -> None:
        """Bucle principal del monitor."""
        logger.info("Iniciando bucle de monitoreo")

        while self.running:
            try:
                self._check_for_new_checkpoints()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error en el bucle de monitoreo: {e}")
                time.sleep(10)  # Esperar un poco antes de reintentar

    def _check_for_new_checkpoints(self) -> None:
        """Verifica si hay nuevos checkpoints."""
        # Obtener tiempo de la última verificación
        last_check_time = self.status.get('last_check_time', 0)

        # Buscar nuevos checkpoints
        new_checkpoints = self.checkpoint_manager.get_new_checkpoints(last_check_time)

        if new_checkpoints:
            logger.info(f"Se encontraron {len(new_checkpoints)} nuevos checkpoints")

            # Procesar cada nuevo checkpoint
            for checkpoint_path in new_checkpoints:
                self._process_checkpoint(checkpoint_path)

            # Seleccionar mejores modelos
            self._select_best_models()

            # Generar visualizaciones
            self._generate_visualizations()

        # Actualizar tiempo de última verificación
        self.status['last_check_time'] = time.time()
        self._save_status()

    def _process_checkpoint(self, checkpoint_path: str) -> None:
        """
        Procesa un nuevo checkpoint.

        Args:
            checkpoint_path: Ruta al archivo de checkpoint
        """
        logger.info(f"Procesando checkpoint: {os.path.basename(checkpoint_path)}")

        # Extraer época del nombre del archivo si es posible
        epoch = None
        checkpoint_name = os.path.basename(checkpoint_path)

        if "_epoch_" in checkpoint_name:
            try:
                epoch = int(checkpoint_name.split("_epoch_")[1].split("_")[0])
            except (IndexError, ValueError):
                pass

        # Extraer métricas del checkpoint
        metrics = self.checkpoint_manager.extract_metrics_from_checkpoint(checkpoint_path)

        # Copiar checkpoint al directorio monitoreado
        target_path = self.checkpoint_manager.copy_checkpoint(
            checkpoint_path, metrics, epoch
        )

        # Evaluar el modelo
        if epoch is not None:
            logger.info(f"Evaluando modelo de época {epoch}")

            # Evaluar contra bot aleatorio
            random_results = self.model_evaluator.evaluate_against_random(target_path)

            # Actualizar métricas con resultados de evaluación
            if 'win_rate' in random_results:
                if 'metrics' not in metrics:
                    metrics = {'metrics': {}}

                metrics['metrics']['win_rate_vs_random'] = random_results['win_rate']

                # Actualizar métricas en el checkpoint manager
                self.checkpoint_manager.metrics_history[checkpoint_name]['metrics'] = metrics['metrics']
                self.checkpoint_manager._save_metrics_history()

        # Añadir a lista de checkpoints monitoreados
        if checkpoint_path not in self.status['monitored_checkpoints']:
            self.status['monitored_checkpoints'].append(checkpoint_path)

    def _select_best_models(self) -> None:
        """Selecciona los mejores modelos basados en diferentes criterios."""
        logger.info("Seleccionando mejores modelos")

        # Guardar mejor modelo basado en tasa de victoria contra bot aleatorio
        best_model_path = self.checkpoint_manager.save_best_model(
            metric_name="win_rate_vs_random", higher_is_better=True
        )

        if best_model_path:
            logger.info(f"Mejor modelo guardado en: {best_model_path}")

    def _generate_visualizations(self) -> None:
        """Genera visualizaciones del progreso del entrenamiento."""
        logger.info("Generando visualizaciones")

        # Crear visualizador con datos actualizados
        visualizer = ModelVisualizer(
            checkpoints_metrics_file=os.path.join(
                self.checkpoint_manager.checkpoint_target_dir,
                "checkpoint_metrics.json"
            ),
            evaluation_results_file=os.path.join(
                self.model_evaluator.evaluation_results_dir,
                "evaluation_results.json"
            ),
            output_dir=self.visualizer.output_dir
        )

        # Generar todas las visualizaciones
        generated_files = visualizer.generate_all_visualizations()
        logger.info(f"Se generaron {len(generated_files)} visualizaciones")

    def run_once(self) -> None:
        """Ejecuta una iteración del monitor."""
        logger.info("Ejecutando verificación única")
        self._check_for_new_checkpoints()
        logger.info("Verificación completada")

    def force_evaluation(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Fuerza la evaluación de un checkpoint específico.

        Args:
            checkpoint_path: Ruta al archivo de checkpoint

        Returns:
            Resultados de la evaluación
        """
        logger.info(f"Forzando evaluación de: {os.path.basename(checkpoint_path)}")

        # Verificar que el archivo existe
        if not os.path.exists(checkpoint_path):
            logger.error(f"El archivo {checkpoint_path} no existe")
            return {'error': 'Archivo no encontrado'}

        # Evaluar contra bot aleatorio
        results = self.model_evaluator.evaluate_against_random(checkpoint_path)

        # Procesar el checkpoint
        self._process_checkpoint(checkpoint_path)

        # Seleccionar mejores modelos
        self._select_best_models()

        # Generar visualizaciones
        self._generate_visualizations()

        return results

    def force_visualizations(self) -> List[str]:
        """
        Fuerza la generación de visualizaciones.

        Returns:
            Lista de rutas a los archivos generados
        """
        logger.info("Forzando generación de visualizaciones")
        return self.visualizer.generate_all_visualizations()


def run_monitor(args: argparse.Namespace) -> None:
    """
    Función principal para ejecutar el monitor.

    Args:
        args: Argumentos de línea de comandos
    """
    monitor = ModelMonitor(
        checkpoint_source_dir=args.source_dir,
        checkpoint_target_dir=args.target_dir,
        best_models_dir=args.best_dir,
        check_interval=args.interval,
        num_evaluation_games=args.games
    )

    if args.once:
        # Ejecutar una vez y salir
        monitor.run_once()
    else:
        # Ejecutar en segundo plano
        monitor.start()
        try:
            # Mantener el script en ejecución
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Interrupción de usuario. Deteniendo monitor...")
            monitor.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor de checkpoints para entrenamiento de RL")
    parser.add_argument("--source-dir", default="models/checkpoints",
                        help="Directorio donde se generan los checkpoints originales")
    parser.add_argument("--target-dir", default="models/checkpoints_monitored",
                        help="Directorio donde se almacenarán los checkpoints monitoreados")
    parser.add_argument("--best-dir", default="models/best_models_auto",
                        help="Directorio donde se almacenarán los mejores modelos")
    parser.add_argument("--interval", type=int, default=300,
                        help="Intervalo de verificación en segundos")
    parser.add_argument("--games", type=int, default=10,
                        help="Número de juegos para evaluar cada modelo")
    parser.add_argument("--once", action="store_true",
                        help="Ejecutar una vez y salir")

    args = parser.parse_args()
    run_monitor(args)
