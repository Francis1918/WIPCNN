"""
Gestor de checkpoints para el entrenamiento de modelos RL.

Este módulo proporciona funcionalidades para gestionar checkpoints generados
durante el entrenamiento, sin modificar el código original.
"""

import os
import torch
import json
import datetime
import shutil
from typing import Dict, Any, Optional, List, Tuple
import logging

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CheckpointManager")

class CheckpointManager:
    """
    Gestor de checkpoints para modelos de aprendizaje por refuerzo.

    Esta clase proporciona funcionalidades para monitorear, copiar y gestionar
    checkpoints generados durante el entrenamiento.
    """

    def __init__(self,
                checkpoint_source_dir: str = "models/checkpoints",
                checkpoint_target_dir: str = "models/checkpoints_monitored",
                best_models_dir: str = "models/best_models_auto"):
        """
        Inicializa el gestor de checkpoints.

        Args:
            checkpoint_source_dir: Directorio donde se generan los checkpoints originales
            checkpoint_target_dir: Directorio donde se almacenarán los checkpoints monitoreados
            best_models_dir: Directorio donde se almacenarán los mejores modelos
        """
        self.checkpoint_source_dir = checkpoint_source_dir
        self.checkpoint_target_dir = checkpoint_target_dir
        self.best_models_dir = best_models_dir

        # Crear directorios si no existen
        os.makedirs(checkpoint_source_dir, exist_ok=True)
        os.makedirs(checkpoint_target_dir, exist_ok=True)
        os.makedirs(best_models_dir, exist_ok=True)

        # Archivo para almacenar métricas de los checkpoints
        self.metrics_file = os.path.join(checkpoint_target_dir, "checkpoint_metrics.json")

        # Cargar métricas existentes si hay
        self.metrics_history = self._load_metrics_history()

        logger.info(f"CheckpointManager inicializado. Directorio origen: {checkpoint_source_dir}")
        logger.info(f"Directorio destino: {checkpoint_target_dir}")
        logger.info(f"Directorio de mejores modelos: {best_models_dir}")

    def _load_metrics_history(self) -> Dict[str, Dict[str, Any]]:
        """Carga el historial de métricas desde el archivo JSON."""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.warning(f"Error al cargar métricas: {e}. Creando nuevo historial.")
                return {}
        return {}

    def _save_metrics_history(self) -> None:
        """Guarda el historial de métricas en el archivo JSON."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=4)

    def copy_checkpoint(self,
                       checkpoint_file: str,
                       metrics: Dict[str, Any] = None,
                       epoch: int = None) -> str:
        """
        Copia un checkpoint al directorio monitoreado y registra sus métricas.

        Args:
            checkpoint_file: Ruta completa al archivo de checkpoint
            metrics: Métricas asociadas al checkpoint
            epoch: Número de época del checkpoint

        Returns:
            Ruta del checkpoint copiado
        """
        # Obtener solo el nombre del archivo sin la ruta
        checkpoint_name = os.path.basename(checkpoint_file)

        # Ruta destino
        target_path = os.path.join(self.checkpoint_target_dir, checkpoint_name)

        # Copiar el archivo
        shutil.copy2(checkpoint_file, target_path)

        # Registrar métricas
        if metrics:
            self.metrics_history[checkpoint_name] = {
                'epoch': epoch if epoch is not None else -1,
                'metrics': metrics,
                'timestamp': datetime.datetime.now().isoformat(),
                'source_path': checkpoint_file,
                'target_path': target_path
            }
            self._save_metrics_history()

        logger.info(f"Checkpoint copiado: {checkpoint_name}")
        return target_path

    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Obtiene el checkpoint más reciente basado en la fecha de modificación.

        Returns:
            Ruta al checkpoint más reciente o None si no hay checkpoints
        """
        checkpoints = self._scan_checkpoints_directory()
        if not checkpoints:
            return None

        # Ordenar por fecha de modificación (más reciente primero)
        checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return checkpoints[0]

    def _scan_checkpoints_directory(self) -> List[str]:
        """
        Escanea el directorio de checkpoints originales.

        Returns:
            Lista de rutas a archivos de checkpoint
        """
        if not os.path.exists(self.checkpoint_source_dir):
            logger.warning(f"El directorio de checkpoints {self.checkpoint_source_dir} no existe")
            return []

        # Buscar archivos .pt o .pth (formatos comunes para modelos PyTorch)
        checkpoints = []
        for file in os.listdir(self.checkpoint_source_dir):
            if file.endswith(('.pt', '.pth')):
                checkpoints.append(os.path.join(self.checkpoint_source_dir, file))

        return checkpoints

    def get_all_checkpoints(self) -> List[str]:
        """
        Obtiene todos los checkpoints disponibles.

        Returns:
            Lista de rutas a archivos de checkpoint
        """
        return self._scan_checkpoints_directory()

    def get_new_checkpoints(self, last_check_time: float = 0) -> List[str]:
        """
        Obtiene los checkpoints nuevos o modificados desde la última verificación.

        Args:
            last_check_time: Tiempo de la última verificación (timestamp)

        Returns:
            Lista de rutas a checkpoints nuevos o modificados
        """
        checkpoints = self._scan_checkpoints_directory()
        new_checkpoints = []

        for checkpoint in checkpoints:
            mod_time = os.path.getmtime(checkpoint)
            if mod_time > last_check_time:
                new_checkpoints.append(checkpoint)

        return new_checkpoints

    def get_best_checkpoint(self,
                           metric_name: str = "win_rate",
                           higher_is_better: bool = True) -> Optional[Tuple[str, Dict]]:
        """
        Obtiene el mejor checkpoint basado en una métrica específica.

        Args:
            metric_name: Nombre de la métrica para evaluar
            higher_is_better: True si valores más altos son mejores, False en caso contrario

        Returns:
            Tupla con (ruta al mejor checkpoint, datos de métricas) o None si no hay checkpoints
        """
        if not self.metrics_history:
            logger.warning("No hay historial de métricas para seleccionar el mejor checkpoint")
            return None

        best_checkpoint = None
        best_metric_value = float('-inf') if higher_is_better else float('inf')
        best_metrics_data = None

        for checkpoint_name, metrics_data in self.metrics_history.items():
            if 'metrics' in metrics_data and metric_name in metrics_data['metrics']:
                metric_value = metrics_data['metrics'][metric_name]

                # Determinar si este checkpoint es mejor
                is_better = (metric_value > best_metric_value) if higher_is_better else (metric_value < best_metric_value)

                if is_better:
                    best_metric_value = metric_value
                    best_checkpoint = metrics_data.get('target_path')
                    best_metrics_data = metrics_data

        if best_checkpoint:
            logger.info(f"Mejor checkpoint encontrado: {os.path.basename(best_checkpoint)} "
                       f"con {metric_name}={best_metric_value}")
            return best_checkpoint, best_metrics_data

        logger.warning(f"No se encontró ningún checkpoint con la métrica {metric_name}")
        return None

    def save_best_model(self,
                       metric_name: str = "win_rate",
                       higher_is_better: bool = True) -> Optional[str]:
        """
        Guarda el mejor modelo en el directorio de mejores modelos.

        Args:
            metric_name: Nombre de la métrica para evaluar
            higher_is_better: True si valores más altos son mejores, False en caso contrario

        Returns:
            Ruta al mejor modelo guardado o None si no hay checkpoints
        """
        best_result = self.get_best_checkpoint(metric_name, higher_is_better)
        if not best_result:
            return None

        best_checkpoint, metrics_data = best_result

        # Crear nombre de archivo para el mejor modelo
        epoch = metrics_data.get('epoch', 'unknown')
        metric_value = metrics_data['metrics'][metric_name]
        best_model_name = f"best_{metric_name}_{metric_value:.4f}_epoch_{epoch}.pt"
        best_model_path = os.path.join(self.best_models_dir, best_model_name)

        # Copiar el mejor modelo
        shutil.copy2(best_checkpoint, best_model_path)
        logger.info(f"Mejor modelo guardado en: {best_model_path}")

        # Actualizar el archivo de mejores modelos
        best_models_file = os.path.join(self.best_models_dir, "best_models.json")
        best_models_data = {}

        if os.path.exists(best_models_file):
            try:
                with open(best_models_file, 'r') as f:
                    best_models_data = json.load(f)
            except json.JSONDecodeError:
                pass

        # Registrar este mejor modelo
        best_models_data[metric_name] = {
            'model_path': best_model_path,
            'metric_value': metric_value,
            'epoch': epoch,
            'timestamp': datetime.datetime.now().isoformat(),
            'checkpoint_source': best_checkpoint
        }

        with open(best_models_file, 'w') as f:
            json.dump(best_models_data, f, indent=4)

        return best_model_path

    def extract_metrics_from_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Extrae métricas de un archivo de checkpoint si están disponibles.

        Args:
            checkpoint_path: Ruta al archivo de checkpoint

        Returns:
            Diccionario con métricas extraídas o diccionario vacío si no hay métricas
        """
        try:
            # Cargar el checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Intentar extraer métricas
            metrics = {}

            # Revisar si el checkpoint tiene formato específico con métricas
            if isinstance(checkpoint, dict):
                # Buscar métricas en diferentes ubicaciones comunes
                if 'metrics' in checkpoint:
                    metrics = checkpoint['metrics']
                elif 'win_rate' in checkpoint:
                    # Extraer métricas individuales
                    metrics['win_rate'] = checkpoint.get('win_rate')
                    metrics['loss'] = checkpoint.get('loss')

                # Intentar extraer la época
                if 'epoch' in checkpoint:
                    metrics['epoch'] = checkpoint['epoch']

            return metrics

        except (RuntimeError, ValueError) as e:
            logger.warning(f"Error al extraer métricas del checkpoint {checkpoint_path}: {e}")
            return {}
