import os
import torch
import shutil
from utils.logger import logger

class ModelCheckpointer:
    """
    Administra los checkpoints del modelo durante el entrenamiento, guardando los
    mejores modelos basados en métricas de rendimiento.
    """

    def __init__(self, experiment_name, base_dir="./models", max_checkpoints=5):
        """
        Inicializa el gestor de checkpoints.

        Args:
            experiment_name: Nombre del experimento para nombrar directorios
            base_dir: Directorio base donde se almacenarán los checkpoints
            max_checkpoints: Número máximo de checkpoints regulares a mantener
        """
        self.experiment_name = experiment_name
        self.max_checkpoints = max_checkpoints
        self.best_performance = -float('inf')
        self.best_epoch = -1

        # Crear estructura de directorios
        self.checkpoints_dir = os.path.join(base_dir, "checkpoints", experiment_name)
        self.best_model_dir = os.path.join(base_dir, "best_models", experiment_name)

        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.best_model_dir, exist_ok=True)

        logger.info(f"Checkpoint manager inicializado para experimento: {experiment_name}")
        logger.info(f"Checkpoints regulares se guardarán en: {self.checkpoints_dir}")
        logger.info(f"Los mejores modelos se guardarán en: {self.best_model_dir}")

    def save_checkpoint(self, model, epoch, metrics=None):
        """
        Guarda un checkpoint del modelo y evalúa si es el mejor hasta ahora.

        Args:
            model: Modelo de PyTorch a guardar
            epoch: Número de época actual
            metrics: Diccionario con métricas de rendimiento (resultados del torneo)

        Returns:
            checkpoint_path: Ruta al checkpoint guardado
        """
        # Usar el método export_model del modelo para guardar el checkpoint
        checkpoint_name = f"{self.experiment_name}_epoch_{epoch:04d}"
        checkpoint_path = model.export_model(checkpoint_name)

        logger.debug(f"Checkpoint guardado: {checkpoint_path}")

        # Si hay métricas disponibles, evaluar si este es el mejor modelo
        if metrics is not None:
            performance = self._calculate_performance(metrics)

            if performance > self.best_performance:
                self.best_performance = performance
                self.best_epoch = epoch

                # Guardar una copia en el directorio de mejores modelos
                best_model_path = os.path.join(self.best_model_dir, f"best_model_epoch_{epoch:04d}.pt")
                latest_best_path = os.path.join(self.best_model_dir, "best_model.pt")

                # Copiar el archivo de checkpoint al directorio de mejores modelos
                shutil.copy(checkpoint_path, best_model_path)
                shutil.copy(checkpoint_path, latest_best_path)

                logger.info(f"¡Nuevo mejor modelo encontrado en la época {epoch}!")
                logger.info(f"Rendimiento: {performance:.4f}")

            # Limpiar checkpoints antiguos
            self._cleanup_old_checkpoints()

        return checkpoint_path

    def _calculate_performance(self, metrics):
        """
        Calcula un valor de rendimiento basado en las métricas del torneo.

        Args:
            metrics: Diccionario con resultados contra cada rival

        Returns:
            float: Valor de rendimiento (promedio de tasas de victoria)
        """
        if not metrics:
            return -float('inf')

        win_rates = []
        for rival, results in metrics.items():
            total = results["wins"] + results["draws"] + results["losses"]
            if total > 0:
                win_rate = (results["wins"] + 0.5 * results["draws"]) / total
                win_rates.append(win_rate)

        return sum(win_rates) / len(win_rates) if win_rates else -float('inf')

    def _cleanup_old_checkpoints(self):
        """
        Elimina checkpoints antiguos si se excede el número máximo establecido.
        Siempre conserva el mejor modelo.
        """
        if self.max_checkpoints <= 0:
            return

        # Obtener lista de archivos de checkpoint ordenados por época
        checkpoints = []
        for filename in os.listdir(self.checkpoints_dir):
            if filename.startswith(f"{self.experiment_name}_epoch_") and filename.endswith(".pt"):
                epoch = int(filename.replace(f"{self.experiment_name}_epoch_", "").replace(".pt", ""))
                checkpoints.append((epoch, os.path.join(self.checkpoints_dir, filename)))

        # Ordenar por época (ascendente)
        checkpoints.sort()

        # Mantener solo los más recientes
        if len(checkpoints) > self.max_checkpoints:
            checkpoints_to_delete = checkpoints[:-self.max_checkpoints]
            for epoch, checkpoint_path in checkpoints_to_delete:
                # No eliminar el mejor modelo
                if epoch != self.best_epoch:
                    try:
                        os.remove(checkpoint_path)
                        logger.debug(f"Checkpoint antiguo eliminado: {checkpoint_path}")
                    except Exception as e:
                        logger.warning(f"No se pudo eliminar checkpoint: {checkpoint_path}. Error: {e}")

    def get_checkpoint_files(self, limit=None):
        """
        Obtiene la lista de archivos de checkpoint más recientes.

        Args:
            limit: Número máximo de checkpoints a devolver

        Returns:
            list: Lista de rutas a los archivos de checkpoint
        """
        checkpoints = []
        for filename in os.listdir(self.checkpoints_dir):
            if filename.startswith(f"{self.experiment_name}_epoch_") and filename.endswith(".pt"):
                checkpoints.append(os.path.join(self.checkpoints_dir, filename))

        # Ordenar por nombre (implícitamente por época)
        checkpoints.sort()

        if limit is not None and limit > 0 and len(checkpoints) > limit:
            return checkpoints[-limit:]
        return checkpoints

    def get_best_model_info(self):
        """
        Obtiene información sobre el mejor modelo.

        Returns:
            tuple: (epoch del mejor modelo, ruta al mejor modelo, rendimiento)
        """
        best_model_path = os.path.join(self.best_model_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            return self.best_epoch, best_model_path, self.best_performance
        return None, None, None
