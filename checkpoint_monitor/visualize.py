"""
Visualizador de resultados para el entrenamiento de RL.

Este módulo proporciona funcionalidades para visualizar los resultados del
entrenamiento sin modificar el código original.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ModelVisualizer")

class ModelVisualizer:
    """
    Visualizador de resultados de entrenamiento.

    Esta clase proporciona funcionalidades para visualizar el progreso del
    entrenamiento a partir de los checkpoints generados.
    """

    def __init__(self,
                checkpoints_metrics_file: str = "models/checkpoints_monitored/checkpoint_metrics.json",
                evaluation_results_file: str = "models/evaluation_results/evaluation_results.json",
                output_dir: str = "checkpoint_monitor/visualizations"):
        """
        Inicializa el visualizador de modelos.

        Args:
            checkpoints_metrics_file: Archivo con métricas de checkpoints
            evaluation_results_file: Archivo con resultados de evaluación
            output_dir: Directorio donde se guardarán las visualizaciones
        """
        self.checkpoints_metrics_file = checkpoints_metrics_file
        self.evaluation_results_file = evaluation_results_file
        self.output_dir = output_dir

        # Crear directorio de salida si no existe
        os.makedirs(output_dir, exist_ok=True)

        # Cargar datos
        self.checkpoint_metrics = self._load_json_file(checkpoints_metrics_file)
        self.evaluation_results = self._load_json_file(evaluation_results_file)

        logger.info(f"ModelVisualizer inicializado. Directorio de salida: {output_dir}")

    def _load_json_file(self, file_path: str) -> Dict:
        """Carga datos desde un archivo JSON."""
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.warning(f"Error al cargar archivo {file_path}: {e}. Usando diccionario vacío.")
                return {}
        return {}

    def plot_metric_by_epoch(self,
                           metric_name: str = "win_rate",
                           title: str = "Progreso de Entrenamiento",
                           save_filename: str = None) -> str:
        """
        Genera un gráfico de una métrica por época.

        Args:
            metric_name: Nombre de la métrica a visualizar
            title: Título del gráfico
            save_filename: Nombre del archivo para guardar (sin extensión)

        Returns:
            Ruta al archivo de imagen guardado
        """
        if not self.checkpoint_metrics:
            logger.warning("No hay métricas de checkpoints para visualizar")
            return ""

        epochs = []
        metric_values = []

        # Extraer datos
        for checkpoint_name, data in self.checkpoint_metrics.items():
            if 'epoch' in data and 'metrics' in data and metric_name in data['metrics']:
                epochs.append(data['epoch'])
                metric_values.append(data['metrics'][metric_name])

        if not epochs:
            logger.warning(f"No hay datos de la métrica {metric_name} para graficar")
            return ""

        # Ordenar por época
        sorted_data = sorted(zip(epochs, metric_values))
        epochs, metric_values = zip(*sorted_data)

        # Crear figura
        plt.figure(figsize=(12, 6), dpi=100)
        plt.plot(epochs, metric_values, 'b-o', linewidth=2, markersize=6)
        plt.title(title)
        plt.xlabel('Época')
        plt.ylabel(metric_name)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Añadir línea de tendencia
        if len(epochs) > 1:
            z = np.polyfit(epochs, metric_values, 1)
            p = np.poly1d(z)
            plt.plot(epochs, p(epochs), "r--", alpha=0.8,
                     label=f"Tendencia: {z[0]:.6f}x + {z[1]:.6f}")
            plt.legend()

        # Guardar figura
        if save_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_filename = f"metric_{metric_name}_{timestamp}"

        save_path = os.path.join(self.output_dir, f"{save_filename}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        logger.info(f"Gráfico guardado en: {save_path}")
        return save_path

    def plot_win_rates_against_random(self,
                                    title: str = "Tasa de Victorias contra Bot Aleatorio",
                                    save_filename: str = None) -> str:
        """
        Genera un gráfico de tasas de victoria contra bot aleatorio.

        Args:
            title: Título del gráfico
            save_filename: Nombre del archivo para guardar (sin extensión)

        Returns:
            Ruta al archivo de imagen guardado
        """
        if not self.evaluation_results or 'vs_random' not in self.evaluation_results:
            logger.warning("No hay resultados de evaluación contra bot aleatorio")
            return ""

        random_results = self.evaluation_results['vs_random']
        models = []
        win_rates = []

        # Extraer datos y asociarlos con épocas si es posible
        model_data = []
        for model_name, data in random_results.items():
            # Intentar extraer la época del nombre del modelo si tiene formato "epoch_X"
            epoch = None
            if "_epoch_" in model_name:
                try:
                    epoch = int(model_name.split("_epoch_")[1].split("_")[0])
                except (IndexError, ValueError):
                    pass

            model_data.append((epoch, model_name, data['win_rate']))

        # Ordenar por época si está disponible, de lo contrario por nombre
        model_data.sort()

        # Separar datos ordenados
        for _, model_name, win_rate in model_data:
            models.append(model_name)
            win_rates.append(win_rate)

        # Crear figura
        plt.figure(figsize=(14, 7), dpi=100)

        # Crear barras
        plt.bar(range(len(models)), win_rates, color='skyblue', edgecolor='darkblue')

        # Añadir etiquetas
        plt.xlabel('Modelo')
        plt.ylabel('Tasa de Victoria')
        plt.title(title)

        # Mostrar nombres de modelos rotados para mejor visualización
        plt.xticks(range(len(models)), models, rotation=45, ha='right')
        plt.ylim(0, 1.1)  # Tasa de victoria entre 0 y 1

        # Añadir valores sobre las barras
        for i, v in enumerate(win_rates):
            plt.text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')

        # Guardar figura
        if save_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_filename = f"win_rates_vs_random_{timestamp}"

        save_path = os.path.join(self.output_dir, f"{save_filename}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        logger.info(f"Gráfico guardado en: {save_path}")
        return save_path

    def plot_model_comparison(self,
                             reference_models: List[str] = None,
                             title: str = "Comparación de Modelos",
                             save_filename: str = None) -> str:
        """
        Genera un gráfico comparativo de varios modelos contra diferentes oponentes.

        Args:
            reference_models: Lista de nombres de modelos a comparar (sin ruta)
            title: Título del gráfico
            save_filename: Nombre del archivo para guardar (sin extensión)

        Returns:
            Ruta al archivo de imagen guardado
        """
        if not self.evaluation_results:
            logger.warning("No hay resultados de evaluación para comparar")
            return ""

        # Si no se especifican modelos, usar todos los disponibles
        if not reference_models:
            # Recopilar todos los modelos evaluados
            all_models = set()
            for opponent_key, results in self.evaluation_results.items():
                for model_name in results.keys():
                    all_models.add(model_name)

            reference_models = list(all_models)

        # Recopilar oponentes disponibles
        opponents = [k for k in self.evaluation_results.keys()]

        # Preparar datos para la visualización
        data = {}
        for model_name in reference_models:
            data[model_name] = {}
            for opponent in opponents:
                if opponent in self.evaluation_results and model_name in self.evaluation_results[opponent]:
                    data[model_name][opponent] = self.evaluation_results[opponent][model_name]['win_rate']

        # Eliminar modelos sin datos
        data = {k: v for k, v in data.items() if v}

        if not data:
            logger.warning("No hay datos suficientes para la comparación")
            return ""

        # Crear figura
        plt.figure(figsize=(14, 8), dpi=100)

        # Número de modelos y oponentes
        n_models = len(data)

        # Crear gráfico de radar si hay suficientes oponentes
        if len(opponents) >= 3:
            return self._plot_radar_chart(data, opponents, title, save_filename)
        else:
            # Crear gráfico de barras agrupadas
            return self._plot_grouped_bars(data, opponents, title, save_filename)

    def _plot_radar_chart(self,
                         data: Dict[str, Dict[str, float]],
                         categories: List[str],
                         title: str,
                         save_filename: str = None) -> str:
        """
        Genera un gráfico de radar para comparar modelos.

        Args:
            data: Diccionario de datos por modelo y categoría
            categories: Lista de categorías (oponentes)
            title: Título del gráfico
            save_filename: Nombre del archivo para guardar

        Returns:
            Ruta al archivo guardado
        """
        # Configurar el gráfico de radar
        n_cats = len(categories)
        angles = np.linspace(0, 2*np.pi, n_cats, endpoint=False).tolist()
        angles += angles[:1]  # Cerrar el círculo

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        # Añadir cada modelo al gráfico
        for i, (model_name, model_data) in enumerate(data.items()):
            values = [model_data.get(cat, 0) for cat in categories]
            values += values[:1]  # Cerrar el círculo

            ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.1)

        # Configurar etiquetas y aspecto
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        ax.set_ylim(0, 1)
        plt.title(title, size=15)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        # Guardar figura
        if save_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_filename = f"model_comparison_radar_{timestamp}"

        save_path = os.path.join(self.output_dir, f"{save_filename}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        logger.info(f"Gráfico de radar guardado en: {save_path}")
        return save_path

    def _plot_grouped_bars(self,
                          data: Dict[str, Dict[str, float]],
                          categories: List[str],
                          title: str,
                          save_filename: str = None) -> str:
        """
        Genera un gráfico de barras agrupadas para comparar modelos.

        Args:
            data: Diccionario de datos por modelo y categoría
            categories: Lista de categorías (oponentes)
            title: Título del gráfico
            save_filename: Nombre del archivo para guardar

        Returns:
            Ruta al archivo guardado
        """
        n_models = len(data)
        n_cats = len(categories)

        fig, ax = plt.subplots(figsize=(12, 6), dpi=100)

        # Ancho de las barras
        width = 0.8 / n_models

        # Posición de las barras
        positions = np.arange(n_cats)

        # Añadir barras para cada modelo
        for i, (model_name, model_data) in enumerate(data.items()):
            values = [model_data.get(cat, 0) for cat in categories]
            ax.bar(positions + (i - n_models/2 + 0.5) * width, values,
                   width, label=model_name)

        # Configurar etiquetas y aspecto
        ax.set_xticks(positions)
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Tasa de Victoria')
        ax.set_title(title)
        ax.legend()

        # Guardar figura
        if save_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_filename = f"model_comparison_bars_{timestamp}"

        save_path = os.path.join(self.output_dir, f"{save_filename}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        logger.info(f"Gráfico de barras guardado en: {save_path}")
        return save_path

    def plot_best_models_progress(self,
                                metric: str = "win_rate_vs_random",
                                title: str = "Progreso de los Mejores Modelos",
                                save_filename: str = None) -> str:
        """
        Genera un gráfico mostrando el progreso de los mejores modelos seleccionados.

        Args:
            metric: Métrica para determinar los mejores modelos
            title: Título del gráfico
            save_filename: Nombre del archivo para guardar

        Returns:
            Ruta al archivo guardado
        """
        # Buscar en el directorio de mejores modelos
        best_models_file = os.path.join(os.path.dirname(self.checkpoints_metrics_file),
                                        "../best_models_auto/best_models.json")

        if not os.path.exists(best_models_file):
            logger.warning(f"Archivo de mejores modelos no encontrado: {best_models_file}")
            return ""

        # Cargar datos de mejores modelos
        best_models_data = self._load_json_file(best_models_file)

        if not best_models_data or metric not in best_models_data:
            logger.warning(f"No hay datos para la métrica {metric} en mejores modelos")
            return ""

        # Extraer datos históricos si están disponibles
        historical_data = []

        if 'history' in best_models_data[metric]:
            for entry in best_models_data[metric]['history']:
                if 'epoch' in entry and 'metric_value' in entry:
                    historical_data.append((entry['epoch'], entry['metric_value']))
        else:
            # Si no hay historial, usar el modelo actual
            current = best_models_data[metric]
            if 'epoch' in current and 'metric_value' in current:
                historical_data.append((current['epoch'], current['metric_value']))

        if not historical_data:
            logger.warning("No hay datos históricos para graficar")
            return ""

        # Ordenar por época
        historical_data.sort()
        epochs, values = zip(*historical_data)

        # Crear figura
        plt.figure(figsize=(12, 6), dpi=100)
        plt.plot(epochs, values, 'g-o', linewidth=2, markersize=8)
        plt.title(title)
        plt.xlabel('Época')
        plt.ylabel(metric)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Añadir etiquetas de valores
        for i, (epoch, value) in enumerate(zip(epochs, values)):
            plt.text(epoch, value + 0.01, f"{value:.4f}", ha='center')

        # Guardar figura
        if save_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_filename = f"best_models_progress_{timestamp}"

        save_path = os.path.join(self.output_dir, f"{save_filename}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        logger.info(f"Gráfico de progreso de mejores modelos guardado en: {save_path}")
        return save_path

    def generate_all_visualizations(self) -> List[str]:
        """
        Genera todas las visualizaciones disponibles.

        Returns:
            Lista de rutas a los archivos generados
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        generated_files = []

        # Generar gráficos de métricas por época
        metrics = set()
        for _, data in self.checkpoint_metrics.items():
            if 'metrics' in data:
                metrics.update(data['metrics'].keys())

        for metric in metrics:
            file_path = self.plot_metric_by_epoch(
                metric_name=metric,
                title=f"Progreso de {metric.capitalize()} por Época",
                save_filename=f"{metric}_by_epoch_{timestamp}"
            )
            if file_path:
                generated_files.append(file_path)

        # Generar gráfico de tasas de victoria contra bot aleatorio
        if 'vs_random' in self.evaluation_results:
            file_path = self.plot_win_rates_against_random(
                save_filename=f"win_rates_vs_random_{timestamp}"
            )
            if file_path:
                generated_files.append(file_path)

        # Generar comparación de modelos
        file_path = self.plot_model_comparison(
            save_filename=f"model_comparison_{timestamp}"
        )
        if file_path:
            generated_files.append(file_path)

        # Generar progreso de mejores modelos
        file_path = self.plot_best_models_progress(
            save_filename=f"best_models_progress_{timestamp}"
        )
        if file_path:
            generated_files.append(file_path)

        return generated_files
