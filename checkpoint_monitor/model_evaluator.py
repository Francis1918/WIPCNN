"""
Evaluador de modelos para el entrenamiento de RL.

Este módulo proporciona funcionalidades para evaluar los modelos generados
durante el entrenamiento y seleccionar los mejores.
"""

import os
import torch
import json
import logging
from typing import Dict, Any, Optional, List, Tuple, Union

# Importamos los módulos necesarios del proyecto original sin modificarlos
from bot.CNN_bot import Quarto_bot
from bot.random_bot import Quarto_random_bot
from models.CNN1 import QuartoCNN
from QuartoRL import run_contest

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ModelEvaluator")

class ModelEvaluator:
    """
    Evaluador de modelos para el proyecto de Quarto RL.

    Esta clase proporciona funcionalidades para evaluar modelos contra diferentes
    oponentes y determinar cuáles son los mejores según diferentes métricas.
    """

    def __init__(self,
                evaluation_results_dir: str = "models/evaluation_results",
                num_evaluation_games: int = 10):
        """
        Inicializa el evaluador de modelos.

        Args:
            evaluation_results_dir: Directorio donde se guardarán los resultados de evaluación
            num_evaluation_games: Número de juegos para evaluar cada modelo
        """
        self.evaluation_results_dir = evaluation_results_dir
        self.num_evaluation_games = num_evaluation_games

        # Crear directorio de resultados si no existe
        os.makedirs(evaluation_results_dir, exist_ok=True)

        # Archivo para resultados de evaluación
        self.results_file = os.path.join(evaluation_results_dir, "evaluation_results.json")

        # Cargar resultados previos si existen
        self.evaluation_results = self._load_evaluation_results()

        logger.info(f"ModelEvaluator inicializado. Directorio de resultados: {evaluation_results_dir}")
        logger.info(f"Número de juegos de evaluación: {num_evaluation_games}")

    def _load_evaluation_results(self) -> Dict[str, Dict[str, Any]]:
        """Carga los resultados de evaluación previos."""
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.warning(f"Error al cargar resultados de evaluación: {e}. Creando nuevos resultados.")
                return {}
        return {}

    def _save_evaluation_results(self) -> None:
        """Guarda los resultados de evaluación."""
        with open(self.results_file, 'w') as f:
            json.dump(self.evaluation_results, f, indent=4)

    def load_model(self, model_path: str) -> Optional[Quarto_bot]:
        """
        Carga un modelo desde un archivo checkpoint.

        Args:
            model_path: Ruta al archivo de checkpoint

        Returns:
            Bot de Quarto con el modelo cargado o None si hay error
        """
        try:
            model = QuartoCNN()
            checkpoint = torch.load(model_path, map_location='cpu')

            # Extraer state_dict dependiendo del formato del checkpoint
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Intentar cargar directamente asumiendo que es el state_dict
                model.load_state_dict(checkpoint)

            # Crear bot con temperatura baja para evaluación (menos exploración)
            model_bot = Quarto_bot(model, temperature=0.1)
            model_bot.DETERMINISTIC = True  # Usar política determinista para evaluación

            logger.info(f"Modelo cargado correctamente desde: {model_path}")
            return model_bot

        except Exception as e:
            logger.error(f"Error al cargar el modelo {model_path}: {e}")
            return None

    def evaluate_against_random(self,
                              model_path: str,
                              num_games: int = None) -> Dict[str, Any]:
        """
        Evalúa un modelo contra un bot aleatorio.

        Args:
            model_path: Ruta al archivo de checkpoint
            num_games: Número de juegos para la evaluación (usa el valor por defecto si es None)

        Returns:
            Diccionario con resultados de la evaluación
        """
        if num_games is None:
            num_games = self.num_evaluation_games

        # Cargar el modelo
        model_bot = self.load_model(model_path)
        if not model_bot:
            return {'error': 'No se pudo cargar el modelo'}

        # Crear oponente aleatorio
        random_bot = Quarto_random_bot()

        # Ejecutar torneo
        results = run_contest(
            player=model_bot,
            rival=random_bot,
            matches=num_games,
            verbose=False,
            match_dir=os.path.join(self.evaluation_results_dir, "vs_random"),
        )

        # Calcular tasa de victorias
        total_games = results['wins'] + results['draws'] + results['losses']
        win_rate = (results['wins'] + 0.5 * results['draws']) / total_games if total_games > 0 else 0

        # Guardar resultados
        model_name = os.path.basename(model_path)
        evaluation_result = {
            'model_path': model_path,
            'opponent': 'random_bot',
            'num_games': num_games,
            'wins': results['wins'],
            'draws': results['draws'],
            'losses': results['losses'],
            'win_rate': win_rate
        }

        # Añadir a historial
        if 'vs_random' not in self.evaluation_results:
            self.evaluation_results['vs_random'] = {}

        self.evaluation_results['vs_random'][model_name] = evaluation_result
        self._save_evaluation_results()

        logger.info(f"Evaluación contra bot aleatorio: {model_name}, win_rate={win_rate:.4f}")
        return evaluation_result

    def evaluate_against_checkpoint(self,
                                  model_path: str,
                                  opponent_path: str,
                                  num_games: int = None) -> Dict[str, Any]:
        """
        Evalúa un modelo contra otro modelo de checkpoint.

        Args:
            model_path: Ruta al archivo de checkpoint a evaluar
            opponent_path: Ruta al archivo de checkpoint del oponente
            num_games: Número de juegos para la evaluación

        Returns:
            Diccionario con resultados de la evaluación
        """
        if num_games is None:
            num_games = self.num_evaluation_games

        # Cargar el modelo principal
        model_bot = self.load_model(model_path)
        if not model_bot:
            return {'error': 'No se pudo cargar el modelo principal'}

        # Cargar el modelo oponente
        opponent_bot = self.load_model(opponent_path)
        if not opponent_bot:
            return {'error': 'No se pudo cargar el modelo oponente'}

        # Nombres para identificación
        model_name = os.path.basename(model_path)
        opponent_name = os.path.basename(opponent_path)

        # Ejecutar torneo
        results = run_contest(
            player=model_bot,
            rival=opponent_bot,
            matches=num_games,
            verbose=False,
            match_dir=os.path.join(self.evaluation_results_dir, f"{model_name}_vs_{opponent_name}"),
        )

        # Calcular tasa de victorias
        total_games = results['wins'] + results['draws'] + results['losses']
        win_rate = (results['wins'] + 0.5 * results['draws']) / total_games if total_games > 0 else 0

        # Guardar resultados
        evaluation_result = {
            'model_path': model_path,
            'opponent_path': opponent_path,
            'opponent': opponent_name,
            'num_games': num_games,
            'wins': results['wins'],
            'draws': results['draws'],
            'losses': results['losses'],
            'win_rate': win_rate
        }

        # Añadir a historial
        key = f"vs_{opponent_name}"
        if key not in self.evaluation_results:
            self.evaluation_results[key] = {}

        self.evaluation_results[key][model_name] = evaluation_result
        self._save_evaluation_results()

        logger.info(f"Evaluación {model_name} vs {opponent_name}: win_rate={win_rate:.4f}")
        return evaluation_result

    def evaluate_against_multiple(self,
                                model_path: str,
                                opponent_paths: List[str],
                                num_games: int = None) -> Dict[str, Dict[str, Any]]:
        """
        Evalúa un modelo contra múltiples oponentes.

        Args:
            model_path: Ruta al archivo de checkpoint a evaluar
            opponent_paths: Lista de rutas a checkpoints oponentes
            num_games: Número de juegos para cada evaluación

        Returns:
            Diccionario con resultados de todas las evaluaciones
        """
        results = {}

        # Evaluar contra bot aleatorio primero
        results['random'] = self.evaluate_against_random(model_path, num_games)

        # Evaluar contra cada oponente
        for opponent_path in opponent_paths:
            opponent_name = os.path.basename(opponent_path)
            results[opponent_name] = self.evaluate_against_checkpoint(
                model_path, opponent_path, num_games
            )

        return results

    def get_best_model(self,
                     models: List[str],
                     metric: str = "win_rate_vs_random",
                     higher_is_better: bool = True) -> Optional[Tuple[str, float]]:
        """
        Obtiene el mejor modelo según una métrica específica.

        Args:
            models: Lista de rutas a modelos para evaluar
            metric: Métrica para comparar (win_rate_vs_random, win_rate_vs_latest, etc.)
            higher_is_better: True si valores más altos son mejores

        Returns:
            Tupla con (ruta al mejor modelo, valor de la métrica) o None
        """
        if not models:
            logger.warning("No hay modelos para evaluar")
            return None

        best_model = None
        best_value = float('-inf') if higher_is_better else float('inf')

        for model_path in models:
            model_name = os.path.basename(model_path)

            # Determinar qué métrica usar
            if metric == "win_rate_vs_random" and 'vs_random' in self.evaluation_results:
                if model_name in self.evaluation_results['vs_random']:
                    value = self.evaluation_results['vs_random'][model_name]['win_rate']
                else:
                    # Evaluar modelo si no se ha evaluado antes
                    result = self.evaluate_against_random(model_path)
                    value = result['win_rate']
            elif metric.startswith("win_rate_vs_") and metric[12:] in self.evaluation_results:
                opponent_key = metric[12:]
                if model_name in self.evaluation_results[f"vs_{opponent_key}"]:
                    value = self.evaluation_results[f"vs_{opponent_key}"][model_name]['win_rate']
                else:
                    # No podemos evaluar sin el oponente específico
                    logger.warning(f"No hay evaluación contra {opponent_key} para {model_name}")
                    continue
            else:
                logger.warning(f"Métrica {metric} no disponible para {model_name}")
                continue

            # Actualizar mejor modelo si este es mejor
            is_better = (value > best_value) if higher_is_better else (value < best_value)
            if is_better:
                best_value = value
                best_model = model_path

        if best_model:
            logger.info(f"Mejor modelo según {metric}: {os.path.basename(best_model)}, valor={best_value:.4f}")
            return best_model, best_value

        logger.warning(f"No se pudo determinar el mejor modelo según {metric}")
        return None
