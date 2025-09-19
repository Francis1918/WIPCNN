"""
Evaluador de modelos para el entrenamiento de RL.

Este módulo proporciona funcionalidades para evaluar los modelos generados
durante el entrenamiento y seleccionar los mejores.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Tuple

import torch

# Importamos los módulos necesarios del proyecto original sin modificarlos
from bot.CNN_bot import Quarto_bot
from bot.random_bot import Quarto_random_bot
from models.CNN1 import QuartoCNN
from QuartoRL.contest import run_contest

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

    def __init__(
        self,
        evaluation_results_dir: str = "models/evaluation_results",
        num_evaluation_games: int = 10,
    ):
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
        self.evaluation_results: Dict[str, Dict[str, Any]] = self._load_evaluation_results()

        logger.info(f"ModelEvaluator inicializado. Directorio de resultados: {evaluation_results_dir}")
        logger.info(f"Número de juegos de evaluación: {num_evaluation_games}")

    # -------------------------
    # Utilidades de persistencia
    # -------------------------
    def _load_evaluation_results(self) -> Dict[str, Dict[str, Any]]:
        """Carga los resultados de evaluación previos."""
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
                    logger.warning("El archivo de resultados no tiene formato dict. Se re-creará.")
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.warning(f"Error al cargar resultados de evaluación: {e}. Creando nuevos resultados.")
        return {}

    def _save_evaluation_results(self) -> None:
        """Guarda los resultados de evaluación."""
        try:
            with open(self.results_file, 'w') as f:
                json.dump(self.evaluation_results, f, indent=4)
        except Exception as e:
            logger.error(f"No se pudieron guardar los resultados de evaluación: {e}")

    # -------------------------
    # Carga de modelos
    # -------------------------
    def load_model(self, model_path: str) -> Optional[Quarto_bot]:
        """
        Carga un modelo desde un archivo checkpoint.

        Args:
            model_path: Ruta al archivo de checkpoint

        Returns:
            Bot de Quarto con el modelo cargado o None si hay error
        """
        try:
            if not os.path.exists(model_path):
                logger.error(f"El checkpoint no existe: {model_path}")
                return None

            model = QuartoCNN()
            checkpoint = torch.load(model_path, map_location='cpu')

            # Extraer state_dict dependiendo del formato del checkpoint
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict)
            model.eval()  # muy importante en evaluación

            # Crear bot para evaluación
            model_bot = Quarto_bot(model=model)
            # Forzar política determinista si el bot lo soporta
            if hasattr(model_bot, "DETERMINISTIC"):
                setattr(model_bot, "DETERMINISTIC", True)

            logger.info(f"Modelo cargado correctamente desde: {model_path}")
            return model_bot

        except Exception as e:
            logger.error(f"Error al cargar el modelo {model_path}: {e}")
            return None

    # -------------------------
    # Evaluaciones
    # -------------------------
    def evaluate_against_random(self, model_path: str, num_games: Optional[int] = None) -> Dict[str, Any]:
        """
        Evalúa un modelo contra un bot aleatorio (formato Opción A: rivales como objetos y
        resultados en dict por oponente).
        """
        if num_games is None:
            num_games = self.num_evaluation_games

        try:
            # Cargar el modelo
            model_bot = self.load_model(model_path)
            if model_bot is None:
                return {'error': 'No se pudo cargar el modelo', 'win_rate': 0.0, 'wins': 0, 'draws': 0, 'losses': 0}

            # Ejecutar torneo contra bot aleatorio (sin wrappers; pasar objetos)
            results = run_contest(
                player=model_bot,
                rivals={"random_bot": Quarto_random_bot()},
                matches=num_games,
                verbose=False,
                match_dir=os.path.join(self.evaluation_results_dir, "vs_random"),
            )

            # Extraer resultados contra el rival aleatorio (asumimos dict por oponente)
            random_results = results["random_bot"]

            # Calcular tasa de victorias
            wins = int(random_results.get('wins', 0))
            draws = int(random_results.get('draws', 0))
            losses = int(random_results.get('losses', 0))
            total_games = wins + draws + losses
            win_rate = (wins + 0.5 * draws) / total_games if total_games > 0 else 0.0

            # Guardar resultados
            model_name = os.path.basename(model_path)
            evaluation_result = {
                'model_path': model_path,
                'opponent': 'random_bot',
                'num_games': num_games,
                'wins': wins,
                'draws': draws,
                'losses': losses,
                'win_rate': win_rate,
            }

            # Añadir a historial
            self.evaluation_results.setdefault('vs_random', {})
            self.evaluation_results['vs_random'][model_name] = evaluation_result
            self._save_evaluation_results()

            logger.info(f"Evaluación contra bot aleatorio: {model_name}, win_rate={win_rate:.4f}")
            return evaluation_result

        except Exception as e:
            logger.error(f"Error al evaluar el modelo {model_path} vs random: {e}")
            return {'error': str(e), 'win_rate': 0.0, 'wins': 0, 'draws': 0, 'losses': 0}

    def evaluate_against_checkpoint(
        self,
        model_path: str,
        opponent_path: str,
        num_games: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evalúa un modelo contra otro modelo de checkpoint (formato Opción A: rivales como objetos y
        resultados en dict por oponente).
        """
        if num_games is None:
            num_games = self.num_evaluation_games

        try:
            # Cargar el modelo principal
            model_bot = self.load_model(model_path)
            if model_bot is None:
                return {'error': 'No se pudo cargar el modelo principal', 'win_rate': 0.0, 'wins': 0, 'draws': 0, 'losses': 0}

            # Cargar el modelo oponente
            opponent_bot = self.load_model(opponent_path)
            if opponent_bot is None:
                return {'error': 'No se pudo cargar el modelo oponente', 'win_rate': 0.0, 'wins': 0, 'draws': 0, 'losses': 0}

            # Nombres para identificación
            model_name = os.path.basename(model_path)
            opponent_name = os.path.basename(opponent_path)

            # Ejecutar torneo (pasar objetos directamente)
            results = run_contest(
                player=model_bot,
                rivals={opponent_name: opponent_bot},
                matches=num_games,
                verbose=False,
                match_dir=os.path.join(self.evaluation_results_dir, f"{model_name}_vs_{opponent_name}"),
            )

            # Extraer resultados
            match_results = results[opponent_name]

            # Calcular tasa de victorias
            wins = int(match_results.get('wins', 0))
            draws = int(match_results.get('draws', 0))
            losses = int(match_results.get('losses', 0))
            total_games = wins + draws + losses
            win_rate = (wins + 0.5 * draws) / total_games if total_games > 0 else 0.0

            # Guardar resultados
            evaluation_result = {
                'model_path': model_path,
                'opponent_path': opponent_path,
                'opponent': opponent_name,
                'num_games': num_games,
                'wins': wins,
                'draws': draws,
                'losses': losses,
                'win_rate': win_rate
            }

            # Añadir a historial
            key = f"vs_{opponent_name}"
            self.evaluation_results.setdefault(key, {})
            self.evaluation_results[key][model_name] = evaluation_result
            self._save_evaluation_results()

            logger.info(f"Evaluación {model_name} vs {opponent_name}: win_rate={win_rate:.4f}")
            return evaluation_result

        except Exception as e:
            logger.error(f"Error al evaluar modelo vs checkpoint: {e}")
            return {'error': str(e), 'win_rate': 0.0, 'wins': 0, 'draws': 0, 'losses': 0}

    def evaluate_against_multiple(
        self,
        model_path: str,
        opponent_paths: List[str],
        num_games: Optional[int] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evalúa un modelo contra múltiples oponentes.
        """
        results: Dict[str, Dict[str, Any]] = {}

        # Evaluar contra bot aleatorio primero
        results['random'] = self.evaluate_against_random(model_path, num_games)

        # Evaluar contra cada oponente
        for opponent_path in opponent_paths:
            opponent_name = os.path.basename(opponent_path)
            results[opponent_name] = self.evaluate_against_checkpoint(
                model_path, opponent_path, num_games
            )

        return results

    # -------------------------
    # Selección de mejores modelos
    # -------------------------
    def get_best_model(
        self,
        models: List[str],
        metric: str = "win_rate_vs_random",
        higher_is_better: bool = True
    ) -> Optional[Tuple[str, float]]:
        """
        Obtiene el mejor modelo según una métrica específica.

        Args:
            models: Lista de rutas a modelos para evaluar
            metric: Métrica para comparar (win_rate_vs_random por defecto)
            higher_is_better: True si valores más altos son mejores

        Returns:
            Tupla con (ruta al mejor modelo, valor de la métrica) o None
        """
        if not models:
            logger.warning("No hay modelos para evaluar")
            return None

        best_model: Optional[str] = None
        best_value: float = float('-inf') if higher_is_better else float('inf')

        for model_path in models:
            model_name = os.path.basename(model_path)

            # Para Opción A, priorizamos vs_random. Si no existe, lo calculamos.
            value: Optional[float] = None
            if metric == "win_rate_vs_random":
                if 'vs_random' in self.evaluation_results and model_name in self.evaluation_results['vs_random']:
                    value = float(self.evaluation_results['vs_random'][model_name].get('win_rate', 0.0))
                else:
                    result = self.evaluate_against_random(model_path)
                    value = float(result.get('win_rate', 0.0))
            else:
                logger.warning(f"Métrica {metric} no soportada en configuración Opción A (solo 'win_rate_vs_random').")
                continue

            if value is None:
                continue

            is_better = (value > best_value) if higher_is_better else (value < best_value)
            if is_better:
                best_value = value
                best_model = model_path

        if best_model is not None:
            logger.info(f"Mejor modelo según {metric}: {os.path.basename(best_model)}, valor={best_value:.4f}")
            return best_model, best_value

        logger.warning(f"No se pudo determinar el mejor modelo según {metric}")
        return None
