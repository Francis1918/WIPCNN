#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
compare_agents_external.py - Herramienta para comparar agentes de diferentes épocas y arquitecturas en el juego Quarto.

Esta herramienta permite cargar agentes desde una ruta externa específica y comparar su rendimiento
en partidas de Quarto. Los resultados se guardan en una carpeta designada.

Uso:
    python compare_agents_external.py [--agent1 RUTA_AGENTE1] [--agent2 RUTA_AGENTE2]
                                     [--matches N] [--temp T] [--output RUTA_SALIDA]

Ejemplos:
    python compare_agents_external.py --agent1 "ruta/al/agente1.pt" --agent2 "ruta/al/agente2.pt" --matches 100
    python compare_agents_external.py --agent1 "ruta/al/agente1.pt" --agent2 "ruta/al/agente2.pt" --temp 0.1 --output "ruta/resultados"
"""

import argparse
import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path
import json
import inspect

# Configurar la ruta base del proyecto
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

# Cargar variables de entorno desde .env
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    try:
        from dotenv import load_dotenv

        load_dotenv(env_file)
        print(f"✅ Cargado archivo .env desde: {env_file}")
    except ImportError:
        # Si no tiene python-dotenv, instalar automáticamente
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        from dotenv import load_dotenv

        load_dotenv(env_file)
        print(f"✅ Instalado python-dotenv y cargado .env desde: {env_file}")
else:
    print(f"⚠️ No se encontró el archivo .env en: {env_file}")

# Importar utilidades del proyecto
from utils.logger import logger

# Intentar importar las dependencias necesarias
try:
    from quartopy import play_games
except ImportError:
    # Fallback para dependencias
    from utils import setup_quartopy
    setup_quartopy.setup(silent=False)
    from quartopy import play_games

# Rutas predeterminadas para agentes y resultados (desde .env o valores por defecto)
DEFAULT_AGENTS_PATH = os.getenv("AGENTS_PATH",
                                "C:/Users/bravo/Documents/Metodos Numericos Pycharm/Mech Interp/Comparacion entre agentes/Agentes")
DEFAULT_RESULTS_PATH = os.getenv("RESULTS_PATH",
                                 "C:/Users/bravo/Documents/Metodos Numericos Pycharm/Mech Interp/Comparacion entre agentes/Resultados")

# Importar el bot flexible que puede manejar cualquier arquitectura
try:
    from bot.FlexibleBot import FlexibleBot
except ImportError:
    logger.warning("No se pudo importar FlexibleBot. Intentando importar desde ruta alternativa...")
    # Intentar importar desde la ruta actual
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bot_dir = os.path.join(current_dir, "bot")
    if bot_dir not in sys.path:
        sys.path.append(bot_dir)
    try:
        from FlexibleBot import FlexibleBot

        logger.info("FlexibleBot importado desde ruta alternativa")
    except ImportError:
        logger.error("No se pudo importar FlexibleBot. Asegúrate de que el archivo esté en la ruta correcta.")
        raise


def load_agent_from_path(agent_path, agent_type="CNN", temperature=0.1, deterministic=False):
    """
    Carga un agente desde una ruta específica, utilizando el bot flexible que puede manejar cualquier arquitectura.

    Args:
        agent_path (str): Ruta al archivo del modelo
        agent_type (str): Tipo de agente (no se utiliza con FlexibleBot, pero se mantiene por compatibilidad)
        temperature (float): Temperatura para la toma de decisiones
        deterministic (bool): Si es True, el agente tomará decisiones deterministas

    Returns:
        object: Instancia del agente cargado
    """
    logger.info(f"Cargando agente desde: {agent_path}")

    # Verificar que el archivo existe
    if not os.path.exists(agent_path):
        raise FileNotFoundError(f"No se encontró el archivo del modelo: {agent_path}")

    # Añadir la ruta de agentes al path para poder importar módulos desde allí
    agents_dir = os.path.dirname(agent_path)
    if agents_dir not in sys.path:
        sys.path.append(agents_dir)

    # Añadir también el directorio predeterminado de agentes
    if DEFAULT_AGENTS_PATH not in sys.path:
        sys.path.append(DEFAULT_AGENTS_PATH)

    try:
        # Usar FlexibleBot que puede manejar cualquier arquitectura
        bot = FlexibleBot(model_path=agent_path)

        # Configurar los atributos de temperatura manualmente
        bot.DETERMINISTIC = deterministic
        bot.TEMPERATURE = temperature
        logger.info(
            f"Bot cargado correctamente desde {agent_path} con temperatura: {temperature}, deterministic: {deterministic}")

        return bot

    except Exception as e:
        logger.error(f"Error al cargar el agente: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def compare_agents(agent1_path, agent2_path, agent1_type="CNN", agent2_type="CNN",
                   agent1_name=None, agent2_name=None, n_matches=500,
                   temperature=0.1, deterministic=False, output_dir=None):
    """
    Compara dos agentes cargados desde rutas específicas.

    Args:
        agent1_path (str): Ruta al archivo del primer agente
        agent2_path (str): Ruta al archivo del segundo agente
        agent1_type (str): Tipo del primer agente (cualquier arquitectura soportada)
        agent2_type (str): Tipo del segundo agente (cualquier arquitectura soportada)
        agent1_name (str): Nombre para identificar al primer agente
        agent2_name (str): Nombre para identificar al segundo agente
        n_matches (int): Número de partidas a jugar
        temperature (float): Temperatura para la toma de decisiones
        deterministic (bool): Si es True, los agentes tomarán decisiones deterministas
        output_dir (str): Directorio donde guardar los resultados

    Returns:
        dict: Resultados de la comparación
    """
    # Establecer nombres de agentes si no se proporcionan
    if agent1_name is None:
        agent1_name = f"Agent1_{Path(agent1_path).stem}"
    if agent2_name is None:
        agent2_name = f"Agent2_{Path(agent2_path).stem}"

    logger.info(f"Comparando {agent1_name} ({agent1_type}) vs {agent2_name} ({agent2_type})")
    logger.info(f"Número de partidas: {n_matches}")
    logger.info(f"Temperatura: {temperature}, Determinista: {deterministic}")

    try:
        # Cargar los agentes
        agent1 = load_agent_from_path(agent1_path, agent1_type, temperature, deterministic)
        agent2 = load_agent_from_path(agent2_path, agent2_type, temperature, deterministic)

        logger.info("Agentes cargados correctamente. Comenzando enfrentamiento...")

        # Inspeccionar la firma de play_games para saber qué parámetros acepta
        sig = inspect.signature(play_games)
        logger.info(f"Parámetros de play_games: {sig}")

        # Crear directorio para resultados si no existe
        if output_dir is None:
            output_dir = DEFAULT_RESULTS_PATH

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Enfrentar los agentes (A vs B)
        logger.info(f"Jugando {n_matches} partidas: {agent1_name} (P1) vs {agent2_name} (P2)...")
        try:
            # Intentar primero con parámetros básicos
            result_ab = play_games(
                matches=n_matches,
                player1=agent1,
                player2=agent2,
                verbose=False
            )
            # Verificar qué tipo de resultado retorna
            logger.info(f"Tipo de resultado de play_games: {type(result_ab)}")
            logger.info(f"Contenido del resultado: {result_ab}")

            # Manejar diferentes tipos de retorno
            if isinstance(result_ab, tuple):
                if len(result_ab) == 2:
                    results_ab, win_rate_ab = result_ab
                elif len(result_ab) == 1:
                    results_ab = result_ab[0]
                    win_rate_ab = None
                else:
                    logger.warning(f"play_games retornó una tupla con {len(result_ab)} elementos")
                    results_ab = result_ab
                    win_rate_ab = None
            else:
                # Si no es tupla, asumir que es solo el diccionario de resultados
                results_ab = result_ab
                win_rate_ab = None

        except TypeError as e:
            logger.warning(f"Error con parámetros nombrados: {e}. Intentando con parámetros posicionales...")
            try:
                # Intentar con solo los parámetros obligatorios posicionales
                result_ab = play_games(n_matches, agent1, agent2)

                if isinstance(result_ab, tuple):
                    if len(result_ab) == 2:
                        results_ab, win_rate_ab = result_ab
                    else:
                        results_ab = result_ab[0] if len(result_ab) > 0 else result_ab
                        win_rate_ab = None
                else:
                    results_ab = result_ab
                    win_rate_ab = None

            except Exception as e2:
                logger.error(f"Error crítico al jugar partidas A vs B: {e2}")
                import traceback
                logger.error(traceback.format_exc())
                raise

        # Enfrentar los agentes (B vs A) para equilibrar la ventaja de jugar primero
        logger.info(f"Jugando {n_matches} partidas: {agent2_name} (P1) vs {agent1_name} (P2)...")
        try:
            # Intentar primero con parámetros básicos
            result_ba = play_games(
                matches=n_matches,
                player1=agent2,
                player2=agent1,
                verbose=False
            )

            # Manejar diferentes tipos de retorno
            if isinstance(result_ba, tuple):
                if len(result_ba) == 2:
                    results_ba, win_rate_ba = result_ba
                elif len(result_ba) == 1:
                    results_ba = result_ba[0]
                    win_rate_ba = None
                else:
                    results_ba = result_ba
                    win_rate_ba = None
            else:
                results_ba = result_ba
                win_rate_ba = None

        except TypeError as e:
            logger.warning(f"Error con parámetros nombrados: {e}. Intentando con parámetros posicionales...")
            try:
                # Intentar con solo los parámetros obligatorios posicionales
                result_ba = play_games(n_matches, agent2, agent1)

                if isinstance(result_ba, tuple):
                    if len(result_ba) == 2:
                        results_ba, win_rate_ba = result_ba
                    else:
                        results_ba = result_ba[0] if len(result_ba) > 0 else result_ba
                        win_rate_ba = None
                else:
                    results_ba = result_ba
                    win_rate_ba = None

            except Exception as e2:
                logger.error(f"Error crítico al jugar partidas B vs A: {e2}")
                import traceback
                logger.error(traceback.format_exc())
                raise

        # Mostrar resultados
        logger.info("\nResultados:")
        logger.info(f"{agent1_name} (P1) vs {agent2_name} (P2) en {n_matches} partidas:")
        if win_rate_ab is not None:
            logger.info(f"Tasa de victoria: {win_rate_ab}")
        logger.info(f"Resultados detallados: {results_ab}")
        logger.info(f"Tipo de results_ab: {type(results_ab)}")
        if isinstance(results_ab, dict):
            logger.info(f"Claves en results_ab: {list(results_ab.keys())[:10]}")  # Mostrar primeras 10 claves

        logger.info(f"{agent2_name} (P1) vs {agent1_name} (P2) en {n_matches} partidas:")
        if win_rate_ba is not None:
            logger.info(f"Tasa de victoria: {win_rate_ba}")
        logger.info(f"Resultados detallados: {results_ba}")

        # Calcular estadísticas combinadas - CORREGIDO
        # Contar victorias correctamente
        wins_agent1_total = 0
        wins_agent2_total = 0
        draws_total = 0

        # Contar resultados de la primera ronda (agent1 como P1, agent2 como P2)
        # Si results_ab es un diccionario con claves como 'P1', 'P2', 'Empates'
        if isinstance(results_ab, dict):
            # Verificar si tiene el formato de conteo directo
            if 'P1' in results_ab or 'P2' in results_ab or 'Empates' in results_ab:
                wins_agent1_total += results_ab.get('P1', 0)
                wins_agent2_total += results_ab.get('P2', 0)
                draws_total += results_ab.get('Empates', 0)
            else:
                # Si es un diccionario de resultados individuales (game_id: resultado)
                for result in results_ab.values():
                    if result == 1:
                        wins_agent1_total += 1
                    elif result == -1:
                        wins_agent2_total += 1
                    elif result == 0:
                        draws_total += 1

        # Contar resultados de la segunda ronda (agent2 como P1, agent1 como P2)
        # IMPORTANTE: Invertir los resultados porque los jugadores cambiaron de posición
        if isinstance(results_ba, dict):
            # Verificar si tiene el formato de conteo directo
            if 'P1' in results_ba or 'P2' in results_ba or 'Empates' in results_ba:
                # P1 en esta ronda es agent2, P2 es agent1
                wins_agent2_total += results_ba.get('P1', 0)
                wins_agent1_total += results_ba.get('P2', 0)
                draws_total += results_ba.get('Empates', 0)
            else:
                # Si es un diccionario de resultados individuales (game_id: resultado)
                for result in results_ba.values():
                    if result == 1:
                        wins_agent2_total += 1  # P1 ganó, y P1 es agent2
                    elif result == -1:
                        wins_agent1_total += 1  # P2 ganó, y P2 es agent1
                    elif result == 0:
                        draws_total += 1

        total_matches = n_matches * 2

        # Evitar división por cero
        if total_matches > 0:
            win_rate_agent1 = wins_agent1_total / total_matches * 100
            win_rate_agent2 = wins_agent2_total / total_matches * 100
            draw_rate = draws_total / total_matches * 100
        else:
            win_rate_agent1 = 0
            win_rate_agent2 = 0
            draw_rate = 0

        logger.info("\nEstadísticas combinadas:")
        logger.info(f"Victorias de {agent1_name}: {wins_agent1_total} ({win_rate_agent1:.2f}%)")
        logger.info(f"Victorias de {agent2_name}: {wins_agent2_total} ({win_rate_agent2:.2f}%)")
        logger.info(f"Empates: {draws_total} ({draw_rate:.2f}%)")

        # Usar las variables corregidas para el resto del código
        total_wins_agent1 = wins_agent1_total
        total_wins_agent2 = wins_agent2_total
        total_draws = draws_total

        # Crear visualización solo si hay datos válidos
        if total_wins_agent1 > 0 or total_wins_agent2 > 0 or total_draws > 0:
            labels = [agent1_name, agent2_name, 'Empates']
            sizes = [total_wins_agent1, total_wins_agent2, total_draws]
            colors = ['#ff9999', '#66b3ff', '#c2c2f0']

            plt.figure(figsize=(10, 7))
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title(f'Comparación de rendimiento: {agent1_name} vs {agent2_name}')

            # Guardar gráfico
            comparison_file = os.path.join(output_dir, f"comparison_{agent1_name}_vs_{agent2_name}_{timestamp}.png")
            plt.savefig(comparison_file)
            plt.close()

            logger.info(f"\nGráfico guardado como {comparison_file}")
        else:
            logger.warning("No hay datos válidos para crear el gráfico de pastel. Todos los valores son cero.")
            logger.warning(
                "Esto puede indicar que play_games() no está retornando los resultados en el formato esperado.")

        # Guardar resultados en CSV
        results_df = pd.DataFrame({
            'agent1_name': [agent1_name],
            'agent2_name': [agent2_name],
            'agent1_type': [agent1_type],
            'agent2_type': [agent2_type],
            'agent1_path': [agent1_path],
            'agent2_path': [agent2_path],
            'wins_agent1': [total_wins_agent1],
            'wins_agent2': [total_wins_agent2],
            'draws': [total_draws],
            'win_rate_agent1': [win_rate_agent1],
            'win_rate_agent2': [win_rate_agent2],
            'draw_rate': [draw_rate],
            'n_matches': [total_matches],
            'temperature': [temperature],
            'deterministic': [deterministic],
            'timestamp': [timestamp]
        })

        csv_file = os.path.join(output_dir, "comparison_results.csv")
        # Añadir al archivo existente o crear uno nuevo
        if os.path.exists(csv_file):
            results_df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            results_df.to_csv(csv_file, index=False)

        logger.info(f"Resultados guardados en {csv_file}")

        # Guardar resultados detallados en JSON
        detailed_results = {
            'agent1': {
                'name': agent1_name,
                'type': agent1_type,
                'path': agent1_path,
                'wins_as_p1': results_ab.get('P1', 0),
                'wins_as_p2': results_ba.get('P2', 0),
                'total_wins': total_wins_agent1,
                'win_rate': win_rate_agent1
            },
            'agent2': {
                'name': agent2_name,
                'type': agent2_type,
                'path': agent2_path,
                'wins_as_p1': results_ba.get('P1', 0),
                'wins_as_p2': results_ab.get('P2', 0),
                'total_wins': total_wins_agent2,
                'win_rate': win_rate_agent2
            },
            'draws': {
                'as_ab': results_ab.get('Empates', 0),
                'as_ba': results_ba.get('Empates', 0),
                'total': total_draws,
                'rate': draw_rate
            },
            'config': {
                'n_matches_per_side': n_matches,
                'total_matches': total_matches,
                'temperature': temperature,
                'deterministic': deterministic,
                'timestamp': timestamp
            }
        }

        json_file = os.path.join(output_dir, f"detailed_results_{agent1_name}_vs_{agent2_name}_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(detailed_results, f, indent=4)

        logger.info(f"Resultados detallados guardados en {json_file}")

        return {
            'agent1': {
                'name': agent1_name,
                'wins': total_wins_agent1,
                'win_rate': win_rate_agent1
            },
            'agent2': {
                'name': agent2_name,
                'wins': total_wins_agent2,
                'win_rate': win_rate_agent2
            },
            'draws': total_draws,
            'draw_rate': draw_rate,
            'total_matches': total_matches
        }

    except Exception as e:
        logger.error(f"Error durante la comparación: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def list_available_agents(agents_dir=None):
    """
    Lista los agentes disponibles en el directorio especificado.

    Args:
        agents_dir (str): Directorio donde buscar los agentes

    Returns:
        list: Lista de rutas a los agentes encontrados
    """
    if agents_dir is None:
        agents_dir = DEFAULT_AGENTS_PATH

    if not os.path.exists(agents_dir):
        logger.warning(f"El directorio de agentes no existe: {agents_dir}")
        return []

    # Buscar archivos .pt (modelos PyTorch)
    agent_files = list(Path(agents_dir).glob("**/*.pt"))

    if not agent_files:
        logger.warning(f"No se encontraron archivos de modelo en {agents_dir}")
        return []

    logger.info(f"Agentes disponibles en {agents_dir}:")
    for i, agent_file in enumerate(agent_files, 1):
        logger.info(f"{i}. {agent_file.relative_to(agents_dir)}")

    return [str(f) for f in agent_files]


def interactive_mode():
    """Modo interactivo para comparar agentes."""
    print("\n===== Comparador de Agentes para Quarto =====")
    print("Este programa permite enfrentar agentes de diferentes épocas y arquitecturas para evaluar su rendimiento.")

    # Listar agentes disponibles
    agents_dir = input(f"\nDirectorio de agentes [{DEFAULT_AGENTS_PATH}]: ") or DEFAULT_AGENTS_PATH
    available_agents = list_available_agents(agents_dir)

    if not available_agents:
        print(f"No se encontraron agentes en {agents_dir}. Por favor, verifica la ruta.")
        return

    # Seleccionar primer agente
    while True:
        try:
            agent1_idx = int(input("\nSeleccione el número del primer agente: ")) - 1
            if 0 <= agent1_idx < len(available_agents):
                agent1_path = available_agents[agent1_idx]
                break
            else:
                print(f"Error: Por favor ingrese un número entre 1 y {len(available_agents)}.")
        except ValueError:
            print("Error: Por favor ingrese un número entero válido.")

    # Seleccionar segundo agente
    while True:
        try:
            agent2_idx = int(input("\nSeleccione el número del segundo agente: ")) - 1
            if 0 <= agent2_idx < len(available_agents):
                agent2_path = available_agents[agent2_idx]
                break
            else:
                print(f"Error: Por favor ingrese un número entre 1 y {len(available_agents)}.")
        except ValueError:
            print("Error: Por favor ingrese un número entero válido.")

    # Tipo de agentes
    agent1_type = input("\nTipo del primer agente [CNN]: ") or "CNN"
    agent2_type = input("Tipo del segundo agente [CNN]: ") or "CNN"

    # Nombres personalizados
    agent1_name = input(f"\nNombre para identificar al primer agente [{Path(agent1_path).stem}]: ") or Path(
        agent1_path).stem
    agent2_name = input(f"Nombre para identificar al segundo agente [{Path(agent2_path).stem}]: ") or Path(
        agent2_path).stem

    # Número de partidas
    while True:
        try:
            n_matches = int(input("\nNúmero de partidas por lado [500]: ") or "500")
            if n_matches <= 0:
                print("Error: El número de partidas debe ser mayor que cero.")
                continue
            break
        except ValueError:
            print("Error: Por favor ingrese un número entero válido.")

    # Temperatura
    while True:
        try:
            temp_input = input("\nTemperatura para los agentes (0.1-1.0) [0.1]: ") or "0.1"
            temperature = float(temp_input)
            if temperature <= 0 or temperature > 1:
                print("Error: La temperatura debe estar entre 0.1 y 1.0.")
                continue
            break
        except ValueError:
            print("Error: Por favor ingrese un número decimal válido.")

    # Modo determinista
    deterministic_input = input("\n¿Usar modo determinista? (s/n) [n]: ").lower() or "n"
    deterministic = deterministic_input in ["s", "si", "sí", "y", "yes"]

    # Directorio de salida
    output_dir = input(f"\nDirectorio para guardar resultados [{DEFAULT_RESULTS_PATH}]: ") or DEFAULT_RESULTS_PATH

    # Resumen de parámetros
    print("\n===== Parámetros de la comparación =====")
    print(f"Agente 1: {agent1_name} ({agent1_type}) - {agent1_path}")
    print(f"Agente 2: {agent2_name} ({agent2_type}) - {agent2_path}")
    print(f"Número de partidas por lado: {n_matches}")
    print(f"Temperatura: {temperature}")
    print(f"Modo determinista: {'Sí' if deterministic else 'No'}")
    print(f"Directorio de resultados: {output_dir}")

    confirm = input("\n¿Iniciar la comparación con estos parámetros? (s/n) [s]: ").lower() or "s"
    if confirm in ["s", "si", "sí", "y", "yes"]:
        print("\nIniciando comparación...\n")
        compare_agents(
            agent1_path=agent1_path,
            agent2_path=agent2_path,
            agent1_type=agent1_type,
            agent2_type=agent2_type,
            agent1_name=agent1_name,
            agent2_name=agent2_name,
            n_matches=n_matches,
            temperature=temperature,
            deterministic=deterministic,
            output_dir=output_dir
        )
    else:
        print("Comparación cancelada por el usuario.")


def auto_compare_all_agents():
    """Compara automáticamente todos los agentes disponibles entre sí."""
    print("\n===== Comparación Automática de Agentes =====")

    # Obtener la lista de agentes
    agents = list_available_agents(DEFAULT_AGENTS_PATH)

    if not agents:
        print(f"No se encontraron agentes en {DEFAULT_AGENTS_PATH}. Verifique la ruta.")
        return

    print(f"Se encontraron {len(agents)} agentes para comparar.")

    # Crear directorio para resultados
    os.makedirs(DEFAULT_RESULTS_PATH, exist_ok=True)

    # Parámetros predeterminados
    n_matches = 500
    temperature = 0.1
    deterministic = False

    # Comparar cada par de agentes
    total_comparisons = len(agents) * (len(agents) - 1) // 2
    current_comparison = 0

    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            current_comparison += 1
            agent1_path = agents[i]
            agent2_path = agents[j]

            agent1_name = Path(agent1_path).stem
            agent2_name = Path(agent2_path).stem

            print(f"\n[{current_comparison}/{total_comparisons}] Comparando {agent1_name} vs {agent2_name}")

            try:
                compare_agents(
                    agent1_path=agent1_path,
                    agent2_path=agent2_path,
                    agent1_type="CNN",
                    agent2_type="CNN",
                    agent1_name=agent1_name,
                    agent2_name=agent2_name,
                    n_matches=n_matches,
                    temperature=temperature,
                    deterministic=deterministic,
                    output_dir=DEFAULT_RESULTS_PATH
                )
            except Exception as e:
                print(f"Error al comparar {agent1_name} vs {agent2_name}: {e}")
                import traceback
                print(traceback.format_exc())

    print("\n===== Comparación Automática Completada =====")
    print(f"Resultados guardados en: {DEFAULT_RESULTS_PATH}")


def main():
    """Función principal para ejecutar la herramienta desde línea de comandos o automáticamente."""
    # Comprobar si se pasaron argumentos por línea de comandos
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            description="Comparar agentes de diferentes épocas y arquitecturas",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=__doc__.split('\n\nUso:')[1]
        )
        parser.add_argument("--agent1", type=str, help="Ruta al archivo del primer agente")
        parser.add_argument("--agent2", type=str, help="Ruta al archivo del segundo agente")
        parser.add_argument("--agent1-type", type=str, default="CNN",
                            help="Tipo del primer agente (cualquier arquitectura soportada, por defecto: CNN)")
        parser.add_argument("--agent2-type", type=str, default="CNN",
                            help="Tipo del segundo agente (cualquier arquitectura soportada, por defecto: CNN)")
        parser.add_argument("--agent1-name", type=str, help="Nombre para identificar al primer agente")
        parser.add_argument("--agent2-name", type=str, help="Nombre para identificar al segundo agente")
        parser.add_argument("--matches", type=int, default=500,
                            help="Número de partidas a jugar por lado (default: 500)")
        parser.add_argument("--temp", type=float, default=0.1, help="Temperatura para ambos agentes (default: 0.1)")
        parser.add_argument("--deterministic", action="store_true", help="Usar modo determinista")
        parser.add_argument("--output", type=str, default=DEFAULT_RESULTS_PATH,
                            help=f"Directorio para guardar resultados (default: {DEFAULT_RESULTS_PATH})")
        parser.add_argument("--list", action="store_true", help="Listar agentes disponibles")
        parser.add_argument("--agents-dir", type=str, default=DEFAULT_AGENTS_PATH,
                            help=f"Directorio donde buscar los agentes (default: {DEFAULT_AGENTS_PATH})")
        parser.add_argument("--auto", action="store_true", help="Comparar automáticamente todos los agentes entre sí")
        parser.add_argument("--interactive", action="store_true", help="Usar modo interactivo")

        args = parser.parse_args()

        # Si se solicita listar agentes
        if args.list:
            list_available_agents(args.agents_dir)
            return

        # Si se solicita modo automático
        if args.auto:
            auto_compare_all_agents()
            return

        # Si se solicita modo interactivo
        if args.interactive:
            interactive_mode()
            return

        # Verificar que se proporcionaron las rutas de los agentes
        if not args.agent1 or not args.agent2:
            parser.error("Se requieren las rutas de ambos agentes (--agent1 y --agent2)")

        # Ejecutar la comparación
        compare_agents(
            agent1_path=args.agent1,
            agent2_path=args.agent2,
            agent1_type=args.agent1_type,
            agent2_type=args.agent2_type,
            agent1_name=args.agent1_name,
            agent2_name=args.agent2_name,
            n_matches=args.matches,
            temperature=args.temp,
            deterministic=args.deterministic,
            output_dir=args.output
        )
    else:
        # Modo automático por defecto
        auto_compare_all_agents()


if __name__ == "__main__":
    main()