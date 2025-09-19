#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
compare_agents.py - Herramienta para comparar agentes de diferentes épocas en el juego Quarto.
Permite enfrentar agentes específicos y analizar su rendimiento relativo.

Uso:
    python compare_agents.py <epoca1> <epoca2> [--matches N] [--temp T] [--visualize]

Ejemplos:
    python compare_agents.py 1 100                   # Enfrentar época 1 vs época 100 (10 partidas por defecto)
    python compare_agents.py 1 100 --matches 50      # Enfrentar con 50 partidas
    python compare_agents.py 1 100 --visualize       # Guardar partidas y generar visualización
    python compare_agents.py 1 100 --temp 0.1        # Usar temperatura baja para decisiones más deterministas
"""

import argparse
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import os
import pandas as pd
import numpy as np
from datetime import datetime
import sys

from utils.logger import logger
from models.CNN1 import QuartoCNN
from bot.CNN_bot import Quarto_bot

# Importar directamente play_games de quartopy para nuestra implementación personalizada
try:
    from quartopy import play_games
except ImportError:
    # Fallback para dependencias
    import setup_dependencies
    setup_dependencies.setup_quartopy(silent=False)
    from quartopy import play_games

def load_agent(epoch, temperature=0.5):
    """Carga un agente desde un checkpoint de la época especificada."""
    # Buscar modelos en la carpeta models/weights/QuartoCNN1
    weights_dir = "models/weights/QuartoCNN1"

    if not os.path.exists(weights_dir):
        logger.error(f"No se encontró el directorio de pesos: {weights_dir}")
        raise FileNotFoundError(f"El directorio {weights_dir} no existe")

    # Buscar modelos con el formato específico para la época solicitada
    model_pattern = f"*-ba_increasing_n_last_states_epoch_{epoch:04d}.pt"
    matching_files = list(Path(weights_dir).glob(model_pattern))

    if not matching_files:
        # Si no se encuentra un modelo específico, listar las épocas disponibles
        logger.warning(f"No se encontró ningún modelo para la época {epoch} en {weights_dir}")
        logger.info("Buscando modelos disponibles...")
        show_available_models()
        raise FileNotFoundError(f"No se encontró el modelo para la época {epoch}. Por favor, verifica las épocas disponibles mostradas arriba.")

    # Ordenar por fecha de modificación (más reciente primero)
    matching_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    model_path = str(matching_files[0])

    logger.info(f"Modelo encontrado para la época {epoch}: {model_path}")

    # Cargar el modelo
    model = QuartoCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Crear el bot con el modelo cargado
    bot = Quarto_bot(model=model)

    # Configurar los atributos de temperatura manualmente
    bot.DETERMINISTIC = False  # Usar modo estocástico
    bot.TEMPERATURE = temperature  # Establecer la temperatura deseada
    logger.info(f"Bot configurado con temperatura: {temperature}")

    return bot

def show_available_models():
    """Muestra una lista de los modelos disponibles para ayudar al usuario."""
    weights_dir = "models/weights/QuartoCNN1"

    if not os.path.exists(weights_dir):
        logger.error(f"No se encontró el directorio de pesos: {weights_dir}")
        return

    logger.info("Modelos disponibles:")

    # Encontrar todos los archivos de modelo
    model_files = list(Path(weights_dir).glob("*-ba_increasing_n_last_states_epoch_*.pt"))

    if not model_files:
        logger.warning(f"No se encontraron modelos en {weights_dir}")
        return

    # Extraer y ordenar las épocas disponibles
    available_epochs = set()
    for model_file in model_files:
        file_name = model_file.name
        try:
            # Extraer el número de época del formato "XXXX-ba_increasing_n_last_states_epoch_YYYY.pt"
            epoch_str = file_name.split("epoch_")[1].split(".")[0]
            epoch = int(epoch_str)
            available_epochs.add(epoch)
        except (IndexError, ValueError):
            continue

    if not available_epochs:
        logger.warning("No se pudo extraer información de época de los nombres de archivo")
        return

    available_epochs = sorted(list(available_epochs))
    logger.info(f"Épocas disponibles: {available_epochs}")

    # Mostrar algunos ejemplos de épocas para que el usuario pueda elegir
    if len(available_epochs) >= 2:
        logger.info("Ejemplos de uso:")
        logger.info(f"  python compare_agents.py {available_epochs[0]} {available_epochs[-1]}")
        logger.info(f"  python compare_agents.py {available_epochs[0]} {available_epochs[len(available_epochs)//2]}")
    elif len(available_epochs) == 1:
        logger.info(f"Solo hay un modelo disponible: época {available_epochs[0]}")
        logger.info(f"  python compare_agents.py {available_epochs[0]} {available_epochs[0]}")

def compare_agents(epoch1, epoch2, n_matches=10, temperature=0.5, visualize=False):
    """Compara dos agentes de diferentes épocas.

    Args:
        epoch1 (int): Número de época del primer agente
        epoch2 (int): Número de época del segundo agente
        n_matches (int): Número de partidas a jugar
        temperature (float): Temperatura para ambos agentes (determina exploración vs explotación)
        visualize (bool): Si es True, guarda partidas y genera visualización

    Returns:
        dict: Resultados del enfrentamiento
    """
    logger.info(f"Comparando agente de época {epoch1} contra agente de época {epoch2}")
    logger.info(f"Número de partidas: {n_matches}")
    logger.info(f"Temperatura: {temperature}")

    try:
        agent1 = load_agent(epoch1, temperature)
        agent2 = load_agent(epoch2, temperature)

        logger.info("Agentes cargados correctamente. Comenzando enfrentamiento...")

        # Crear directorio para guardar partidas si se solicita visualización
        save_dir = None
        if visualize:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f"partidas_guardadas/compare_{epoch1}_vs_{epoch2}_{timestamp}"
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"Las partidas se guardarán en: {save_dir}")

        # Enfrentar los agentes usando la función existente
        results = play_games(
            player1=agent1,
            player2=agent2,
            matches=n_matches,
            verbose=True,
            match_dir=save_dir if save_dir else None,
            return_file_paths=False
        )

        # Mostrar resultados
        logger.info("\nResultados:")
        logger.info(f"Victorias del agente de época {epoch1}: {results['P1']}")
        logger.info(f"Victorias del agente de época {epoch2}: {results['P2']}")
        logger.info(f"Empates: {results['Empates']}")

        win_rate_agent1 = results['P1'] / n_matches * 100
        win_rate_agent2 = results['P2'] / n_matches * 100
        draw_rate = results['Empates'] / n_matches * 100

        logger.info(f"\nTasa de victoria del agente de época {epoch1}: {win_rate_agent1:.2f}%")
        logger.info(f"Tasa de victoria del agente de época {epoch2}: {win_rate_agent2:.2f}%")
        logger.info(f"Tasa de empate: {draw_rate:.2f}%")

        # Visualizar resultados si se solicita
        if visualize:
            # Crear gráfico circular
            labels = [f'Época {epoch1}', f'Época {epoch2}', 'Empates']
            sizes = [results['P1'], results['P2'], results['Empates']]
            colors = ['#ff9999', '#66b3ff', '#c2c2f0']

            plt.figure(figsize=(10, 7))
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title(f'Comparación de rendimiento: Época {epoch1} vs Época {epoch2}')

            # Crear directorio para visualizaciones si no existe
            vis_dir = "analysis/agent_comparisons"
            os.makedirs(vis_dir, exist_ok=True)

            # Guardar gráfico
            comparison_file = f"{vis_dir}/comparison_{epoch1}_vs_{epoch2}_{timestamp}.png"
            plt.savefig(comparison_file)
            plt.close()

            logger.info(f"\nGráfico guardado como {comparison_file}")

            # Guardar resultados en CSV para análisis posterior
            results_df = pd.DataFrame({
                'epoch1': [epoch1],
                'epoch2': [epoch2],
                'wins_epoch1': [results['P1']],
                'wins_epoch2': [results['P2']],
                'draws': [results['Empates']],
                'win_rate_epoch1': [win_rate_agent1],
                'win_rate_epoch2': [win_rate_agent2],
                'draw_rate': [draw_rate],
                'n_matches': [n_matches],
                'temperature': [temperature],
                'timestamp': [timestamp]
            })

            csv_file = f"{vis_dir}/comparison_results.csv"
            # Añadir al archivo existente o crear uno nuevo
            if os.path.exists(csv_file):
                results_df.to_csv(csv_file, mode='a', header=False, index=False)
            else:
                results_df.to_csv(csv_file, index=False)

            logger.info(f"Resultados guardados en {csv_file}")

        return results

    except Exception as e:
        logger.error(f"Error durante la comparación: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    """Función principal para ejecutar la herramienta desde línea de comandos o interactivamente."""
    # Comprobar si se pasaron argumentos por línea de comandos
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            description="Comparar agentes de diferentes épocas",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=__doc__.split('\n\nUso:')[1]
        )
        parser.add_argument("epoch1", type=int, help="Número de época del primer agente")
        parser.add_argument("epoch2", type=int, help="Número de época del segundo agente")
        parser.add_argument("--matches", type=int, default=10, help="Número de partidas a jugar (default: 10)")
        parser.add_argument("--temp", type=float, default=0.5, help="Temperatura para ambos agentes (default: 0.5)")
        parser.add_argument("--visualize", action="store_true", help="Guardar partidas y generar visualización")

        args = parser.parse_args()
        compare_agents(args.epoch1, args.epoch2, args.matches, args.temp, args.visualize)
    else:
        # Modo interactivo - pedir parámetros al usuario
        print("\n===== Comparador de Agentes para Quarto =====")
        print("Este programa permite enfrentar agentes de diferentes épocas para evaluar su rendimiento.")

        # Mostrar épocas disponibles
        weights_dir = "models/weights/QuartoCNN1"
        available_epochs = []

        if os.path.exists(weights_dir):
            model_files = list(Path(weights_dir).glob("*-ba_increasing_n_last_states_epoch_*.pt"))
            for model_file in model_files:
                file_name = model_file.name
                try:
                    epoch_str = file_name.split("epoch_")[1].split(".")[0]
                    epoch = int(epoch_str)
                    available_epochs.append(epoch)
                except (IndexError, ValueError):
                    continue

            available_epochs = sorted(list(set(available_epochs)))
            if available_epochs:
                print("\nÉpocas disponibles:", available_epochs)

                # Sugerir algunas épocas interesantes para comparar
                if len(available_epochs) >= 2:
                    print(f"Sugerencias: Primera época ({available_epochs[0]}) vs. Última época ({available_epochs[-1]})")
                    if len(available_epochs) > 10:
                        mid_index = len(available_epochs) // 2
                        print(f"             Época temprana ({available_epochs[3]}) vs. Época media ({available_epochs[mid_index]})")
                        print(f"             Época media ({available_epochs[mid_index]}) vs. Última época ({available_epochs[-1]})")

        # Solicitar parámetros
        while True:
            try:
                epoch1 = int(input("\nIngrese el número de época del primer agente: "))
                break
            except ValueError:
                print("Error: Por favor ingrese un número entero válido.")

        while True:
            try:
                epoch2 = int(input("Ingrese el número de época del segundo agente: "))
                break
            except ValueError:
                print("Error: Por favor ingrese un número entero válido.")

        while True:
            try:
                n_matches = int(input("Ingrese el número de partidas a jugar [10]: ") or "10")
                if n_matches <= 0:
                    print("Error: El número de partidas debe ser mayor que cero.")
                    continue
                break
            except ValueError:
                print("Error: Por favor ingrese un número entero válido.")

        while True:
            try:
                temp_input = input("Ingrese la temperatura para los agentes (0.1-1.0) [0.5]: ") or "0.5"
                temperature = float(temp_input)
                if temperature <= 0 or temperature > 1:
                    print("Error: La temperatura debe estar entre 0.1 y 1.0.")
                    continue
                break
            except ValueError:
                print("Error: Por favor ingrese un número decimal válido.")

        visualize_input = input("¿Desea visualizar y guardar los resultados? (s/n) [s]: ").lower() or "s"
        visualize = visualize_input in ["s", "si", "sí", "y", "yes"]

        # Resumen de parámetros
        print("\n===== Parámetros de la comparación =====")
        print(f"Agente 1: Época {epoch1}")
        print(f"Agente 2: Época {epoch2}")
        print(f"Número de partidas: {n_matches}")
        print(f"Temperatura: {temperature}")
        print(f"Visualizar resultados: {'Sí' if visualize else 'No'}")

        confirm = input("\n¿Iniciar la comparación con estos parámetros? (s/n) [s]: ").lower() or "s"
        if confirm in ["s", "si", "sí", "y", "yes"]:
            print("\nIniciando comparación...\n")
            compare_agents(epoch1, epoch2, n_matches, temperature, visualize)
        else:
            print("Comparación cancelada por el usuario.")

if __name__ == "__main__":
    main()
