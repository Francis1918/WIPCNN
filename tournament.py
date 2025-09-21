#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
tournament.py - Torneo "todos contra todos" para agentes de Quarto.
Enfrenta a múltiples agentes de diferentes épocas entre sí y determina al campeón.

Uso:
    python tournament.py [--epochs E1 E2 E3...] [--matches N] [--temp T] [--visualize]

Ejemplos:
    python tournament.py                                 # Modo interactivo
    python tournament.py --epochs 1 50 100 150 200       # Torneo con épocas específicas
    python tournament.py --epochs 1 50 100 --matches 20  # 20 partidas por enfrentamiento
    python tournament.py --all                           # Usar todas las épocas disponibles
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import itertools
from tqdm import tqdm

# Importar funciones desde compare_agents.py (sin modificarlo)
from compare_agents import load_agent, show_available_models, compare_agents
from utils.logger import logger

def get_all_available_epochs():
    """Obtiene todas las épocas disponibles en el sistema."""
    weights_dir = "models/weights/QuartoCNN1"

    if not os.path.exists(weights_dir):
        logger.error(f"No se encontró el directorio de pesos: {weights_dir}")
        return []

    # Encontrar todos los archivos de modelo
    model_files = list(Path(weights_dir).glob("*-ba_increasing_n_last_states_epoch_*.pt"))

    if not model_files:
        logger.warning(f"No se encontraron modelos en {weights_dir}")
        return []

    # Extraer y ordenar las épocas disponibles
    available_epochs = set()
    for model_file in model_files:
        file_name = model_file.name
        try:
            # Extraer el número de época del formato
            epoch_str = file_name.split("epoch_")[1].split(".")[0]
            epoch = int(epoch_str)
            available_epochs.add(epoch)
        except (IndexError, ValueError):
            continue

    return sorted(list(available_epochs))

def select_epochs_for_tournament(max_agents=10):
    """Selecciona épocas representativas para un torneo manejable."""
    all_epochs = get_all_available_epochs()

    if not all_epochs:
        logger.error("No se encontraron épocas disponibles")
        return []

    if len(all_epochs) <= max_agents:
        return all_epochs

    # Seleccionar épocas distribuidas uniformemente
    n = min(max_agents, len(all_epochs))
    indices = np.round(np.linspace(0, len(all_epochs) - 1, n)).astype(int)
    selected_epochs = [all_epochs[i] for i in indices]

    return selected_epochs

def run_tournament(epochs, n_matches=10, temperature=0.5, visualize=False):
    """Ejecuta un torneo completo de todos contra todos entre las épocas especificadas.

    Args:
        epochs (list): Lista de épocas a enfrentar
        n_matches (int): Número de partidas por enfrentamiento
        temperature (float): Temperatura para los agentes
        visualize (bool): Si se deben guardar visualizaciones

    Returns:
        pd.DataFrame: Tabla de resultados del torneo
    """
    if len(epochs) < 2:
        logger.error("Se necesitan al menos 2 épocas para un torneo")
        return None

    # Crear un directorio para los resultados del torneo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tournament_dir = f"partidas_guardadas/tournament_{timestamp}"
    os.makedirs(tournament_dir, exist_ok=True)

    # Crear DataFrame para almacenar resultados
    results_df = pd.DataFrame(
        index=epochs,
        columns=epochs + ['Victorias', 'Derrotas', 'Empates', 'Puntos', 'Posición']
    )

    # Inicializar con ceros
    results_df.fillna(0, inplace=True)

    # Matriz para almacenar resultados de enfrentamientos directos
    matches_data = []

    # Progreso
    total_matches = len(list(itertools.combinations(epochs, 2)))
    logger.info(f"Iniciando torneo con {len(epochs)} agentes")
    logger.info(f"Total de enfrentamientos: {total_matches}")
    logger.info(f"Partidas por enfrentamiento: {n_matches}")

    # Todos contra todos
    for epoch1, epoch2 in tqdm(itertools.combinations(epochs, 2), total=total_matches, desc="Enfrentamientos"):
        logger.info(f"\n{'=' * 30}")
        logger.info(f"Enfrentamiento: Época {epoch1} vs Época {epoch2}")

        # Directorio específico para este enfrentamiento
        match_dir = f"{tournament_dir}/match_{epoch1}_vs_{epoch2}"
        os.makedirs(match_dir, exist_ok=True)

        # Realizar el enfrentamiento (usando la función de compare_agents.py)
        match_results = compare_agents(
            epoch1,
            epoch2,
            n_matches=n_matches,
            temperature=temperature,
            visualize=visualize
        )

        if not match_results:
            logger.warning(f"No se obtuvieron resultados para {epoch1} vs {epoch2}")
            continue

        # Registrar resultados en la matriz
        wins_1 = match_results['P1']
        wins_2 = match_results['P2']
        draws = match_results['Empates']

        # Actualizar DataFrame de resultados
        results_df.at[epoch1, epoch2] = wins_1
        results_df.at[epoch2, epoch1] = wins_2

        # Sumar victorias, derrotas y empates
        results_df.at[epoch1, 'Victorias'] += wins_1
        results_df.at[epoch2, 'Victorias'] += wins_2
        results_df.at[epoch1, 'Derrotas'] += wins_2
        results_df.at[epoch2, 'Derrotas'] += wins_1
        results_df.at[epoch1, 'Empates'] += draws
        results_df.at[epoch2, 'Empates'] += draws

        # Calcular puntos (3 por victoria, 1 por empate)
        results_df.at[epoch1, 'Puntos'] += (wins_1 * 3 + draws * 1)
        results_df.at[epoch2, 'Puntos'] += (wins_2 * 3 + draws * 1)

        # Guardar detalles del enfrentamiento
        matches_data.append({
            'Epoch1': epoch1,
            'Epoch2': epoch2,
            'Wins_Epoch1': wins_1,
            'Wins_Epoch2': wins_2,
            'Draws': draws,
            'Win_Rate_Epoch1': wins_1 / n_matches * 100,
            'Win_Rate_Epoch2': wins_2 / n_matches * 100,
            'Draw_Rate': draws / n_matches * 100
        })

    # Calcular posiciones finales
    positions = results_df['Puntos'].rank(method='min', ascending=False)
    results_df['Posición'] = positions

    # Ordenar por posición
    results_df = results_df.sort_values('Posición')

    # Mostrar tabla de resultados
    logger.info("\n" + "=" * 50)
    logger.info("RESULTADOS DEL TORNEO:")
    logger.info("=" * 50)

    # Tabla de posiciones
    position_table = results_df[['Victorias', 'Derrotas', 'Empates', 'Puntos', 'Posición']].sort_values('Posición')
    logger.info("\nTABLA DE POSICIONES:")
    logger.info("\n" + position_table.to_string())

    # Definir el campeón
    champion = position_table.index[0]
    logger.info(f"\n¡El CAMPEÓN del torneo es el agente de la Época {champion}!")

    # Si se solicitó visualización, crear gráficos
    if visualize:
        # Crear directorio para visualizaciones
        vis_dir = f"{tournament_dir}/visualizations"
        os.makedirs(vis_dir, exist_ok=True)

        # Graficar puntos totales
        plt.figure(figsize=(12, 6))
        plt.bar(position_table.index.astype(str), position_table['Puntos'], color='skyblue')
        plt.title('Puntos totales por agente')
        plt.xlabel('Época del agente')
        plt.ylabel('Puntos')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{vis_dir}/points_by_agent.png")
        plt.close()

        # Graficar victorias
        plt.figure(figsize=(12, 6))
        plt.bar(position_table.index.astype(str), position_table['Victorias'], color='green')
        plt.title('Victorias por agente')
        plt.xlabel('Época del agente')
        plt.ylabel('Número de victorias')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{vis_dir}/wins_by_agent.png")
        plt.close()

        # Matriz de calor para enfrentamientos directos
        plt.figure(figsize=(10, 8))
        matchup_matrix = results_df[epochs].copy()

        # Normalizar por número de partidas
        for col in epochs:
            matchup_matrix[col] = matchup_matrix[col] / n_matches

        plt.imshow(matchup_matrix, cmap='YlOrRd', interpolation='nearest')
        plt.colorbar(label='Tasa de victoria')
        plt.title('Matriz de enfrentamientos directos')
        plt.xlabel('Oponente (época)')
        plt.ylabel('Agente (época)')

        # Configurar etiquetas de ejes
        plt.xticks(np.arange(len(epochs)), epochs, rotation=45)
        plt.yticks(np.arange(len(epochs)), epochs)

        # Mostrar valores en las celdas
        for i in range(len(epochs)):
            for j in range(len(epochs)):
                plt.text(j, i, f'{matchup_matrix.iloc[i, j]:.2f}',
                         ha='center', va='center',
                         color='white' if matchup_matrix.iloc[i, j] > 0.5 else 'black')

        plt.tight_layout()
        plt.savefig(f"{vis_dir}/matchup_heatmap.png")
        plt.close()

    # Guardar resultados
    # Tabla de posiciones
    position_table.to_csv(f"{tournament_dir}/positions.csv")

    # Matriz completa
    results_df.to_csv(f"{tournament_dir}/full_results.csv")

    # Detalles de enfrentamientos
    pd.DataFrame(matches_data).to_csv(f"{tournament_dir}/matches_detail.csv", index=False)

    logger.info(f"\nResultados guardados en {tournament_dir}")

    return results_df

def main():
    """Función principal para ejecutar el torneo desde línea de comandos o interactivamente."""
    # Comprobar si se pasaron argumentos por línea de comandos
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            description="Torneo todos contra todos para agentes de Quarto",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=__doc__.split('\n\nUso:')[1]
        )
        parser.add_argument("--epochs", type=int, nargs='+', help="Lista de épocas para el torneo")
        parser.add_argument("--all", action="store_true", help="Usar todas las épocas disponibles")
        parser.add_argument("--max", type=int, default=10, help="Número máximo de agentes a incluir (con --all)")
        parser.add_argument("--matches", type=int, default=10, help="Número de partidas por enfrentamiento (default: 10)")
        parser.add_argument("--temp", type=float, default=0.5, help="Temperatura para los agentes (default: 0.5)")
        parser.add_argument("--visualize", action="store_true", help="Guardar partidas y generar visualización")

        args = parser.parse_args()

        if args.all:
            # Usar todas las épocas disponibles (o un subconjunto representativo)
            epochs = select_epochs_for_tournament(max_agents=args.max)
            if not epochs:
                logger.error("No se encontraron épocas disponibles")
                return
        elif args.epochs:
            # Usar las épocas especificadas
            epochs = args.epochs
        else:
            # Modo interactivo si no se especificaron épocas
            parser.print_help()
            return

        run_tournament(epochs, args.matches, args.temp, args.visualize)
    else:
        # Modo interactivo - pedir parámetros al usuario
        print("\n===== Torneo de Agentes para Quarto =====")
        print("Este programa organiza un torneo 'todos contra todos' entre agentes de diferentes épocas.")

        # Obtener épocas disponibles
        available_epochs = get_all_available_epochs()

        if not available_epochs:
            print("No se encontraron épocas disponibles. Verifique que existan modelos entrenados.")
            return

        print(f"\nSe encontraron {len(available_epochs)} épocas disponibles.")

        # Preguntar si quiere todas las épocas o selección
        all_epochs = input("¿Desea incluir todas las épocas disponibles? (s/n) [n]: ").lower() or "n"

        if all_epochs in ["s", "si", "sí", "y", "yes"]:
            # Verificar si hay demasiadas épocas
            if len(available_epochs) > 10:
                print(f"Hay {len(available_epochs)} épocas disponibles, lo que generaría {len(available_epochs) * (len(available_epochs) - 1) // 2} enfrentamientos.")
                max_agents = input("Ingrese el número máximo de agentes para el torneo [10]: ") or "10"
                try:
                    max_agents = int(max_agents)
                    if max_agents < 2:
                        max_agents = 2
                    epochs = select_epochs_for_tournament(max_agents)
                except ValueError:
                    print("Valor inválido, usando 10 como máximo.")
                    epochs = select_epochs_for_tournament(10)
            else:
                epochs = available_epochs

            print(f"Se utilizarán {len(epochs)} épocas: {epochs}")
        else:
            # Pedir épocas específicas
            print("\nÉpocas disponibles:", available_epochs)

            # Ofrecer algunas selecciones predefinidas
            print("\nOpciones recomendadas:")
            print("1. Primeros modelos vs últimos (selecciona automáticamente 5 épocas distribuidas)")
            print("2. Comparación de modelos clave (primero, 25%, 50%, 75%, último)")
            print("3. Selección manual de épocas")

            option = input("\nSeleccione una opción [1]: ") or "1"

            if option == "1":
                # 5 épocas distribuidas
                epochs = select_epochs_for_tournament(5)
            elif option == "2":
                # Modelos clave
                indices = [0, len(available_epochs)//4, len(available_epochs)//2,
                           3*len(available_epochs)//4, len(available_epochs)-1]
                epochs = [available_epochs[i] for i in indices]
            else:
                # Selección manual
                epochs_input = input("\nIngrese las épocas separadas por espacios (ej: 1 50 100 150 200): ")
                try:
                    epochs = [int(e) for e in epochs_input.split()]
                    # Verificar que las épocas existan
                    invalid_epochs = [e for e in epochs if e not in available_epochs]
                    if invalid_epochs:
                        print(f"Advertencia: Las siguientes épocas no están disponibles: {invalid_epochs}")
                        epochs = [e for e in epochs if e in available_epochs]
                except ValueError:
                    print("Entrada inválida. Usando las primeras 5 épocas disponibles.")
                    epochs = available_epochs[:5]

            if len(epochs) < 2:
                print("Se necesitan al menos 2 épocas para un torneo. Usando las primeras 2 épocas disponibles.")
                epochs = available_epochs[:2]

            print(f"Épocas seleccionadas para el torneo: {epochs}")

        # Solicitar otros parámetros
        while True:
            try:
                n_matches = int(input("\nIngrese el número de partidas por enfrentamiento [10]: ") or "10")
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
        print("\n===== Parámetros del torneo =====")
        print(f"Número de agentes: {len(epochs)}")
        print(f"Épocas: {epochs}")
        print(f"Número de enfrentamientos: {len(epochs) * (len(epochs) - 1) // 2}")
        print(f"Partidas por enfrentamiento: {n_matches}")
        print(f"Temperatura: {temperature}")
        print(f"Visualizar resultados: {'Sí' if visualize else 'No'}")

        confirm = input("\n¿Iniciar el torneo con estos parámetros? (s/n) [s]: ").lower() or "s"
        if confirm in ["s", "si", "sí", "y", "yes"]:
            print("\nIniciando torneo...\n")
            run_tournament(epochs, n_matches, temperature, visualize)
        else:
            print("Torneo cancelado por el usuario.")

if __name__ == "__main__":
    main()
