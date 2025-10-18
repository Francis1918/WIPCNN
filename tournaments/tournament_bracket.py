#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
tournament_bracket.py - Torneo de eliminación directa para agentes de Quarto.
Implementa un sistema de bracket similar al de la Champions League con rondas eliminatorias.

Uso:
    python tournament_bracket.py [--epochs E1 E2 E3...] [--matches N] [--temp T] [--visualize] [--workers W]
    python tournament_bracket.py [--physical-only]

Ejemplos:
    python tournament_bracket.py                                # Modo interactivo
    python tournament_bracket.py --epochs 1 50 100 150 200      # Bracket con épocas específicas
    python tournament_bracket.py --all --workers 4              # Usar todas las épocas y 4 trabajadores
    python tournament_bracket.py --physical-only                # Usar solo núcleos físicos
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from datetime import datetime
import itertools
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import platform
import subprocess
import re
import ctypes
import math
import random

# Importar funciones desde compare_agents.py
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

def get_next_power_of_2(n):
    """Encuentra la siguiente potencia de 2 mayor o igual a n."""
    return 2 ** math.ceil(math.log2(n))

def select_epochs_for_bracket(epochs_list, bracket_size=None):
    """Selecciona y organiza épocas para un bracket de eliminación directa."""
    if bracket_size is None:
        bracket_size = get_next_power_of_2(len(epochs_list))

    # Si hay más épocas que espacios en el bracket, seleccionar las mejores distribuidas
    if len(epochs_list) > bracket_size:
        # Seleccionar épocas distribuidas uniformemente
        indices = np.round(np.linspace(0, len(epochs_list) - 1, bracket_size)).astype(int)
        selected_epochs = [epochs_list[i] for i in indices]
    else:
        selected_epochs = epochs_list.copy()

    # Completar con "BYE" si es necesario
    while len(selected_epochs) < bracket_size:
        selected_epochs.append("BYE")

    # Organizar el bracket de manera que los mejores (épocas más altas) no se enfrenten al principio
    # Usar seeding típico de torneos deportivos
    seeded_bracket = []
    n = len([e for e in selected_epochs if e != "BYE"])

    # Separar agentes reales de BYEs
    real_agents = [e for e in selected_epochs if e != "BYE"]
    byes = [e for e in selected_epochs if e == "BYE"]

    # Ordenar agentes reales de mayor a menor (mejores primero)
    real_agents.sort(reverse=True)

    # Aplicar seeding estándar
    seeded = [None] * bracket_size
    seeds = list(range(1, len(real_agents) + 1))

    # Colocar seeds en posiciones según el patrón estándar de torneos
    for i, seed in enumerate(seeds):
        if i < len(real_agents):
            seeded[get_bracket_position(seed, bracket_size)] = real_agents[i]

    # Llenar posiciones vacías con BYEs
    for i in range(bracket_size):
        if seeded[i] is None:
            if byes:
                seeded[i] = byes.pop()
            else:
                seeded[i] = "BYE"

    return seeded

def get_bracket_position(seed, bracket_size):
    """Calcula la posición en el bracket según el seed usando el patrón estándar."""
    # Patrón estándar de seeding para torneos de eliminación directa
    if bracket_size == 2:
        return seed - 1
    elif bracket_size == 4:
        positions = {1: 0, 4: 1, 2: 2, 3: 3}
    elif bracket_size == 8:
        positions = {1: 0, 8: 1, 4: 2, 5: 3, 2: 4, 7: 5, 3: 6, 6: 7}
    elif bracket_size == 16:
        positions = {1: 0, 16: 1, 8: 2, 9: 3, 4: 4, 13: 5, 5: 6, 12: 7,
                    2: 8, 15: 9, 7: 10, 10: 11, 3: 12, 14: 13, 6: 14, 11: 15}
    else:
        # Para tamaños más grandes, usar posición secuencial
        return seed - 1

    return positions.get(seed, seed - 1)

def run_match_bracket(args):
    """Función para ejecutar un enfrentamiento individual en el bracket."""
    agent1, agent2, n_matches, temperature, visualize, tournament_dir, round_name, match_id = args

    match_start = time.time()
    process_id = os.getpid()

    print(f"[Proceso {process_id}] {round_name} - Enfrentamiento {match_id}: {agent1} vs {agent2}")

    # Manejar BYEs
    if agent1 == "BYE":
        print(f"[Proceso {process_id}] {agent2} avanza automáticamente (BYE)")
        return agent1, agent2, {"winner": agent2, "P1": 0, "P2": 0, "Empates": 0}, None
    elif agent2 == "BYE":
        print(f"[Proceso {process_id}] {agent1} avanza automáticamente (BYE)")
        return agent1, agent2, {"winner": agent1, "P1": 0, "P2": 0, "Empates": 0}, None

    # Directorio específico para este enfrentamiento
    match_dir = f"{tournament_dir}/{round_name}/match_{match_id}_{agent1}_vs_{agent2}"
    os.makedirs(match_dir, exist_ok=True)

    # Realizar el enfrentamiento
    match_results = compare_agents(
        agent1,
        agent2,
        n_matches=n_matches,
        temperature=temperature,
        visualize=visualize
    )

    if not match_results:
        print(f"[Proceso {process_id}] No se obtuvieron resultados para {agent1} vs {agent2}")
        return agent1, agent2, None, None

    # Extraer resultados
    wins_1 = match_results['P1']
    wins_2 = match_results['P2']
    draws = match_results['Empates']

    # Determinar ganador
    if wins_1 > wins_2:
        winner = agent1
    elif wins_2 > wins_1:
        winner = agent2
    else:
        # En caso de empate, jugar partidas de desempate
        print(f"[Proceso {process_id}] Empate {wins_1}-{wins_2}, jugando desempate...")

        # Partida de desempate (mejor de 3)
        tiebreak_results = compare_agents(
            agent1,
            agent2,
            n_matches=3,
            temperature=temperature,
            visualize=False
        )

        if tiebreak_results:
            tb_wins_1 = tiebreak_results['P1']
            tb_wins_2 = tiebreak_results['P2']

            if tb_wins_1 > tb_wins_2:
                winner = agent1
            elif tb_wins_2 > tb_wins_1:
                winner = agent2
            else:
                # Si aún hay empate, decidir por época más alta
                winner = max(agent1, agent2)
                print(f"[Proceso {process_id}] Empate en desempate, avanza {winner} (época más alta)")
        else:
            # Si no se puede resolver, decidir por época más alta
            winner = max(agent1, agent2)

    # Guardar detalles del enfrentamiento
    match_data = {
        'Round': round_name,
        'Match_ID': match_id,
        'Agent1': agent1,
        'Agent2': agent2,
        'Wins_Agent1': wins_1,
        'Wins_Agent2': wins_2,
        'Draws': draws,
        'Winner': winner,
        'Duration': time.time() - match_start
    }

    # Actualizar results with winner
    match_results['winner'] = winner

    print(f"[Proceso {process_id}] Completado: {agent1} vs {agent2} - Resultado: {wins_1}-{wins_2}-{draws} - Ganador: {winner}")

    return agent1, agent2, match_results, match_data

def run_bracket_tournament(epochs, n_matches=10, temperature=0.5, visualize=False, n_workers=None, physical_only=False, specific_cores=None):
    """Ejecuta un torneo de eliminación directa completo."""

    if len(epochs) < 2:
        logger.error("Se necesitan al menos 2 épocas para un torneo")
        return None

    # Determinar tamaño del bracket
    bracket_size = get_next_power_of_2(len(epochs))
    logger.info(f"Creando bracket de {bracket_size} participantes para {len(epochs)} épocas")

    # Organizar el bracket
    bracket_epochs = select_epochs_for_bracket(epochs, bracket_size)

    # Determinar número de trabajadores
    from tournament_parallel import get_cores_for_parallelism, get_cpu_info

    if specific_cores is not None:
        n_workers = len(specific_cores)
    elif n_workers is None:
        n_workers = get_cores_for_parallelism(physical_only)

    # Crear directorio para resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tournament_dir = f"partidas_guardadas/tournament_bracket_{timestamp}"
    os.makedirs(tournament_dir, exist_ok=True)

    # Estructura para almacenar resultados de todas las rondas
    all_rounds_data = []
    bracket_results = {}

    # Inicializar el bracket
    current_round_participants = bracket_epochs.copy()
    round_number = 1

    logger.info(f"Iniciando torneo de eliminación directa")
    logger.info(f"Participantes iniciales: {[str(p) for p in current_round_participants if p != 'BYE']}")

    # Procesar cada ronda hasta que quede un ganador
    while len(current_round_participants) > 1:
        # Determinar nombre de la ronda
        participants_count = len([p for p in current_round_participants if p != "BYE"])
        if participants_count <= 1:
            break
        elif participants_count == 2:
            round_name = "Final"
        elif participants_count <= 4:
            round_name = "Semifinales"
        elif participants_count <= 8:
            round_name = "Cuartos_de_Final"
        elif participants_count <= 16:
            round_name = "Octavos_de_Final"
        else:
            round_name = f"Ronda_{round_number}"

        logger.info(f"\n===== {round_name.replace('_', ' ').upper()} =====")
        logger.info(f"Participantes: {[str(p) for p in current_round_participants if p != 'BYE']}")

        # Crear enfrentamientos para esta ronda
        matches = []
        for i in range(0, len(current_round_participants), 2):
            agent1 = current_round_participants[i]
            agent2 = current_round_participants[i + 1] if i + 1 < len(current_round_participants) else "BYE"
            match_id = f"{round_name}_M{len(matches) + 1}"
            matches.append((agent1, agent2, n_matches, temperature, visualize, tournament_dir, round_name, match_id))

        # Ejecutar enfrentamientos de esta ronda en paralelo
        round_winners = []
        round_data = []

        if n_workers > 1 and len(matches) > 1:
            # Ejecución paralela
            with ProcessPoolExecutor(max_workers=min(n_workers, len(matches))) as executor:
                future_to_match = {executor.submit(run_match_bracket, match): match for match in matches}

                for future in tqdm(as_completed(future_to_match), total=len(matches), desc=f"{round_name}"):
                    try:
                        agent1, agent2, match_results, match_data = future.result()

                        if match_results and 'winner' in match_results:
                            round_winners.append(match_results['winner'])
                            if match_data:
                                round_data.append(match_data)
                    except Exception as e:
                        logger.error(f"Error en enfrentamiento: {e}")
        else:
            # Ejecución secuencial
            for match in tqdm(matches, desc=f"{round_name}"):
                try:
                    agent1, agent2, match_results, match_data = run_match_bracket(match)

                    if match_results and 'winner' in match_results:
                        round_winners.append(match_results['winner'])
                        if match_data:
                            round_data.append(match_data)
                except Exception as e:
                    logger.error(f"Error en enfrentamiento: {e}")

        # Guardar datos de esta ronda
        all_rounds_data.extend(round_data)
        bracket_results[round_name] = {
            'participants': [p for p in current_round_participants if p != "BYE"],
            'winners': round_winners,
            'matches_data': round_data
        }

        # Preparar para la siguiente ronda
        current_round_participants = round_winners
        round_number += 1

        logger.info(f"Ganadores de {round_name.replace('_', ' ')}: {round_winners}")

    # Determinar campeón
    champion = current_round_participants[0] if current_round_participants else None

    if champion:
        logger.info(f"\n🏆 ¡CAMPEÓN DEL TORNEO: Época {champion}! 🏆")
    else:
        logger.error("No se pudo determinar un campeón")

    # Crear visualización del bracket
    if visualize and champion:
        create_bracket_visualization(bracket_results, bracket_epochs, champion, tournament_dir)

    # Guardar resultados detallados
    save_bracket_results(bracket_results, all_rounds_data, champion, tournament_dir)

    return bracket_results, champion

def create_bracket_visualization(bracket_results, initial_participants, champion, tournament_dir):
    """Crea una visualización del bracket de eliminación directa."""

    vis_dir = f"{tournament_dir}/visualizations"
    os.makedirs(vis_dir, exist_ok=True)

    # Crear figura grande para el bracket
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Título
    ax.text(5, 9.5, 'TORNEO DE ELIMINACIÓN DIRECTA - BRACKET',
            fontsize=20, fontweight='bold', ha='center')
    ax.text(5, 9, f'*** CAMPEÓN: Época {champion} ***',
            fontsize=16, fontweight='bold', ha='center', color='gold')

    # Dibujar el bracket (simplificado)
    rounds = list(bracket_results.keys())

    # Posiciones y para cada ronda
    y_positions = {
        1: [1, 2, 3, 4, 5, 6, 7, 8],  # Primera ronda
        2: [1.5, 3.5, 5.5, 7.5],      # Segunda ronda
        3: [2.5, 6.5],                # Tercera ronda
        4: [4.5]                      # Final
    }

    x_positions = [1, 3, 5, 7]  # Posiciones x para cada ronda

    # Dibujar cada ronda
    for round_idx, round_name in enumerate(rounds):
        round_data = bracket_results[round_name]
        x = x_positions[min(round_idx, len(x_positions) - 1)]

        # Dibujar participantes
        participants = round_data['participants']
        winners = round_data['winners']

        for i, participant in enumerate(participants[:8]):  # Limitar a 8 para visualización
            if i < len(y_positions.get(round_idx + 1, [])):
                y = y_positions[round_idx + 1][i]

                # Color según si es ganador
                color = 'lightgreen' if participant in winners else 'lightblue'
                if participant == champion:
                    color = 'gold'

                # Dibujar rectángulo del participante
                rect = patches.Rectangle((x-0.3, y-0.1), 0.6, 0.2,
                                       linewidth=1, edgecolor='black',
                                       facecolor=color)
                ax.add_patch(rect)

                # Texto del participante
                text = f"E{participant}" if participant != "BYE" else "BYE"
                ax.text(x, y, text, ha='center', va='center', fontweight='bold')

    # Guardar visualización
    plt.tight_layout()
    plt.savefig(f"{vis_dir}/bracket_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Crear gráfico de resumen de rondas
    fig, ax = plt.subplots(figsize=(12, 8))

    round_names = []
    participants_count = []

    for round_name, data in bracket_results.items():
        round_names.append(round_name.replace('_', ' '))
        participants_count.append(len(data['participants']))

    bars = ax.bar(round_names, participants_count, color='skyblue', edgecolor='black')

    # Añadir valores en las barras
    for bar, count in zip(bars, participants_count):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}', ha='center', va='bottom', fontweight='bold')

    ax.set_title('Participantes por Ronda', fontsize=16, fontweight='bold')
    ax.set_ylabel('Número de Participantes')
    ax.set_xlabel('Ronda')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{vis_dir}/rounds_summary.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_bracket_results(bracket_results, all_rounds_data, champion, tournament_dir):
    """Guarda los resultados detallados del torneo de bracket."""

    # Guardar resultados por ronda
    rounds_summary = []
    for round_name, data in bracket_results.items():
        rounds_summary.append({
            'Round': round_name,
            'Participants': len(data['participants']),
            'Matches': len(data['matches_data']),
            'Winners': ', '.join(map(str, data['winners']))
        })

    rounds_df = pd.DataFrame(rounds_summary)
    rounds_df.to_csv(f"{tournament_dir}/rounds_summary.csv", index=False)

    # Guardar todos los enfrentamientos
    if all_rounds_data:
        matches_df = pd.DataFrame(all_rounds_data)
        matches_df.to_csv(f"{tournament_dir}/all_matches.csv", index=False)

    # Guardar información del campeón
    with open(f"{tournament_dir}/champion_info.txt", "w", encoding='utf-8') as f:
        f.write(f"TORNEO DE ELIMINACIÓN DIRECTA\n")
        f.write(f"================================\n\n")
        f.write(f"🏆 CAMPEÓN: Época {champion}\n\n")
        f.write(f"RESUMEN DEL TORNEO:\n")
        f.write(f"Total de rondas: {len(bracket_results)}\n")
        f.write(f"Total de enfrentamientos: {len(all_rounds_data)}\n\n")

        f.write(f"RONDAS DISPUTADAS:\n")
        for round_name, data in bracket_results.items():
            f.write(f"\n{round_name.replace('_', ' ').upper()}:\n")
            f.write(f"  Participantes: {', '.join(map(str, data['participants']))}\n")
            f.write(f"  Ganadores: {', '.join(map(str, data['winners']))}\n")

def main():
    """Función principal para ejecutar el torneo de bracket."""
    # Reutilizar la lógica de argumentos de tournament_parallel
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            description="Torneo de eliminación directa para agentes de Quarto",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=__doc__.split('\n\nUso:')[1]
        )
        parser.add_argument("--epochs", type=int, nargs='+', help="Lista de épocas para el torneo")
        parser.add_argument("--all", action="store_true", help="Usar todas las épocas disponibles")
        parser.add_argument("--max", type=int, default=64, help="Número máximo de agentes a incluir (con --all)")
        parser.add_argument("--matches", type=int, default=10, help="Número de partidas por enfrentamiento (default: 10)")
        parser.add_argument("--temp", type=float, default=0.5, help="Temperatura para los agentes (default: 0.5)")
        parser.add_argument("--visualize", action="store_true", help="Guardar partidas y generar visualización")
        parser.add_argument("--workers", type=int, default=None,
                           help="Número de trabajadores paralelos (default: número de CPUs)")
        parser.add_argument("--cores", type=str, help="Lista de núcleos específicos a utilizar (ej: 0,1,2,5)")
        parser.add_argument("--physical-only", action="store_true", help="Usar solo núcleos físicos")

        args = parser.parse_args()

        if args.all:
            epochs = get_all_available_epochs()
            if len(epochs) > args.max:
                # Seleccionar épocas distribuidas
                indices = np.round(np.linspace(0, len(epochs) - 1, args.max)).astype(int)
                epochs = [epochs[i] for i in indices]
        elif args.epochs:
            epochs = args.epochs
        else:
            parser.print_help()
            return

        # Procesar núcleos específicos
        specific_cores = None
        if args.cores:
            try:
                specific_cores = [int(c.strip()) for c in args.cores.split(',')]
            except ValueError:
                logger.error("Formato inválido para núcleos específicos")
                return

        # Ejecutar torneo
        run_bracket_tournament(
            epochs=epochs,
            n_matches=args.matches,
            temperature=args.temp,
            visualize=args.visualize,
            n_workers=args.workers,
            physical_only=args.physical_only,
            specific_cores=specific_cores
        )
    else:
        # Modo interactivo
        print("\n===== Torneo de Eliminación Directa para Agentes de Quarto =====")
        print("Este programa organiza un torneo de bracket con eliminación directa.")
        print("Similar al formato de la Champions League con rondas eliminatorias.")

        # Obtener épocas disponibles
        available_epochs = get_all_available_epochs()
        if not available_epochs:
            print("No se encontraron épocas disponibles.")
            return

        print(f"\nSe encontraron {len(available_epochs)} épocas disponibles.")
        print("Épocas disponibles:", available_epochs)

        # Selección de participantes
        print("\nOpciones de participantes:")
        print("1. Seleccionar épocas específicas")
        print("2. Usar todas las épocas disponibles")
        print("3. Usar épocas distribuidas (recomendado para brackets grandes)")

        option = input("\nSeleccione una opción [1]: ") or "1"

        match option:
            case "2":
                epochs = available_epochs
            case "3":
                max_agents = input("Número máximo de participantes [16]: ") or "16"
                try:
                    max_agents = int(max_agents)
                    if len(available_epochs) > max_agents:
                        indices = np.round(np.linspace(0, len(available_epochs) - 1, max_agents)).astype(int)
                        epochs = [available_epochs[i] for i in indices]
                    else:
                        epochs = available_epochs
                except ValueError:
                    print("Valor inválido, usando todas las épocas disponibles.")
                    epochs = available_epochs
            case _:
                # Selección manual
                epochs_input = input("Ingrese las épocas separadas por espacios: ")
                try:
                    epochs = [int(e) for e in epochs_input.split()]
                    invalid_epochs = [e for e in epochs if e not in available_epochs]
                    if invalid_epochs:
                        print(f"Épocas no disponibles: {invalid_epochs}")
                        epochs = [e for e in epochs if e in available_epochs]
                except ValueError:
                    print("Entrada inválida. Usando las primeras 8 épocas.")
                    epochs = available_epochs[:8]

        if len(epochs) < 2:
            print("Se necesitan al menos 2 épocas. Usando las primeras 2 disponibles.")
            epochs = available_epochs[:2]

        # Mostrar información del bracket
        bracket_size = get_next_power_of_2(len(epochs))
        print(f"\nÉpocas seleccionadas: {epochs}")
        print(f"Tamaño del bracket: {bracket_size} participantes")

        if bracket_size > len(epochs):
            print(f"Se añadirá(n) {bracket_size - len(epochs)} BYE(s) para completar el bracket")

        # Otros parámetros
        while True:
            try:
                n_matches = int(input("\nNúmero de partidas por enfrentamiento [10]: ") or "10")
                if n_matches > 0:
                    break
                print("Error: Debe ser mayor que 0.")
            except ValueError:
                print("Error: Ingrese un número válido.")

        while True:
            try:
                temperature = float(input("Temperatura para los agentes (0.1-1.0) [0.5]: ") or "0.5")
                if 0 < temperature <= 1:
                    break
                print("Error: Debe estar entre 0.1 y 1.0.")
            except ValueError:
                print("Error: Ingrese un número válido.")

        visualize = input("\n¿Generar visualizaciones? (s/n) [s]: ").lower() or "s"
        visualize = visualize in ["s", "si", "sí", "y", "yes"]

        # Resumen
        print(f"\n===== RESUMEN DEL TORNEO =====")
        print(f"Formato: Eliminación directa")
        print(f"Participantes: {len(epochs)} épocas")
        print(f"Tamaño del bracket: {bracket_size}")
        print(f"Partidas por enfrentamiento: {n_matches}")
        print(f"Temperatura: {temperature}")
        print(f"Visualizaciones: {'Sí' if visualize else 'No'}")

        confirm = input("\n¿Iniciar el torneo? (s/n) [s]: ").lower() or "s"
        if confirm in ["s", "si", "sí", "y", "yes"]:
            print("\n🏆 Iniciando torneo de eliminación directa...\n")
            run_bracket_tournament(
                epochs=epochs,
                n_matches=n_matches,
                temperature=temperature,
                visualize=visualize
            )
        else:
            print("Torneo cancelado.")

if __name__ == "__main__":
    # Configurar para multiprocessing en Windows
    mp.freeze_support()
    main()
