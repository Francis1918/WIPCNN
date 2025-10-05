#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
tournament.py - Torneo "todos contra todos" para agentes de Quarto.
Enfrenta a múltiples agentes de diferentes épocas entre sí y determina al campeón.
Sistema de clasificación: Bradley-Terry + ELO (análisis estadístico avanzado)

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
from tabulate import tabulate

# Importar funciones desde compare_agents.py (sin modificarlo)
from compare_agents import load_agent, show_available_models, compare_agents
from utils.logger import logger


class BradleyTerryELO:
    """
    Sistema híbrido que usa Bradley-Terry para calcular fuerzas y las presenta como ratings ELO.
    Bradley-Terry domina el cálculo, ELO solo sirve para presentación amigable.
    """

    def __init__(self, agents, initial_rating=1500):
        self.agents = list(agents)
        self.n_agents = len(self.agents)

        # Fuerzas Bradley-Terry (π_i) - estas son las que dominan el sistema
        self.pi_forces = {agent: 1.0 for agent in self.agents}

        # Ratings ELO para presentación (convertidos desde π_i)
        self.elo_ratings = {agent: initial_rating for agent in self.agents}
        self.initial_rating = initial_rating

        # Historial de enfrentamientos para Bradley-Terry
        self.match_results = []  # Lista de (agente1, agente2, wins1, wins2, draws)

        # Historial de ratings para visualización
        self.rating_history = {agent: [initial_rating] for agent in self.agents}

        # Matriz de enfrentamientos para estadísticas
        self.wins_matrix = pd.DataFrame(0, index=self.agents, columns=self.agents)
        self.total_games_matrix = pd.DataFrame(0, index=self.agents, columns=self.agents)

    def update_after_match(self, agent1, agent2, wins1, wins2, draws):
        """
        Actualiza el sistema después de un enfrentamiento.
        Recalcula TODAS las fuerzas π usando Bradley-Terry y convierte a ELO.
        """
        # Registrar el enfrentamiento
        self.match_results.append((agent1, agent2, wins1, wins2, draws))

        # Actualizar matrices de estadísticas
        self.wins_matrix.loc[agent1, agent2] += wins1
        self.wins_matrix.loc[agent2, agent1] += wins2

        total_games = wins1 + wins2 + draws
        self.total_games_matrix.loc[agent1, agent2] += total_games
        self.total_games_matrix.loc[agent2, agent1] += total_games

        # Recalcular TODAS las fuerzas π usando Bradley-Terry
        self._recalculate_bradley_terry_forces()

        # Convertir fuerzas π a ratings ELO para presentación
        self._convert_forces_to_elo()

        # Guardar en historial
        for agent in self.agents:
            self.rating_history[agent].append(self.elo_ratings[agent])

    def _recalculate_bradley_terry_forces(self):
        """
        Recalcula las fuerzas Bradley-Terry usando Maximum Likelihood Estimation.
        Este es el corazón del sistema - Bradley-Terry domina aquí.
        """
        if not self.match_results:
            return

        # Algoritmo iterativo para Bradley-Terry MLE
        pi = np.ones(self.n_agents)  # Inicializar todas las fuerzas en 1

        for iteration in range(100):  # Máximo 100 iteraciones
            pi_new = np.zeros(self.n_agents)

            for i, agent_i in enumerate(self.agents):
                wins_i = 0
                denominator = 0

                for j, agent_j in enumerate(self.agents):
                    if i != j:
                        # Obtener resultados entre agente_i y agente_j
                        wins_ij = self.wins_matrix.loc[agent_i, agent_j]
                        total_games = self.total_games_matrix.loc[agent_i, agent_j]

                        if total_games > 0:
                            wins_i += wins_ij
                            denominator += total_games / (pi[i] + pi[j])

                if denominator > 0:
                    pi_new[i] = wins_i / denominator
                else:
                    pi_new[i] = 1.0

            # Normalizar para evitar que las fuerzas crezcan indefinidamente
            pi_new = pi_new / np.sum(pi_new) * self.n_agents

            # Verificar convergencia
            if np.allclose(pi, pi_new, rtol=1e-6):
                break

            pi = pi_new

        # Actualizar las fuerzas
        for i, agent in enumerate(self.agents):
            self.pi_forces[agent] = pi[i]

    def _convert_forces_to_elo(self):
        """
        Convierte las fuerzas Bradley-Terry a ratings ELO para presentación amigable.
        """
        # Escala logarítmica: ELO = base + scale * log(π_i)
        base_rating = self.initial_rating
        scale_factor = 200  # Factor de escala para hacer ratings interpretables

        for agent in self.agents:
            pi_force = self.pi_forces[agent]
            if pi_force > 0:
                self.elo_ratings[agent] = base_rating + scale_factor * np.log(pi_force)
            else:
                self.elo_ratings[agent] = base_rating - 500  # Rating muy bajo para π = 0

    def get_win_probability(self, agent1, agent2):
        """
        Calcula la probabilidad de que agent1 venza a agent2 usando Bradley-Terry.
        """
        pi1 = self.pi_forces[agent1]
        pi2 = self.pi_forces[agent2]
        return pi1 / (pi1 + pi2)

    def get_results_table(self):
        """
        Genera una tabla elegante de resultados usando pandas con formato profesional.
        """
        # Calcular estadísticas
        stats_data = []

        for agent in self.agents:
            # Estadísticas básicas
            total_wins = self.wins_matrix.loc[agent].sum()
            total_losses = self.wins_matrix[agent].sum()

            # Calcular empates
            total_draws = 0
            for other_agent in self.agents:
                if agent != other_agent:
                    total_games = self.total_games_matrix.loc[agent, other_agent]
                    wins = self.wins_matrix.loc[agent, other_agent]
                    losses = self.wins_matrix.loc[other_agent, agent]
                    total_draws += max(0, total_games - wins - losses)

            # Puntos tradicionales
            points = total_wins * 3 + total_draws * 1

            # Rating change
            initial_rating = self.rating_history[agent][0]
            current_rating = self.elo_ratings[agent]
            rating_change = current_rating - initial_rating

            # Probabilidad promedio contra otros agentes
            avg_win_prob = 0
            valid_opponents = 0
            for other_agent in self.agents:
                if agent != other_agent:
                    avg_win_prob += self.get_win_probability(agent, other_agent)
                    valid_opponents += 1

            if valid_opponents > 0:
                avg_win_prob = avg_win_prob / valid_opponents * 100

            stats_data.append({
                'Agente': f'Época {agent}',
                'BT-Force': self.pi_forces[agent],
                'ELO-Rating': int(round(self.elo_ratings[agent])),
                'Rating Δ': int(round(rating_change)),
                'Victorias': int(total_wins),
                'Derrotas': int(total_losses),
                'Empates': int(total_draws),
                'Puntos': int(points),
                'vs Promedio': f'{avg_win_prob:.1f}%'
            })

        # Crear DataFrame
        results_df = pd.DataFrame(stats_data)

        # Ordenar por BT-Force (dominante) y luego por ELO-Rating
        results_df = results_df.sort_values(['BT-Force', 'ELO-Rating'], ascending=[False, False])

        # Agregar posición
        results_df['Posición'] = range(1, len(results_df) + 1)

        # Reordenar columnas para mejor presentación
        column_order = ['Posición', 'Agente', 'BT-Force', 'ELO-Rating', 'Rating Δ',
                       'Victorias', 'Derrotas', 'Empates', 'Puntos', 'vs Promedio']
        results_df = results_df[column_order]

        return results_df

    def get_probability_matrix(self):
        """
        Genera una matriz de probabilidades de victoria usando Bradley-Terry.
        """
        prob_matrix = pd.DataFrame(index=self.agents, columns=self.agents, dtype=float)

        for agent1 in self.agents:
            for agent2 in self.agents:
                if agent1 == agent2:
                    prob_matrix.loc[agent1, agent2] = 0.5  # Empate consigo mismo
                else:
                    prob_matrix.loc[agent1, agent2] = self.get_win_probability(agent1, agent2)

        return prob_matrix

    def print_elegant_table(self):
        """
        Imprime una tabla elegante usando tabulate con formato profesional.
        """
        results_df = self.get_results_table()

        # Formatear la tabla para impresión
        headers = results_df.columns.tolist()
        data = results_df.values.tolist()

        # Crear tabla con formato elegante
        table = tabulate(
            data,
            headers=headers,
            tablefmt='grid',
            numalign='center',
            stralign='center'
        )

        return table
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

def display_pandas_scoreboard(bt_elo_system):
    """
    Muestra una tabla de puntuaciones elegante usando pandas con formato profesional.
    Sistema único: Bradley-Terry + ELO
    """
    print("\n" + "=" * 80)
    print("📊 TABLA DE PUNTUACIONES FINAL (BRADLEY-TERRY + ELO)")
    print("=" * 80)

    # Tabla principal Bradley-Terry + ELO
    results_df = bt_elo_system.get_results_table()

    # Personalizar la tabla para mejor visualización
    display_df = results_df.copy()

    # Formatear columnas numéricas para mejor presentación
    display_df['BT-Force'] = display_df['BT-Force'].apply(lambda x: f"{x:.3f}")
    display_df['Rating Δ'] = display_df['Rating Δ'].apply(lambda x: f"{x:+d}" if x != 0 else "0")

    # Mostrar tabla con pandas estilizada
    print("\n🏆 CLASIFICACIÓN FINAL:")
    print("-" * 80)

    # Configurar pandas para mejor visualización en terminal
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    print(display_df.to_string(index=False, justify='center'))

    # Mostrar estadísticas del podio
    champion_row = results_df.iloc[0]
    print(f"\n🥇 CAMPEÓN: {champion_row['Agente']}")
    print(f"   • BT-Force: {champion_row['BT-Force']:.3f}")
    print(f"   • ELO Rating: {champion_row['ELO-Rating']}")
    print(f"   • Probabilidad promedio de victoria: {champion_row['vs Promedio']}")
    print(f"   • Record: {champion_row['Victorias']}V-{champion_row['Derrotas']}D-{champion_row['Empates']}E")

    if len(results_df) > 1:
        runner_up = results_df.iloc[1]
        print(f"\n🥈 SUBCAMPEÓN: {runner_up['Agente']}")
        print(f"   • BT-Force: {runner_up['BT-Force']:.3f}")
        print(f"   • ELO Rating: {runner_up['ELO-Rating']}")
        print(f"   • Probabilidad promedio de victoria: {runner_up['vs Promedio']}")

    if len(results_df) > 2:
        third_place = results_df.iloc[2]
        print(f"\n🥉 TERCER LUGAR: {third_place['Agente']}")
        print(f"   • BT-Force: {third_place['BT-Force']:.3f}")
        print(f"   • ELO Rating: {third_place['ELO-Rating']}")
        print(f"   • Probabilidad promedio de victoria: {third_place['vs Promedio']}")

    # Mostrar análisis estadístico de las fuerzas Bradley-Terry
    forces = results_df['BT-Force'].astype(float)
    ratings = results_df['ELO-Rating'].astype(int)

    print(f"\n📈 ANÁLISIS ESTADÍSTICO:")
    print(f"   • Fuerza BT máxima: {forces.max():.3f} ({results_df.iloc[0]['Agente']})")
    print(f"   • Fuerza BT mínima: {forces.min():.3f} ({results_df.iloc[-1]['Agente']})")
    print(f"   • Promedio de fuerzas: {forces.mean():.3f}")
    print(f"   • Desviación estándar: {forces.std():.3f}")
    print(f"   • Ratio dominancia: {forces.max()/forces.min():.2f}x")

    print(f"\n⭐ ANÁLISIS DE RATINGS ELO:")
    print(f"   • Rating más alto: {ratings.max()} ({results_df.iloc[0]['Agente']})")
    print(f"   • Rating más bajo: {ratings.min()} ({results_df.iloc[-1]['Agente']})")
    print(f"   • Diferencia de rating: {ratings.max() - ratings.min()} puntos")
    print(f"   • Promedio de ratings: {ratings.mean():.0f}")

    print("\n" + "=" * 80)


def run_tournament(epochs, n_matches=10, temperature=0.5, visualize=False):
    """Ejecuta un torneo completo usando el sistema Bradley-Terry + ELO.

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

    # Crear estructura de directorios organizada
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Carpeta principal de torneos
    main_tournaments_dir = "tournaments"
    os.makedirs(main_tournaments_dir, exist_ok=True)

    # Carpeta específica para este torneo
    tournament_dir = f"{main_tournaments_dir}/tournament_{timestamp}"
    os.makedirs(tournament_dir, exist_ok=True)

    # Subcarpetas organizadas
    results_dir = f"{tournament_dir}/results"
    matches_dir = f"{tournament_dir}/matches"
    vis_dir = f"{tournament_dir}/visualizations"

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(matches_dir, exist_ok=True)
    if visualize:
        os.makedirs(vis_dir, exist_ok=True)

    # Inicializar sistema Bradley-Terry-ELO
    bt_elo_system = BradleyTerryELO(agents=epochs)
    logger.info("Usando sistema de rating Bradley-Terry + ELO")

    # Matriz para almacenar resultados de enfrentamientos directos
    matches_data = []

    # Progreso
    total_matches = len(list(itertools.combinations(epochs, 2)))
    logger.info(f"Iniciando torneo con {len(epochs)} agentes")
    logger.info(f"Total de enfrentamientos: {total_matches}")
    logger.info(f"Partidas por enfrentamiento: {n_matches}")
    logger.info(f"Resultados se guardarán en: {tournament_dir}")

    # Todos contra todos
    for epoch1, epoch2 in tqdm(itertools.combinations(epochs, 2), total=total_matches, desc="Enfrentamientos"):
        logger.info(f"\n{'=' * 30}")
        logger.info(f"Enfrentamiento: Época {epoch1} vs Época {epoch2}")

        # Directorio específico para este enfrentamiento
        match_dir = f"{matches_dir}/match_{epoch1}_vs_{epoch2}"
        os.makedirs(match_dir, exist_ok=True)

        # Realizar el enfrentamiento
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

        # Actualizar sistema Bradley-Terry-ELO
        bt_elo_system.update_after_match(epoch1, epoch2, wins_1, wins_2, draws)

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

    # Mostrar resultados del torneo
    logger.info("\n" + "=" * 70)
    logger.info("RESULTADOS DEL TORNEO (SISTEMA BRADLEY-TERRY + ELO)")
    logger.info("=" * 70)

    # Tabla principal con tabulate
    bt_elo_table = bt_elo_system.print_elegant_table()
    logger.info("\n" + bt_elo_table)

    # Obtener el campeón
    bt_results_df = bt_elo_system.get_results_table()
    champion = bt_results_df.iloc[0]['Agente']
    champion_epoch = int(champion.split(' ')[1])
    logger.info(f"\n🏆 ¡El CAMPEÓN del torneo es el agente de la {champion}!")
    logger.info(f"   BT-Force: {bt_elo_system.pi_forces[champion_epoch]:.3f}")
    logger.info(f"   ELO Rating: {bt_elo_system.elo_ratings[champion_epoch]:.0f}")

    # Mostrar matriz de probabilidades
    logger.info("\n" + "=" * 50)
    logger.info("MATRIZ DE PROBABILIDADES DE VICTORIA")
    logger.info("=" * 50)
    prob_matrix = bt_elo_system.get_probability_matrix()
    prob_matrix_display = prob_matrix.copy()

    # Formatear probabilidades como porcentajes
    for col in prob_matrix_display.columns:
        prob_matrix_display[col] = prob_matrix_display[col].apply(
            lambda x: f"{x*100:.1f}%" if x != 0.5 else "N/A"
        )

    # Renombrar índices y columnas para mejor presentación
    prob_matrix_display.index = [f"E{epoch}" for epoch in prob_matrix_display.index]
    prob_matrix_display.columns = [f"E{epoch}" for epoch in prob_matrix_display.columns]

    logger.info(f"\n{prob_matrix_display.to_string()}")

    # Mostrar tabla final con pandas
    display_pandas_scoreboard(bt_elo_system)

    # Si se solicitó visualización, crear gráficos
    if visualize:
        # Crear gráficos específicos para BT-ELO
        create_bt_elo_charts(bt_elo_system, vis_dir)

    # Guardar todos los resultados en archivos
    bt_results_df.to_csv(f"{results_dir}/bt_elo_results.csv", index=False)
    prob_matrix.to_csv(f"{results_dir}/probability_matrix.csv")
    pd.DataFrame(matches_data).to_csv(f"{results_dir}/matches_detail.csv", index=False)

    # Guardar resumen completo en texto
    summary_file = f"{tournament_dir}/RESUMEN_TORNEO.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RESUMEN DEL TORNEO - SISTEMA BRADLEY-TERRY + ELO\n")
        f.write("=" * 80 + "\n\n")

        # Información general
        f.write(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Número de agentes: {len(epochs)}\n")
        f.write(f"Épocas participantes: {epochs}\n")
        f.write(f"Total de enfrentamientos: {total_matches}\n")
        f.write(f"Partidas por enfrentamiento: {n_matches}\n")
        f.write(f"Temperatura: {temperature}\n")
        f.write(f"Visualización habilitada: {'Sí' if visualize else 'No'}\n")
        f.write("\n" + "=" * 80 + "\n\n")

        # Clasificación final
        f.write("🏆 CLASIFICACIÓN FINAL\n")
        f.write("=" * 80 + "\n\n")
        f.write(bt_results_df.to_string(index=False))
        f.write("\n\n" + "=" * 80 + "\n\n")

        # Podio
        f.write("🥇 PODIO\n")
        f.write("=" * 80 + "\n\n")

        champion_row = bt_results_df.iloc[0]
        f.write(f"🥇 CAMPEÓN: {champion_row['Agente']}\n")
        f.write(f"   • BT-Force: {champion_row['BT-Force']:.3f}\n")
        f.write(f"   • ELO Rating: {champion_row['ELO-Rating']}\n")
        f.write(f"   • Probabilidad promedio de victoria: {champion_row['vs Promedio']}\n")
        f.write(f"   • Record: {champion_row['Victorias']}V-{champion_row['Derrotas']}D-{champion_row['Empates']}E\n\n")

        if len(bt_results_df) > 1:
            runner_up = bt_results_df.iloc[1]
            f.write(f"🥈 SUBCAMPEÓN: {runner_up['Agente']}\n")
            f.write(f"   • BT-Force: {runner_up['BT-Force']:.3f}\n")
            f.write(f"   • ELO Rating: {runner_up['ELO-Rating']}\n")
            f.write(f"   • Probabilidad promedio de victoria: {runner_up['vs Promedio']}\n\n")

        if len(bt_results_df) > 2:
            third_place = bt_results_df.iloc[2]
            f.write(f"🥉 TERCER LUGAR: {third_place['Agente']}\n")
            f.write(f"   • BT-Force: {third_place['BT-Force']:.3f}\n")
            f.write(f"   • ELO Rating: {third_place['ELO-Rating']}\n")
            f.write(f"   • Probabilidad promedio de victoria: {third_place['vs Promedio']}\n\n")

        f.write("=" * 80 + "\n\n")

        # Estadísticas
        forces = bt_results_df['BT-Force'].astype(float)
        ratings = bt_results_df['ELO-Rating'].astype(int)

        f.write("📈 ANÁLISIS ESTADÍSTICO\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Fuerza BT máxima: {forces.max():.3f} ({bt_results_df.iloc[0]['Agente']})\n")
        f.write(f"Fuerza BT mínima: {forces.min():.3f} ({bt_results_df.iloc[-1]['Agente']})\n")
        f.write(f"Promedio de fuerzas: {forces.mean():.3f}\n")
        f.write(f"Desviación estándar: {forces.std():.3f}\n")
        f.write(f"Ratio dominancia: {forces.max()/forces.min():.2f}x\n\n")

        f.write("⭐ ANÁLISIS DE RATINGS ELO\n")
        f.write("-" * 80 + "\n\n")
        f.write(f"Rating más alto: {ratings.max()} ({bt_results_df.iloc[0]['Agente']})\n")
        f.write(f"Rating más bajo: {ratings.min()} ({bt_results_df.iloc[-1]['Agente']})\n")
        f.write(f"Diferencia de rating: {ratings.max() - ratings.min()} puntos\n")
        f.write(f"Promedio de ratings: {ratings.mean():.0f}\n\n")

        f.write("=" * 80 + "\n\n")

        # Matriz de probabilidades
        f.write("📊 MATRIZ DE PROBABILIDADES DE VICTORIA\n")
        f.write("=" * 80 + "\n\n")
        f.write(prob_matrix_display.to_string())
        f.write("\n\n" + "=" * 80 + "\n\n")

        # Enfrentamientos detallados
        f.write("🎯 DETALLES DE ENFRENTAMIENTOS\n")
        f.write("=" * 80 + "\n\n")
        matches_df = pd.DataFrame(matches_data)
        f.write(matches_df.to_string(index=False))
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("Fin del resumen\n")
        f.write("=" * 80 + "\n")

    logger.info(f"\n✅ Resultados guardados exitosamente en: {tournament_dir}")
    logger.info(f"   📁 Carpeta principal: {tournament_dir}")
    logger.info(f"   📊 Resultados: {results_dir}")
    logger.info(f"   🎮 Enfrentamientos: {matches_dir}")
    if visualize:
        logger.info(f"   📈 Visualizaciones: {vis_dir}")
    logger.info(f"   📄 Resumen completo: {summary_file}")

    return bt_results_df


def create_bt_elo_charts(bt_elo_system, vis_dir):
    """Crea gráficos específicos para el sistema Bradley-Terry-ELO."""
    results_df = bt_elo_system.get_results_table()

    # Gráfico de ratings ELO
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_df)))
    bars = plt.bar(results_df['Agente'], results_df['ELO-Rating'], color=colors)
    plt.title('Ratings ELO por agente (Sistema Bradley-Terry + ELO)')
    plt.xlabel('Agente')
    plt.ylabel('Rating ELO')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)

    # Agregar valores en las barras
    for bar, rating in zip(bars, results_df['ELO-Rating']):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                f'{rating}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{vis_dir}/bt_elo_ratings.png")
    plt.close()

    # Gráfico de fuerzas Bradley-Terry
    plt.figure(figsize=(12, 6))
    colors = plt.cm.plasma(np.linspace(0, 1, len(results_df)))
    bars = plt.bar(results_df['Agente'], results_df['BT-Force'], color=colors)
    plt.title('Fuerzas Bradley-Terry por agente')
    plt.xlabel('Agente')
    plt.ylabel('Fuerza π (Bradley-Terry)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)

    # Agregar valores en las barras
    for bar, force in zip(bars, results_df['BT-Force']):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{force:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{vis_dir}/bt_forces.png")
    plt.close()

    # Gráfico comparativo: BT-Force vs Puntos tradicionales
    plt.figure(figsize=(12, 8))

    # Subplot 1: BT-Force
    plt.subplot(2, 1, 1)
    plt.bar(results_df['Agente'], results_df['BT-Force'], color='orange', alpha=0.7)
    plt.title('Comparación: Fuerzas Bradley-Terry vs Puntos Tradicionales')
    plt.ylabel('Fuerza BT')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)

    # Subplot 2: Puntos tradicionales
    plt.subplot(2, 1, 2)
    plt.bar(results_df['Agente'], results_df['Puntos'], color='blue', alpha=0.7)
    plt.ylabel('Puntos Tradicionales')
    plt.xlabel('Agente')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{vis_dir}/bt_vs_traditional_comparison.png")
    plt.close()

    # Matriz de probabilidades como heatmap
    prob_matrix = bt_elo_system.get_probability_matrix()

    plt.figure(figsize=(10, 8))
    plt.imshow(prob_matrix.values, cmap='RdYlBu_r', interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar(label='Probabilidad de victoria')
    plt.title('Matriz de Probabilidades Bradley-Terry')
    plt.xlabel('Oponente (época)')
    plt.ylabel('Agente (época)')

    # Configurar etiquetas de ejes
    epochs = prob_matrix.index.tolist()
    plt.xticks(np.arange(len(epochs)), [f'E{e}' for e in epochs], rotation=45)
    plt.yticks(np.arange(len(epochs)), [f'E{e}' for e in epochs])

    # Mostrar valores en las celdas
    for i in range(len(epochs)):
        for j in range(len(epochs)):
            if i != j:  # No mostrar probabilidades diagonales
                prob = prob_matrix.iloc[i, j]
                plt.text(j, i, f'{prob:.3f}',
                        ha='center', va='center',
                        color='white' if prob > 0.5 else 'black',
                        fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{vis_dir}/bt_probability_matrix.png")
    plt.close()
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

        # Ejecutar torneo con el sistema seleccionado
        results = run_tournament(epochs, args.matches, args.temp, args.visualize)

        if results is not None:
            logger.info("¡Torneo completado exitosamente!")
        else:
            logger.error("Error al ejecutar el torneo")

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
        print(f"Sistema de rating: Bradley-Terry + ELO")
        print(f"Visualizar resultados: {'Sí' if visualize else 'No'}")

        confirm = input("\n¿Iniciar el torneo con estos parámetros? (s/n) [s]: ").lower() or "s"
        if confirm in ["s", "si", "sí", "y", "yes"]:
            print("\nIniciando torneo...\n")
            results = run_tournament(epochs, n_matches, temperature, visualize)

            if results is not None:
                print("\n🎉 ¡Torneo completado exitosamente!")
                print("📊 Los resultados incluyen:")
                print("   - Tabla principal con sistema Bradley-Terry + ELO")
                print("   - Matriz de probabilidades de victoria")
                print("   - Gráficos avanzados (si se habilitaron)")
            else:
                print("❌ Error al ejecutar el torneo")
        else:
            print("Torneo cancelado por el usuario.")

if __name__ == "__main__":
    main()
