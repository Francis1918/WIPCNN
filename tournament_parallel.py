#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
tournament_parallel.py - Versión paralela del torneo "todos contra todos" para agentes de Quarto.
Utiliza multiprocesamiento para acelerar la ejecución de los enfrentamientos entre agentes.

Uso:
    python tournament_parallel.py [--epochs E1 E2 E3...] [--matches N] [--temp T] [--visualize] [--workers W]
    python tournament_parallel.py [--physical-only]

Ejemplos:
    python tournament_parallel.py                                # Modo interactivo
    python tournament_parallel.py --epochs 1 50 100 150 200      # Torneo con épocas específicas
    python tournament_parallel.py --all --workers 4              # Usar todas las épocas y 4 trabajadores
    python tournament_parallel.py --physical-only                # Usar solo núcleos físicos
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
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import platform
import subprocess
import scipy.optimize

# Importar funciones desde compare_agents.py
from compare_agents import compare_agents
from utils.logger import logger

def compute_bradley_terry_skills(results_df, epochs):
    """
    Calcula las habilidades Bradley-Terry de los agentes usando optimización iterativa.

    Args:
        results_df (pd.DataFrame): DataFrame con los resultados de los enfrentamientos
        epochs (list): Lista de épocas/agentes

    Returns:
        dict: Diccionario con las habilidades estimadas por época
    """
    try:
        n_agents = len(epochs)
        epoch_to_idx = {epoch: i for i, epoch in enumerate(epochs)}

        # Construir matriz de comparaciones
        wins = np.zeros((n_agents, n_agents))

        for i, epoch_i in enumerate(epochs):
            for j, epoch_j in enumerate(epochs):
                if i != j:
                    wins[i, j] = results_df.at[epoch_i, epoch_j]

        # Función objetivo para Bradley-Terry
        def objective(skills):
            """Función de log-verosimilitud negativa para Bradley-Terry"""
            log_likelihood = 0
            for i in range(n_agents):
                for j in range(n_agents):
                    if i != j and (wins[i, j] + wins[j, i]) > 0:
                        # Probabilidad de que i gane contra j
                        prob_i_beats_j = np.exp(skills[i]) / (np.exp(skills[i]) + np.exp(skills[j]))

                        # Evitar log(0)
                        prob_i_beats_j = max(prob_i_beats_j, 1e-15)
                        prob_i_beats_j = min(prob_i_beats_j, 1 - 1e-15)

                        # Agregar a la log-verosimilitud
                        log_likelihood += wins[i, j] * np.log(prob_i_beats_j)

            return -log_likelihood  # Negativo porque minimizamos

        # Optimizar con restricción de que la suma de habilidades sea 0
        def constraint(skills):
            return np.sum(skills)

        # Punto inicial aleatorio
        initial_skills = np.random.normal(0, 1, n_agents)
        initial_skills -= np.mean(initial_skills)  # Centrar en 0

        # Optimización con restricciones
        result = scipy.optimize.minimize(
            objective,
            initial_skills,
            method='SLSQP',
            constraints={'type': 'eq', 'fun': constraint},
            options={'maxiter': 1000}
        )

        if result.success:
            skills = result.x
            # Convertir a diccionario con épocas como claves
            skills_dict = {epochs[i]: skills[i] for i in range(n_agents)}
            return skills_dict
        else:
            logger.warning("La optimización Bradley-Terry no convergió, usando método alternativo")
            return compute_bradley_terry_simple(results_df, epochs)

    except Exception as e:
        logger.warning(f"Error en Bradley-Terry optimizado: {e}, usando método simple")
        return compute_bradley_terry_simple(results_df, epochs)

def compute_bradley_terry_simple(results_df, epochs):
    """
    Método simple de Bradley-Terry usando iteración de punto fijo.

    Args:
        results_df (pd.DataFrame): DataFrame con los resultados
        epochs (list): Lista de épocas

    Returns:
        dict: Habilidades estimadas por época
    """
    try:
        n_agents = len(epochs)

        # Inicializar habilidades
        skills = {epoch: 1.0 for epoch in epochs}

        # Iteración de punto fijo
        for iteration in range(100):  # Máximo 100 iteraciones
            new_skills = {}

            for epoch_i in epochs:
                numerator = 0
                denominator = 0

                for epoch_j in epochs:
                    if epoch_i != epoch_j:
                        wins_i = results_df.at[epoch_i, epoch_j]
                        wins_j = results_df.at[epoch_j, epoch_i]
                        total_games = wins_i + wins_j

                        if total_games > 0:
                            numerator += wins_i
                            denominator += total_games * skills[epoch_j] / (skills[epoch_i] + skills[epoch_j])

                if denominator > 0:
                    new_skills[epoch_i] = numerator / denominator
                else:
                    new_skills[epoch_i] = 1.0

            # Normalizar para evitar crecimiento ilimitado
            total_skill = sum(new_skills.values())
            if total_skill > 0:
                for epoch in epochs:
                    new_skills[epoch] = new_skills[epoch] * len(epochs) / total_skill

            # Verificar convergencia
            converged = True
            for epoch in epochs:
                if abs(new_skills[epoch] - skills[epoch]) > 1e-6:
                    converged = False
                    break

            skills = new_skills

            if converged:
                logger.info(f"Bradley-Terry convergió en {iteration + 1} iteraciones")
                break

        # Convertir a escala logarítmica para mejor interpretación
        log_skills = {}
        for epoch in epochs:
            log_skills[epoch] = np.log(skills[epoch])

        # Centrar en 0
        mean_log_skill = np.mean(list(log_skills.values()))
        for epoch in epochs:
            log_skills[epoch] -= mean_log_skill

        return log_skills

    except Exception as e:
        logger.error(f"Error en Bradley-Terry simple: {e}")
        # Retornar habilidades neutras si todo falla
        return {epoch: 0.0 for epoch in epochs}

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

def run_match_parallel(args):
    """Función para ejecutar un enfrentamiento individual en un proceso separado.

    Args:
        args (tuple): Tupla con los parámetros:
            - epoch1 (int): Época del primer agente
            - epoch2 (int): Época del segundo agente
            - n_matches (int): Número de partidas
            - temperature (float): Temperatura para los agentes
            - visualize (bool): Si se debe visualizar
            - tournament_dir (str): Directorio para guardar resultados

    Returns:
        tuple: (epoch1, epoch2, match_results, match_data)
    """
    epoch1, epoch2, n_matches, temperature, visualize, tournament_dir = args

    match_start = time.time()
    process_id = os.getpid()

    print(f"[Proceso {process_id}] Iniciando enfrentamiento: Época {epoch1} vs Época {epoch2}")

    # Directorio específico para este enfrentamiento
    match_dir = f"{tournament_dir}/match_{epoch1}_vs_{epoch2}"
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
        print(f"[Proceso {process_id}] No se obtuvieron resultados para {epoch1} vs {epoch2}")
        return epoch1, epoch2, None, None

    # Extraer resultados
    wins_1 = match_results['P1']
    wins_2 = match_results['P2']
    draws = match_results['Empates']

    # Guardar detalles del enfrentamiento
    match_data = {
        'Epoch1': epoch1,
        'Epoch2': epoch2,
        'Wins_Epoch1': wins_1,
        'Wins_Epoch2': wins_2,
        'Draws': draws,
        'Win_Rate_Epoch1': wins_1 / n_matches * 100,
        'Win_Rate_Epoch2': wins_2 / n_matches * 100,
        'Draw_Rate': draws / n_matches * 100,
        'Duration': time.time() - match_start
    }

    print(f"[Proceso {process_id}] Completado: Época {epoch1} vs Época {epoch2} - Resultado: {wins_1}-{wins_2}-{draws}")

    return epoch1, epoch2, match_results, match_data

def initialize_worker(cores):
    """Inicializa un proceso trabajador estableciendo la afinidad de CPU si se especifica.

    Args:
        cores (list): Lista de núcleos específicos a los que debe asignarse el proceso
    """
    if cores is not None:
        # Solo en plataformas que soportan afinidad de CPU
        try:
            import os
            import psutil
            process = psutil.Process(os.getpid())

            # Establecer afinidad de CPU para este proceso
            process.cpu_affinity(cores)

            # Configurar prioridad alta para el proceso
            if hasattr(process, 'nice'):  # Linux/macOS
                try:
                    process.nice(-10)  # Prioridad más alta en Unix/Linux (-20 a 19, -20 es la más alta)
                except psutil.AccessDenied:
                    logger.warning("No se pudo establecer prioridad alta (se requieren privilegios de administrador)")
            elif hasattr(process, 'nice'):  # Windows
                try:
                    import win32process
                    import win32con
                    handle = win32process.GetCurrentProcess()
                    win32process.SetPriorityClass(handle, win32con.HIGH_PRIORITY_CLASS)
                except (ImportError, AttributeError):
                    # Intentar método alternativo para Windows
                    try:
                        process.nice(psutil.HIGH_PRIORITY_CLASS)
                    except (AttributeError, psutil.AccessDenied):
                        logger.warning("No se pudo establecer prioridad alta")

            # Obtener y registrar la afinidad actual para verificación
            current_affinity = process.cpu_affinity()
            logger.info(f"Proceso {os.getpid()} iniciado con afinidad a núcleos: {current_affinity}")

        except (ImportError, AttributeError, NotImplementedError) as e:
            logger.warning(f"No se pudo establecer la afinidad de CPU: {e}")

def run_tournament_parallel(epochs, n_matches=10, temperature=0.5, visualize=False, n_workers=None, physical_only=False, specific_cores=None):
    """Ejecuta un torneo completo de todos contra todos entre las épocas especificadas
    utilizando multiprocesamiento.

    Args:
        epochs (list): Lista de épocas a enfrentar
        n_matches (int): Número de partidas por enfrentamiento
        temperature (float): Temperatura para los agentes
        visualize (bool): Si se deben guardar visualizaciones
        n_workers (int): Número de trabajadores paralelos (default: número de CPUs)
        physical_only (bool): Si se deben usar solo núcleos físicos
        specific_cores (list): Lista de núcleos específicos a utilizar (ej: [0,1,2,5])

    Returns:
        pd.DataFrame: Tabla de resultados del torneo
    """
    if len(epochs) < 2:
        logger.error("Se necesitan al menos 2 épocas para un torneo")
        return None

    # Determinar número de trabajadores
    if specific_cores is not None:
        # Si se especificaron núcleos específicos, usar tantos trabajadores como núcleos especificados
        n_workers = len(specific_cores)
        cpu_affinity_str = f"específicos ({','.join(map(str, specific_cores))})"
    elif n_workers is None:
        # Si no se especificó, usar la función de detección de núcleos
        n_workers = get_cores_for_parallelism(physical_only)
        cpu_affinity_str = ""
    else:
        # Si se especificó manualmente el número de trabajadores
        cpu_affinity_str = ""

    # Obtener información de la CPU para el registro
    cpu_info = get_cpu_info()
    cpu_type_str = ""

    if specific_cores is not None:
        cpu_type_str = "específicos"
    elif physical_only:
        cpu_type_str = "físicos"
    else:
        cpu_type_str = "lógicos"

    # Crear estructura de directorios organizada
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Carpeta principal de torneos paralelos
    main_tournaments_dir = "tournaments_parallel"
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

    # Crear DataFrame para almacenar resultados
    results_df = pd.DataFrame(
        index=epochs,
        columns=epochs + ['Victorias', 'Derrotas', 'Empates', 'Puntos', 'Posición']
    )

    # Inicializar con ceros
    # Convertir el DataFrame a tipo numérico para evitar FutureWarning
    results_df = results_df.astype(float)
    results_df.fillna(0, inplace=True)

    # Matriz para almacenar resultados de enfrentamientos directos
    matches_data = []

    # Generar todas las combinaciones de enfrentamientos
    match_combinations = list(itertools.combinations(epochs, 2))
    total_matches = len(match_combinations)

    # Registrar información de CPU
    logger.info(f"Información de CPU: {cpu_info['logical_cores']} núcleos lógicos, {cpu_info['physical_cores']} núcleos físicos")

    logger.info(f"Iniciando torneo con {len(epochs)} agentes")
    logger.info(f"Total de enfrentamientos: {total_matches}")
    logger.info(f"Partidas por enfrentamiento: {n_matches}")
    logger.info(f"Resultados se guardarán en: {tournament_dir}")
    if specific_cores:
        logger.info(f"Utilizando {n_workers} núcleos {cpu_type_str} {cpu_affinity_str}")
    else:
        logger.info(f"Utilizando {n_workers} núcleos {cpu_type_str}")

    # Preparar argumentos para los procesos paralelos - actualizar con matches_dir
    match_args = [
        (epoch1, epoch2, n_matches, temperature, visualize, matches_dir)
        for epoch1, epoch2 in match_combinations
    ]

    start_time = time.time()
    completed_matches = 0

    # Ejecutar enfrentamientos en paralelo
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=initialize_worker if specific_cores else None,
        initargs=(specific_cores,) if specific_cores else ()
    ) as executor:
        # Iniciar todos los trabajos
        future_to_match = {
            executor.submit(run_match_parallel, args): args
            for args in match_args
        }

        # Mostrar progreso y procesar resultados a medida que se completan
        for future in tqdm(as_completed(future_to_match), total=len(future_to_match), desc="Enfrentamientos"):
            try:
                epoch1, epoch2, match_results, match_data = future.result()
                completed_matches += 1

                if match_results is None:
                    continue

                # Registrar resultados en el DataFrame
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
                matches_data.append(match_data)

                # Mostrar progreso en tiempo real
                elapsed = time.time() - start_time
                matches_per_second = completed_matches / elapsed
                estimated_total = elapsed * (total_matches / completed_matches) if completed_matches > 0 else 0
                remaining = estimated_total - elapsed

                logger.info(f"Progreso: {completed_matches}/{total_matches} enfrentamientos")
                logger.info(f"Velocidad: {matches_per_second:.2f} enfrentamientos/segundo")
                logger.info(f"Tiempo restante estimado: {remaining/60:.1f} minutos")

            except Exception as e:
                logger.error(f"Error en enfrentamiento: {e}")

    # Calcular posiciones finales
    positions = results_df['Puntos'].rank(method='min', ascending=False)
    results_df['Posición'] = positions

    # Ordenar por posición
    results_df = results_df.sort_values('Posición')

    # Estadísticas de tiempo
    total_time = time.time() - start_time
    time_per_match = total_time / total_matches if total_matches > 0 else 0

    # Mostrar tabla de resultados
    logger.info("\n" + "=" * 50)
    logger.info("RESULTADOS DEL TORNEO:")
    logger.info("=" * 50)
    logger.info(f"Tiempo total: {total_time/60:.2f} minutos")
    logger.info(f"Tiempo promedio por enfrentamiento: {time_per_match:.2f} segundos")
    logger.info(f"Aceleración estimada: {n_workers}x")

    # Tabla de posiciones
    position_table = results_df[['Victorias', 'Derrotas', 'Empates', 'Puntos', 'Posición']].sort_values('Posición')
    logger.info("\nTABLA DE POSICIONES:")
    logger.info("\n" + position_table.to_string())

    # Definir el campeón
    champion = position_table.index[0]
    logger.info(f"\n🏆 ¡El CAMPEÓN del torneo es el agente de la Época {champion}!")

    # Si se solicitó visualización, crear gráficos
    if visualize:
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

        # Graficar tiempo por enfrentamiento
        if matches_data:
            plt.figure(figsize=(12, 6))
            match_times = [match['Duration'] for match in matches_data]
            plt.hist(match_times, bins=20, color='purple', alpha=0.7)
            plt.axvline(x=np.mean(match_times), color='red', linestyle='--',
                        label=f'Promedio: {np.mean(match_times):.2f}s')
            plt.title('Distribución de tiempo por enfrentamiento')
            plt.xlabel('Tiempo (segundos)')
            plt.ylabel('Frecuencia')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{vis_dir}/match_durations.png")
            plt.close()

    # Guardar resultados en la carpeta results/
    position_table.to_csv(f"{results_dir}/positions.csv")
    results_df.to_csv(f"{results_dir}/full_results.csv")
    pd.DataFrame(matches_data).to_csv(f"{results_dir}/matches_detail.csv", index=False)

    # Guardar estadísticas de rendimiento
    performance_file = f"{results_dir}/performance_stats.txt"
    with open(performance_file, "w", encoding='utf-8') as f:
        f.write(f"Tiempo total: {total_time:.2f} segundos ({total_time/60:.2f} minutos)\n")
        f.write(f"Tiempo promedio por enfrentamiento: {time_per_match:.2f} segundos\n")
        f.write(f"Trabajadores utilizados: {n_workers}\n")
        f.write(f"Tipo de núcleos utilizados: {cpu_type_str}\n")
        if specific_cores:
            f.write(f"Núcleos específicos: {','.join(map(str, specific_cores))}\n")
        f.write(f"Aceleración estimada: {n_workers}x\n")
        f.write(f"Enfrentamientos totales: {total_matches}\n")
        f.write(f"Enfrentamientos completados: {completed_matches}\n")

        # Información detallada de CPU
        f.write("\nInformación de CPU:\n")
        f.write(f"Sistema operativo: {platform.system()}\n")
        f.write(f"Núcleos lógicos: {cpu_info['logical_cores']}\n")
        f.write(f"Núcleos físicos: {cpu_info['physical_cores']}\n")

        if matches_data:
            f.write(f"\nTiempo mínimo por enfrentamiento: {min([m['Duration'] for m in matches_data]):.2f} segundos\n")
            f.write(f"Tiempo máximo por enfrentamiento: {max([m['Duration'] for m in matches_data]):.2f} segundos\n")

    # ========== SISTEMA DE PUNTUACIÓN BRADLEY-TERRY ==========
    logger.info("\n" + "=" * 60)
    logger.info("CALCULANDO HABILIDADES BRADLEY-TERRY")
    logger.info("=" * 60)

    try:
        # Calcular habilidades Bradley-Terry
        bt_skills = compute_bradley_terry_skills(results_df, epochs)

        # Crear DataFrame con las habilidades Bradley-Terry
        bt_df = pd.DataFrame({
            'Época': epochs,
            'Habilidad_BT': [bt_skills[epoch] for epoch in epochs],
            'Victorias': [results_df.at[epoch, 'Victorias'] for epoch in epochs],
            'Derrotas': [results_df.at[epoch, 'Derrotas'] for epoch in epochs],
            'Empates': [results_df.at[epoch, 'Empates'] for epoch in epochs]
        })

        # Calcular probabilidades de victoria promedio contra todos los oponentes
        bt_df['Prob_Victoria_Promedio'] = 0.0
        for i, epoch_i in enumerate(epochs):
            prob_sum = 0.0
            for epoch_j in epochs:
                if epoch_i != epoch_j:
                    # Probabilidad de victoria usando Bradley-Terry: exp(skill_i) / (exp(skill_i) + exp(skill_j))
                    prob_i_beats_j = np.exp(bt_skills[epoch_i]) / (np.exp(bt_skills[epoch_i]) + np.exp(bt_skills[epoch_j]))
                    prob_sum += prob_i_beats_j
            bt_df.at[i, 'Prob_Victoria_Promedio'] = prob_sum / (len(epochs) - 1)

        # Convertir a porcentaje
        bt_df['Prob_Victoria_Promedio'] *= 100

        # Ordenar por habilidad Bradley-Terry (descendente)
        bt_df = bt_df.sort_values('Habilidad_BT', ascending=False).reset_index(drop=True)

        # Agregar ranking
        bt_df['Ranking_BT'] = range(1, len(bt_df) + 1)

        # Mostrar tabla de habilidades Bradley-Terry
        logger.info("\nTABLA DE HABILIDADES BRADLEY-TERRY:")
        display_columns = ['Ranking_BT', 'Época', 'Habilidad_BT', 'Prob_Victoria_Promedio', 'Victorias', 'Derrotas', 'Empates']
        bt_display = bt_df[display_columns].copy()
        bt_display['Habilidad_BT'] = bt_display['Habilidad_BT'].round(3)
        bt_display['Prob_Victoria_Promedio'] = bt_display['Prob_Victoria_Promedio'].round(1)

        logger.info("\n" + bt_display.to_string(index=False))

        # Identificar al campeón según Bradley-Terry
        bt_champion = bt_df.iloc[0]['Época']
        bt_champion_skill = bt_df.iloc[0]['Habilidad_BT']
        bt_champion_prob = bt_df.iloc[0]['Prob_Victoria_Promedio']

        logger.info(f"\n🏆 CAMPEÓN SEGÚN BRADLEY-TERRY: Época {bt_champion}")
        logger.info(f"   Habilidad: {bt_champion_skill:.3f}")
        logger.info(f"   Probabilidad promedio de victoria: {bt_champion_prob:.1f}%")

        # Comparar con el campeón tradicional
        if bt_champion != champion:
            logger.info(f"\n📊 COMPARACIÓN DE SISTEMAS:")
            logger.info(f"   Campeón tradicional (puntos): Época {champion}")
            logger.info(f"   Campeón Bradley-Terry: Época {bt_champion}")
            logger.info(f"   Los sistemas de puntuación difieren en el ganador.")
        else:
            logger.info(f"\n✅ Ambos sistemas de puntuación coinciden: Época {champion} es el campeón")

        # Guardar resultados Bradley-Terry en results/
        bt_df.to_csv(f"{results_dir}/bradley_terry_skills.csv", index=False)
        logger.info(f"\n💾 Tabla Bradley-Terry guardada en: {results_dir}/bradley_terry_skills.csv")

        # Crear visualización adicional si se solicitó
        if visualize:
            # Gráfico de habilidades Bradley-Terry
            plt.figure(figsize=(12, 8))
            colors = ['gold' if i == 0 else 'silver' if i == 1 else 'chocolate' if i == 2 else 'skyblue'
                     for i in range(len(bt_df))]

            bars = plt.bar(bt_df['Época'].astype(str), bt_df['Habilidad_BT'], color=colors, alpha=0.8)
            plt.title('Habilidades Bradley-Terry por Época', fontsize=16, fontweight='bold')
            plt.xlabel('Época del Agente', fontsize=12)
            plt.ylabel('Habilidad Bradley-Terry', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(axis='y', alpha=0.3)

            # Agregar valores en las barras
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

            # Agregar leyenda para los colores
            legend_elements = [
                plt.Rectangle((0,0),1,1, facecolor='gold', alpha=0.8, label='1º Lugar'),
                plt.Rectangle((0,0),1,1, facecolor='silver', alpha=0.8, label='2º Lugar'),
                plt.Rectangle((0,0),1,1, facecolor='chocolate', alpha=0.8, label='3º Lugar'),
                plt.Rectangle((0,0),1,1, facecolor='skyblue', alpha=0.8, label='Otros')
            ]
            plt.legend(handles=legend_elements, loc='upper right')

            plt.tight_layout()
            plt.savefig(f"{vis_dir}/bradley_terry_skills.png", dpi=300, bbox_inches='tight')
            plt.close()

            # Gráfico comparativo: Habilidad BT vs Probabilidad de Victoria
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(bt_df['Habilidad_BT'], bt_df['Prob_Victoria_Promedio'],
                                c=bt_df['Ranking_BT'], cmap='viridis_r', s=100, alpha=0.7)

            # Agregar etiquetas para cada punto
            for i, row in bt_df.iterrows():
                plt.annotate(f"Época {row['Época']}",
                           (row['Habilidad_BT'], row['Prob_Victoria_Promedio']),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)

            plt.colorbar(scatter, label='Ranking Bradley-Terry')
            plt.title('Habilidad Bradley-Terry vs Probabilidad de Victoria', fontsize=16, fontweight='bold')
            plt.xlabel('Habilidad Bradley-Terry', fontsize=12)
            plt.ylabel('Probabilidad Promedio de Victoria (%)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{vis_dir}/bt_skill_vs_probability.png", dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"📊 Gráficos Bradley-Terry guardados en: {vis_dir}/")

    except Exception as e:
        logger.error(f"❌ Error al calcular las habilidades Bradley-Terry: {e}")
        logger.info("El torneo se completó exitosamente, pero no se pudieron calcular las habilidades Bradley-Terry.")

    # Guardar resumen completo en texto
    summary_file = f"{tournament_dir}/RESUMEN_TORNEO_PARALELO.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RESUMEN DEL TORNEO PARALELO\n")
        f.write("=" * 80 + "\n\n")

        # Información general
        f.write(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Número de agentes: {len(epochs)}\n")
        f.write(f"Épocas participantes: {epochs}\n")
        f.write(f"Total de enfrentamientos: {total_matches}\n")
        f.write(f"Partidas por enfrentamiento: {n_matches}\n")
        f.write(f"Temperatura: {temperature}\n")
        f.write(f"Visualización habilitada: {'Sí' if visualize else 'No'}\n")
        f.write(f"\nParalelización:\n")
        f.write(f"  Trabajadores: {n_workers}\n")
        f.write(f"  Tipo de núcleos: {cpu_type_str}\n")
        if specific_cores:
            f.write(f"  Núcleos específicos: {','.join(map(str, specific_cores))}\n")
        f.write(f"\nRendimiento:\n")
        f.write(f"  Tiempo total: {total_time/60:.2f} minutos\n")
        f.write(f"  Tiempo promedio por enfrentamiento: {time_per_match:.2f} segundos\n")
        f.write(f"  Velocidad: {total_matches/total_time:.2f} enfrentamientos/segundo\n")
        f.write("\n" + "=" * 80 + "\n\n")

        # Clasificación final
        f.write("🏆 TABLA DE POSICIONES\n")
        f.write("=" * 80 + "\n\n")
        f.write(position_table.to_string())
        f.write("\n\n" + "=" * 80 + "\n\n")

        # Campeón
        f.write("🥇 CAMPEÓN\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Época {champion}\n")
        f.write(f"  Victorias: {position_table.at[champion, 'Victorias']:.0f}\n")
        f.write(f"  Derrotas: {position_table.at[champion, 'Derrotas']:.0f}\n")
        f.write(f"  Empates: {position_table.at[champion, 'Empates']:.0f}\n")
        f.write(f"  Puntos: {position_table.at[champion, 'Puntos']:.0f}\n\n")

        f.write("=" * 80 + "\n\n")

        # Bradley-Terry si está disponible
        try:
            if 'bt_df' in locals():
                f.write("📊 HABILIDADES BRADLEY-TERRY\n")
                f.write("=" * 80 + "\n\n")
                f.write(bt_display.to_string(index=False))
                f.write("\n\n" + "=" * 80 + "\n\n")

                f.write(f"🏆 CAMPEÓN BRADLEY-TERRY: Época {bt_champion}\n")
                f.write(f"  Habilidad: {bt_champion_skill:.3f}\n")
                f.write(f"  Probabilidad promedio de victoria: {bt_champion_prob:.1f}%\n\n")

                if bt_champion != champion:
                    f.write(f"📊 NOTA: El campeón tradicional (Época {champion}) difiere del campeón Bradley-Terry (Época {bt_champion})\n\n")

                f.write("=" * 80 + "\n\n")
        except:
            pass

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
    logger.info(f"   ⚡ Estadísticas de rendimiento: {performance_file}")

    return results_df
def get_cpu_info():
    """Obtiene información básica sobre la CPU del sistema.

    Returns:
        dict: Diccionario con información de la CPU, incluyendo:
            - physical_cores: número de núcleos físicos
            - logical_cores: número de núcleos lógicos (físicos + virtuales)
    """
    info = {
        'physical_cores': 0,
        'logical_cores': mp.cpu_count()
    }

    # Detectar núcleos físicos (multiplataforma)
    if hasattr(os, "sched_getaffinity"):
        # Linux
        try:
            info['physical_cores'] = len(os.sched_getaffinity(0))
        except AttributeError:
            pass

    if info['physical_cores'] == 0:
        try:
            # Windows, macOS, Linux
            import psutil
            info['physical_cores'] = psutil.cpu_count(logical=False)
        except (ImportError, AttributeError):
            pass

    # Si aún no tenemos núcleos físicos, usar métodos específicos por plataforma
    if info['physical_cores'] == 0:
        system = platform.system()
        if system == 'Windows':
            info['physical_cores'] = _get_physical_cores_windows()
        elif system == 'Darwin':  # macOS
            info['physical_cores'] = _get_physical_cores_macos()
        elif system == 'Linux':
            info['physical_cores'] = _get_physical_cores_linux()

    # Si no se pudo determinar, usar la mitad de los núcleos lógicos como estimación
    if info['physical_cores'] == 0:
        info['physical_cores'] = max(1, info['logical_cores'] // 2)

    return info

def _get_physical_cores_windows():
    """Obtiene el número de núcleos físicos en Windows."""
    try:
        # Intentar usar WMI
        import wmi
        w = wmi.WMI()
        return int(w.Win32_ComputerSystem()[0].NumberOfProcessors)
    except (ImportError, Exception):
        try:
            # Alternativa usando subprocess
            output = subprocess.check_output(['wmic', 'cpu', 'get', 'NumberOfCores']).decode()
            return int(output.strip().split('\n')[1])
        except Exception:
            return 0

def _get_physical_cores_macos():
    """Obtiene el número de núcleos físicos en macOS."""
    try:
        output = subprocess.check_output(['sysctl', '-n', 'hw.physicalcpu']).decode().strip()
        return int(output)
    except Exception:
        return 0

def _get_physical_cores_linux():
    """Obtiene el número de núcleos físicos en Linux."""
    try:
        # Método 1: contar núcleos físicos únicos
        with open('/proc/cpuinfo', 'r') as f:
            lines = f.readlines()

        physical_ids = set()
        core_ids = {}

        current_physical_id = None

        for line in lines:
            if line.strip().startswith('physical id'):
                current_physical_id = line.strip().split(':')[1].strip()
                physical_ids.add(current_physical_id)
            elif line.strip().startswith('core id') and current_physical_id is not None:
                core_id = line.strip().split(':')[1].strip()
                if current_physical_id not in core_ids:
                    core_ids[current_physical_id] = set()
                core_ids[current_physical_id].add(core_id)

        total_cores = sum(len(cores) for cores in core_ids.values())
        if total_cores > 0:
            return total_cores

        # Método 2: usar lscpu
        output = subprocess.check_output(['lscpu']).decode()
        for line in output.split('\n'):
            if 'Core(s) per socket' in line:
                cores_per_socket = int(line.split(':')[1].strip())
            elif 'Socket(s)' in line:
                sockets = int(line.split(':')[1].strip())
                return cores_per_socket * sockets
    except Exception:
        return 0

def get_cores_for_parallelism(physical_only=False):
    """Determina el número óptimo de núcleos a utilizar para paralelización.

    Args:
        physical_only (bool): Si True, usa solo núcleos físicos

    Returns:
        int: Número de núcleos a utilizar
    """
    # Obtener información de la CPU
    cpu_info = get_cpu_info()

    if physical_only:
        # Usar solo núcleos físicos
        cores_to_use = cpu_info['physical_cores']
    else:
        # Usar todos los núcleos lógicos
        cores_to_use = cpu_info['logical_cores']

    # Garantizar al menos 1 núcleo
    return max(1, cores_to_use)

def main():
    """Función principal para ejecutar el torneo desde línea de comandos o interactivamente."""
    # Comprobar si se pasaron argumentos por línea de comandos
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            description="Torneo paralelo para agentes de Quarto",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=__doc__.split('\n\nUso:')[1]
        )
        parser.add_argument("--epochs", type=int, nargs='+', help="Lista de épocas para el torneo")
        parser.add_argument("--all", action="store_true", help="Usar todas las épocas disponibles")
        parser.add_argument("--max", type=int, default=10, help="Número máximo de agentes a incluir (con --all)")
        parser.add_argument("--matches", type=int, default=10, help="Número de partidas por enfrentamiento (default: 10)")
        parser.add_argument("--temp", type=float, default=0.5, help="Temperatura para los agentes (default: 0.5)")
        parser.add_argument("--visualize", action="store_true", help="Guardar partidas y generar visualización")
        parser.add_argument("--workers", type=int, default=None,
                           help="Número de trabajadores paralelos (default: número de CPUs)")
        parser.add_argument("--cores", type=str, help="Lista de núcleos específicos a utilizar (ej: 0,1,2,5)")
        parser.add_argument("--physical-only", action="store_true", help="Usar solo núcleos físicos")

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

        # Procesar específicos núcleos si se especificaron
        specific_cores = None
        if args.cores:
            try:
                specific_cores = [int(c.strip()) for c in args.cores.split(',')]
                logger.info(f"Utilizando núcleos específicos: {specific_cores}")
            except ValueError:
                logger.error(f"Formato inválido para la lista de núcleos: {args.cores}. Debe ser una lista de enteros separados por comas.")
                return

        # Ejecutar el torneo con las opciones de núcleos especificadas
        run_tournament_parallel(
            epochs=epochs,
            n_matches=args.matches,
            temperature=args.temp,
            visualize=args.visualize,
            n_workers=args.workers,
            physical_only=args.physical_only,
            specific_cores=specific_cores
        )
    else:
        # Modo interactivo - pedir parámetros al usuario
        print("\n===== Torneo Paralelo de Agentes para Quarto =====")
        print("Este programa organiza un torneo 'todos contra todos' entre agentes de diferentes épocas.")
        print("Utiliza multiprocesamiento para acelerar la ejecución.")

        # Obtener información de CPU y mostrarla
        cpu_info = get_cpu_info()
        print(f"\nInformación de CPU detectada:")
        print(f"- Núcleos lógicos: {cpu_info['logical_cores']}")
        print(f"- Núcleos físicos: {cpu_info['physical_cores']}")

        # Obtener épocas disponibles
        available_epochs = get_all_available_epochs()

        if not available_epochs:
            print("No se encontraron épocas disponibles. Verifique que existan modelos entrenados.")
            return

        print(f"\n📋 Épocas disponibles:")
        print(f"   {available_epochs}")
        print(f"\n   Total disponibles: {len(available_epochs)}")

        # Mostrar sugerencia si hay muchas épocas
        if len(available_epochs) > 8:
            suggested = select_epochs_for_tournament(max_agents=8)
            print(f"   Sugerencia (8 agentes): {suggested}")

        # Preguntar si quiere todas las épocas o selección
        print("\nOpciones de selección de épocas:")
        print("1. Usar todas las épocas disponibles")
        print("2. Selección automática (épocas distribuidas uniformemente)")
        print("3. Selección manual de épocas específicas")

        selection = input("\nSeleccione una opción [2]: ").strip() or "2"

        if selection == "1":
            # Usar todas las épocas
            epochs = available_epochs
            print(f"Se utilizarán todas las {len(epochs)} épocas disponibles")
        elif selection == "2":
            # Selección automática
            max_agents = input(f"¿Cuántos agentes desea incluir? [8]: ").strip() or "8"
            try:
                max_agents = int(max_agents)
                if max_agents < 2:
                    max_agents = 2
                epochs = select_epochs_for_tournament(max_agents)
                print(f"Épocas seleccionadas automáticamente: {epochs}")
            except ValueError:
                print("Valor inválido, usando 8 agentes.")
                epochs = select_epochs_for_tournament(8)
        else:
            # Selección manual
            print(f"\nÉpocas disponibles: {available_epochs}")
            epochs_input = input("\nIngrese las épocas separadas por espacios (ej: 1 50 100 150 200): ")
            try:
                epochs = [int(e) for e in epochs_input.split()]
                # Verificar que las épocas existan
                invalid_epochs = [e for e in epochs if e not in available_epochs]
                if invalid_epochs:
                    print(f"Advertencia: Las siguientes épocas no están disponibles: {invalid_epochs}")
                    epochs = [e for e in epochs if e in available_epochs]
            except ValueError:
                print("Entrada inválida. Usando selección automática.")
                epochs = select_epochs_for_tournament(5)

        if len(epochs) < 2:
            print("Se necesitan al menos 2 épocas para un torneo. Usando las primeras 2 épocas disponibles.")
            epochs = available_epochs[:2]

        print(f"Épocas finales seleccionadas: {epochs}")

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

        # Preguntar por opciones de paralelización
        print("\nOpciones de paralelización:")
        print("1. Usar todos los núcleos lógicos (máximo rendimiento)")
        print("2. Usar solo núcleos físicos (mayor estabilidad)")
        print("3. Especificar manualmente los núcleos a utilizar")

        cores_option = input("\nSeleccione una opción [1]: ") or "1"

        physical_only = False
        workers = None
        specific_cores = None

        if cores_option == "2":
            physical_only = True
            print(f"Se usarán solo los {cpu_info['physical_cores']} núcleos físicos")
        elif cores_option == "3":
            # Especificar manualmente los núcleos
            total_cores = cpu_info['logical_cores']
            print(f"\nSu sistema tiene {total_cores} núcleos lógicos (numerados del 0 al {total_cores-1}).")
            cores_input = input("Ingrese los números de núcleos a utilizar, separados por comas (ej: 0,1,2,5): ")

            try:
                specific_cores = [int(c.strip()) for c in cores_input.split(',')]

                # Validar que los núcleos estén en el rango correcto
                invalid_cores = [c for c in specific_cores if c < 0 or c >= total_cores]
                if invalid_cores:
                    print(f"Advertencia: Los siguientes núcleos están fuera de rango: {invalid_cores}")
                    specific_cores = [c for c in specific_cores if c >= 0 and c < total_cores]

                if not specific_cores:
                    print(f"No se especificaron núcleos válidos. Usando todos los núcleos lógicos.")
                    specific_cores = None
                else:
                    print(f"Se utilizarán los siguientes núcleos: {specific_cores}")
            except ValueError:
                print("Formato inválido. Usando todos los núcleos lógicos.")
                specific_cores = None
        else:
            # Caso por defecto: usar todos los núcleos lógicos
            print(f"Se usarán todos los {cpu_info['logical_cores']} núcleos lógicos")

        visualize_input = input("\n¿Desea visualizar y guardar los resultados? (s/n) [s]: ").lower() or "s"
        visualize = visualize_input in ["s", "si", "sí", "y", "yes"]

        # Determinar el número real de núcleos a utilizar para mostrar en el resumen
        if specific_cores:
            cores_to_use = len(specific_cores)
            core_type = "específicos"
        else:
            cores_to_use = get_cores_for_parallelism(physical_only)
            if physical_only:
                core_type = "físicos"
            else:
                core_type = "lógicos"

        # Resumen de parámetros
        print("\n===== Parámetros del torneo =====")
        print(f"Número de agentes: {len(epochs)}")
        print(f"Épocas: {epochs}")
        print(f"Número de enfrentamientos: {len(epochs) * (len(epochs) - 1) // 2}")
        print(f"Partidas por enfrentamiento: {n_matches}")
        print(f"Temperatura: {temperature}")
        if specific_cores:
            print(f"Paralelización: {cores_to_use} núcleos {core_type} ({','.join(map(str, specific_cores))})")
        else:
            print(f"Paralelización: {cores_to_use} núcleos {core_type}")
        print(f"Visualizar resultados: {'Sí' if visualize else 'No'}")

        confirm = input("\n¿Iniciar el torneo con estos parámetros? (s/n) [s]: ").lower() or "s"
        if confirm in ["s", "si", "sí", "y", "yes"]:
            print("\nIniciando torneo paralelo...\n")
            run_tournament_parallel(
                epochs=epochs,
                n_matches=n_matches,
                temperature=temperature,
                visualize=visualize,
                n_workers=workers,
                physical_only=physical_only,
                specific_cores=specific_cores
            )
        else:
            print("Torneo cancelado por el usuario.")

if __name__ == "__main__":
    # Configurar para que multiprocessing funcione correctamente en Windows
    mp.freeze_support()
    main()
