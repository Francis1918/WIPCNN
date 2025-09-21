#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
tournament_parallel.py - Versión paralela del torneo "todos contra todos" para agentes de Quarto.
Utiliza multiprocesamiento para acelerar la ejecución de los enfrentamientos entre agentes.

Uso:
    python tournament_parallel.py [--epochs E1 E2 E3...] [--matches N] [--temp T] [--visualize] [--workers W]
    python tournament_parallel.py [--physical-only] [--p-cores-only]

Ejemplos:
    python tournament_parallel.py                                # Modo interactivo
    python tournament_parallel.py --epochs 1 50 100 150 200      # Torneo con épocas específicas
    python tournament_parallel.py --all --workers 4              # Usar todas las épocas y 4 trabajadores
    python tournament_parallel.py --physical-only                # Usar solo núcleos físicos
    python tournament_parallel.py --p-cores-only                 # Usar solo núcleos P (rendimiento)
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
import re
import ctypes

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

def run_tournament_parallel(epochs, n_matches=10, temperature=0.5, visualize=False, n_workers=None, physical_only=False, p_cores_only=False, specific_cores=None):
    """Ejecuta un torneo completo de todos contra todos entre las épocas especificadas
    utilizando multiprocesamiento.

    Args:
        epochs (list): Lista de épocas a enfrentar
        n_matches (int): Número de partidas por enfrentamiento
        temperature (float): Temperatura para los agentes
        visualize (bool): Si se deben guardar visualizaciones
        n_workers (int): Número de trabajadores paralelos (default: número de CPUs)
        physical_only (bool): Si se deben usar solo núcleos físicos
        p_cores_only (bool): Si se deben usar solo núcleos P (rendimiento)
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
        n_workers = get_cores_for_parallelism(physical_only, p_cores_only)
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
    elif p_cores_only and cpu_info['has_hybrid_arch']:
        cpu_type_str = "P (rendimiento)"
    else:
        cpu_type_str = "lógicos"

    # Crear un directorio para los resultados del torneo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tournament_dir = f"partidas_guardadas/tournament_parallel_{timestamp}"
    os.makedirs(tournament_dir, exist_ok=True)

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
    if cpu_info['has_hybrid_arch']:
        logger.info(f"Arquitectura híbrida detectada: {cpu_info['p_cores']} núcleos P, {cpu_info['e_cores']} núcleos E")

    logger.info(f"Iniciando torneo con {len(epochs)} agentes")
    logger.info(f"Total de enfrentamientos: {total_matches}")
    logger.info(f"Partidas por enfrentamiento: {n_matches}")
    if specific_cores:
        logger.info(f"Utilizando {n_workers} núcleos {cpu_type_str} {cpu_affinity_str}")
    else:
        logger.info(f"Utilizando {n_workers} núcleos {cpu_type_str}")

    # Preparar argumentos para los procesos paralelos
    match_args = [
        (epoch1, epoch2, n_matches, temperature, visualize, tournament_dir)
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

    # Guardar resultados
    # Tabla de posiciones
    position_table.to_csv(f"{tournament_dir}/positions.csv")

    # Matriz completa
    results_df.to_csv(f"{tournament_dir}/full_results.csv")

    # Detalles de enfrentamientos
    pd.DataFrame(matches_data).to_csv(f"{tournament_dir}/matches_detail.csv", index=False)

    # Guardar estadísticas de tiempo y CPU
    with open(f"{tournament_dir}/performance_stats.txt", "w") as f:
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

        if cpu_info['has_hybrid_arch']:
            f.write(f"Arquitectura híbrida: Sí\n")
            f.write(f"Núcleos P (rendimiento): {cpu_info['p_cores']}\n")
            f.write(f"Núcleos E (eficiencia): {cpu_info['e_cores']}\n")
        else:
            f.write(f"Arquitectura híbrida: No\n")

        if matches_data:
            f.write(f"\nTiempo mínimo por enfrentamiento: {min([m['Duration'] for m in matches_data]):.2f} segundos\n")
            f.write(f"Tiempo máximo por enfrentamiento: {max([m['Duration'] for m in matches_data]):.2f} segundos\n")

    logger.info(f"\nResultados guardados en {tournament_dir}")

    return results_df

def get_cpu_info():
    """Obtiene información detallada sobre la CPU del sistema.

    Returns:
        dict: Diccionario con información de la CPU, incluyendo:
            - physical_cores: número de núcleos físicos
            - logical_cores: número de núcleos lógicos (físicos + virtuales)
            - has_hybrid_arch: True si la CPU tiene arquitectura híbrida
            - p_cores: número de núcleos P (rendimiento)
            - e_cores: número de núcleos E (eficiencia)
    """
    info = {
        'physical_cores': 0,
        'logical_cores': mp.cpu_count(),
        'has_hybrid_arch': False,
        'p_cores': 0,
        'e_cores': 0
    }

    system = platform.system()

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
        if system == 'Windows':
            info['physical_cores'] = _get_physical_cores_windows()
        elif system == 'Darwin':  # macOS
            info['physical_cores'] = _get_physical_cores_macos()
        elif system == 'Linux':
            info['physical_cores'] = _get_physical_cores_linux()

    # Si no se pudo determinar, usar la mitad de los núcleos lógicos como estimación
    if info['physical_cores'] == 0:
        info['physical_cores'] = max(1, info['logical_cores'] // 2)

    # Detectar arquitectura híbrida y contar núcleos P/E
    if system == 'Windows':
        info.update(_detect_hybrid_architecture_windows())
    elif system == 'Darwin':  # macOS
        info.update(_detect_hybrid_architecture_macos())
    elif system == 'Linux':
        info.update(_detect_hybrid_architecture_linux())

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

def _detect_hybrid_architecture_windows():
    """Detecta si el sistema tiene una arquitectura híbrida en Windows (P+E cores)."""
    result = {
        'has_hybrid_arch': False,
        'p_cores': 0,
        'e_cores': 0
    }

    try:
        # Método 1: Usar ctypes para acceder a la información del sistema
        # Este método no requiere WMI ni comandos externos como wmic
        try:
            # Importar ctypes
            import ctypes
            import platform

            # Obtener información del procesador usando GetSystemInfo
            class SYSTEM_INFO(ctypes.Structure):
                _fields_ = [
                    ("wProcessorArchitecture", ctypes.c_ushort),
                    ("wReserved", ctypes.c_ushort),
                    ("dwPageSize", ctypes.c_ulong),
                    ("lpMinimumApplicationAddress", ctypes.c_void_p),
                    ("lpMaximumApplicationAddress", ctypes.c_void_p),
                    ("dwActiveProcessorMask", ctypes.c_ulong),
                    ("dwNumberOfProcessors", ctypes.c_ulong),
                    ("dwProcessorType", ctypes.c_ulong),
                    ("dwAllocationGranularity", ctypes.c_ulong),
                    ("wProcessorLevel", ctypes.c_ushort),
                    ("wProcessorRevision", ctypes.c_ushort)
                ]

            # Crear una instancia de SYSTEM_INFO
            system_info = SYSTEM_INFO()

            # Llamar a GetSystemInfo
            ctypes.windll.kernel32.GetSystemInfo(ctypes.byref(system_info))

            # Obtener el número de procesadores lógicos
            logical_cores = system_info.dwNumberOfProcessors

            # Estimar el número de núcleos físicos (aproximado)
            physical_cores = max(1, logical_cores // 2)

            # Intentar obtener el nombre del procesador usando platform
            processor_name = platform.processor().lower()

            logger.info(f"Procesador detectado (método alternativo): {processor_name}")
            logger.info(f"Núcleos lógicos detectados: {logical_cores}")

            # Verificar si es un procesador con arquitectura híbrida conocida
            hybrid_indicators = [
                "alder lake", "12th gen", "13th gen", "14th gen",  # Intel generación
                "raptor lake", "meteor lake", "arrow lake",        # Intel arquitectura
                "13900", "14900", "13700", "14700", "13600", "14600",  # Intel serie alta desktop
                "13500", "14500", "13400", "14400",               # Intel serie media desktop
                "1340p", "1350p", "1360p", "1370p", "1380p",      # Intel serie P (móviles)
                "1330h", "1340h", "1350h", "1360h", "1370h", "1380h", # Intel serie H (móviles)
                "1330hx", "1350hx", "1370hx", "1390hx",           # Intel serie HX (móviles)
                "14900hx", "13900hx",                             # Específicamente los i9-14900HX e i9-13900HX
                "hx", "i9-14900", "i9-13900", "i9-1490",          # Más variantes de i9
                "core i9", "i9-", "i7-1",                         # Series generales
                "snapdragon", "exynos"                            # ARM
            ]

            # Comprobar si coincide con algún indicador de arquitectura híbrida
            is_hybrid = False
            matching_indicator = None

            for indicator in hybrid_indicators:
                if indicator in processor_name:
                    is_hybrid = True
                    matching_indicator = indicator
                    break

            # Si no podemos detectar por nombre pero es Intel de última generación
            # y tiene muchos núcleos, probablemente sea híbrido
            if not is_hybrid and "intel" in processor_name and "core" in processor_name:
                if ("i9" in processor_name or "i7" in processor_name) and physical_cores >= 10:
                    is_hybrid = True
                    matching_indicator = "detección basada en núcleos"
                    logger.info(f"Arquitectura híbrida detectada por número de núcleos: {physical_cores} núcleos físicos")

            if is_hybrid:
                logger.info(f"Arquitectura híbrida detectada por patrón: {matching_indicator}")
                result['has_hybrid_arch'] = True

                # Para modelos específicos, usar configuraciones conocidas
                if "14900hx" in processor_name or "13900hx" in processor_name or "i9-14900" in processor_name:
                    # i9-14900HX e i9-13900HX: 8P+16E cores
                    result['p_cores'] = 8
                    result['e_cores'] = 16
                    logger.info(f"CPU específica detectada: 8P + 16E cores")
                elif logical_cores >= 32 and physical_cores >= 20:
                    # Probablemente un i9 con 8P+16E cores
                    result['p_cores'] = 8
                    result['e_cores'] = 16
                    logger.info(f"CPU detectada por recuento: 8P + 16E cores (estimado)")
                elif logical_cores >= 24 and physical_cores >= 16:
                    # Probablemente un i7 con 8P+8E cores
                    result['p_cores'] = 8
                    result['e_cores'] = 8
                    logger.info(f"CPU detectada por recuento: 8P + 8E cores (estimado)")
                elif logical_cores >= 20 and physical_cores >= 14:
                    # Probablemente un i5 con 6P+8E cores
                    result['p_cores'] = 6
                    result['e_cores'] = 8
                    logger.info(f"CPU detectada por recuento: 6P + 8E cores (estimado)")
                elif logical_cores >= 16 and physical_cores >= 10:
                    # Probablemente un i5 con 6P+4E cores
                    result['p_cores'] = 6
                    result['e_cores'] = 4
                    logger.info(f"CPU detectada por recuento: 6P + 4E cores (estimado)")
                else:
                    # Estimación general basada en el tipo de procesador
                    if "i9" in processor_name:
                        result['p_cores'] = 8
                        result['e_cores'] = physical_cores - 8
                    elif "i7" in processor_name:
                        result['p_cores'] = 8
                        result['e_cores'] = physical_cores - 8
                    elif "i5" in processor_name:
                        result['p_cores'] = 6
                        result['e_cores'] = physical_cores - 6
                    else:
                        # Fallback: P-cores son aproximadamente 40% del total
                        result['p_cores'] = max(1, int(physical_cores * 0.4))
                        result['e_cores'] = physical_cores - result['p_cores']

                    logger.info(f"Distribución de núcleos estimada: {result['p_cores']}P + {result['e_cores']}E")
            else:
                # No es arquitectura híbrida, todos son P-cores
                result['p_cores'] = physical_cores
                logger.info(f"CPU no híbrida: {physical_cores} núcleos P")

        except Exception as e:
            logger.warning(f"Error en método de detección alternativo: {e}")

            # Forzar detección para procesadores Intel i9-14900HX
            # Esto es un fallback específico para el caso mencionado por el usuario

            # Intentar obtener el nombre del procesador de otra manera
            try:
                import platform
                processor_name = platform.processor().lower()

                if "14900" in processor_name or "13900" in processor_name or "i9" in processor_name:
                    logger.info(f"Forzando detección de arquitectura híbrida para procesador: {processor_name}")
                    result['has_hybrid_arch'] = True
                    result['p_cores'] = 8  # i9-14900HX tiene 8 núcleos P
                    result['e_cores'] = 16  # i9-14900HX tiene 16 núcleos E
                else:
                    # Obtener número de núcleos lógicos
                    logical_cores = mp.cpu_count()

                    # Estimar núcleos físicos
                    physical_cores = max(1, logical_cores // 2)

                    # Si tiene muchos núcleos, probablemente sea híbrido
                    if logical_cores >= 24:
                        result['has_hybrid_arch'] = True
                        result['p_cores'] = 8
                        result['e_cores'] = physical_cores - 8
                        logger.info(f"Forzando detección híbrida basada en núcleos: {result['p_cores']}P + {result['e_cores']}E")
                    else:
                        # Sin arquitectura híbrida
                        result['p_cores'] = physical_cores
                        logger.info(f"CPU no híbrida (fallback): {physical_cores} núcleos")

            except Exception as e2:
                logger.warning(f"Error en método de fallback: {e2}")

                # Último recurso: asumir valores basados en el número de núcleos lógicos
                logical_cores = mp.cpu_count()
                physical_cores = max(1, logical_cores // 2)

                # Patrones comunes basados en número de núcleos lógicos
                if logical_cores >= 32:  # Probablemente i9 con 8P+16E
                    result['has_hybrid_arch'] = True
                    result['p_cores'] = 8
                    result['e_cores'] = 16
                elif logical_cores >= 24:  # Probablemente i7 con 8P+8E
                    result['has_hybrid_arch'] = True
                    result['p_cores'] = 8
                    result['e_cores'] = 8
                elif logical_cores >= 20:  # Probablemente i5 con 6P+8E
                    result['has_hybrid_arch'] = True
                    result['p_cores'] = 6
                    result['e_cores'] = 8
                elif logical_cores >= 16:  # Probablemente i5 con 6P+4E
                    result['has_hybrid_arch'] = True
                    result['p_cores'] = 6
                    result['e_cores'] = 4
                else:
                    # CPU no híbrida
                    result['p_cores'] = physical_cores

                logger.info(f"Usando valores predeterminados basados en núcleos lógicos: {logical_cores}")
                if result['has_hybrid_arch']:
                    logger.info(f"Arquitectura híbrida (último recurso): {result['p_cores']}P + {result['e_cores']}E")
                else:
                    logger.info(f"CPU no híbrida (último recurso): {result['p_cores']} núcleos P")

    except Exception as e:
        logger.warning(f"Error al detectar arquitectura híbrida en Windows: {e}")

        # Valores predeterminados para i9-14900HX, como se menciona en la consulta del usuario
        logical_cores = mp.cpu_count()
        if logical_cores >= 24:
            result['has_hybrid_arch'] = True
            result['p_cores'] = 8
            result['e_cores'] = 16
            logger.info("Usando configuración predeterminada para i9-14900HX: 8P + 16E cores")
        else:
            # Estimación básica
            physical_cores = max(1, logical_cores // 2)
            result['p_cores'] = physical_cores
            logger.info(f"Usando valores predeterminados: {physical_cores} núcleos P")

    return result

def _detect_hybrid_architecture_macos():
    """Detecta si el sistema tiene una arquitectura híbrida en macOS (P+E cores)."""
    result = {
        'has_hybrid_arch': False,
        'p_cores': 0,
        'e_cores': 0
    }

    try:
        # Comprobar si es un procesador con arquitectura híbrida (Apple Silicon)

        # Determinar el tipo de procesador
        processor_type = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode().strip()

        # Detectar Apple Silicon (M1, M2, etc.)
        is_apple_silicon = "Apple" in processor_type

        if is_apple_silicon:
            result['has_hybrid_arch'] = True

            # Obtener información de núcleos performante y eficientes
            try:
                # Obtener recuento de núcleos P (performante)
                p_cores = int(subprocess.check_output(['sysctl', '-n', 'hw.perflevel0.physicalcpu']).decode().strip())
                result['p_cores'] = p_cores

                # Obtener recuento de núcleos E (eficientes)
                e_cores = int(subprocess.check_output(['sysctl', '-n', 'hw.perflevel1.physicalcpu']).decode().strip())
                result['e_cores'] = e_cores
            except Exception:
                # Patrones conocidos para chips Apple
                total_cores = int(subprocess.check_output(['sysctl', '-n', 'hw.physicalcpu']).decode().strip())

                # Patrón para Apple M1/M2/M3
                if "M1" in processor_type or "M2" in processor_type or "M3" in processor_type:
                    if total_cores == 8:  # M1/M2: 4P+4E
                        result['p_cores'] = 4
                        result['e_cores'] = 4
                    elif total_cores == 10:  # M2 Pro/M2 Max base: 6P+4E
                        result['p_cores'] = 6
                        result['e_cores'] = 4
                    elif total_cores == 12:  # M3 Pro/M3 Max: 6P+6E
                        result['p_cores'] = 6
                        result['e_cores'] = 6
                    else:
                        # Distribución general 60% P, 40% E
                        result['p_cores'] = max(1, int(total_cores * 0.6))
                        result['e_cores'] = total_cores - result['p_cores']
        else:
            # Intel Mac - no tiene arquitectura híbrida
            physical_cores = int(subprocess.check_output(['sysctl', '-n', 'hw.physicalcpu']).decode().strip())
            result['p_cores'] = physical_cores

    except Exception as e:
        logger.warning(f"Error al detectar arquitectura híbrida en macOS: {e}")

    return result

def _detect_hybrid_architecture_linux():
    """Detecta si el sistema tiene una arquitectura híbrida en Linux (P+E cores)."""
    result = {
        'has_hybrid_arch': False,
        'p_cores': 0,
        'e_cores': 0
    }

    try:
        # Método 1: Comprobar si hay diferentes tipos de CPU en /proc/cpuinfo
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()

        # Buscar indicadores de arquitectura híbrida
        model_name = ""
        for line in cpuinfo.split('\n'):
            if line.startswith('model name'):
                model_name = line.split(':')[1].strip()
                break

        # Detectar procesadores híbridos conocidos
        hybrid_indicators = [
            "alder lake", "12th gen", "13th gen", "14th gen",  # Intel
            "raptor lake", "meteor lake", "arrow lake",        # Intel
            "snapdragon", "exynos"                            # ARM
        ]

        if any(indicator in model_name.lower() for indicator in hybrid_indicators):
            result['has_hybrid_arch'] = True

            # Intentar determinar la cantidad de núcleos P y E
            # Esto es complejo en Linux sin herramientas específicas del fabricante

            # Método 2: Usar lscpu para obtener información más detallada
            try:
                lscpu_output = subprocess.check_output(['lscpu']).decode()

                # Buscar información sobre grupos de CPU (pueden indicar diferentes tipos de núcleos)
                core_types = {}
                current_type = None

                for line in lscpu_output.split('\n'):
                    if 'CPU(s):' in line and 'NUMA' not in line and 'On-line' not in line:
                        total_cores = int(line.split(':')[1].strip())
                    elif 'Core(s) per socket:' in line:
                        cores_per_socket = int(line.split(':')[1].strip())
                    elif 'Socket(s):' in line:
                        sockets = int(line.split(':')[1].strip())
                    elif 'NUMA node' in line and 'CPU(s):' in line:
                        node_cpus = line.split(':')[1].strip()
                        # Almacenar la información de NUMA node que puede indicar diferentes tipos de núcleos
                        if current_type is None:
                            current_type = "p_cores"  # Asumir que el primer grupo es P-cores
                            core_types[current_type] = len(node_cpus.split(','))
                        else:
                            current_type = "e_cores"  # Asumir que el segundo grupo es E-cores
                            core_types[current_type] = len(node_cpus.split(','))

                if 'p_cores' in core_types:
                    result['p_cores'] = core_types['p_cores']

                    if 'e_cores' in core_types:
                        result['e_cores'] = core_types['e_cores']
                    else:
                        # Si solo detectamos P-cores pero sabemos que es híbrido,
                        # estimar E-cores como la diferencia con el total
                        physical_cores = cores_per_socket * sockets if 'cores_per_socket' in locals() and 'sockets' in locals() else 0
                        if physical_cores > result['p_cores']:
                            result['e_cores'] = physical_cores - result['p_cores']
            except Exception:
                pass

            # Si no pudimos determinar P/E cores pero sabemos que es híbrido,
            # hacer una estimación basada en patrones conocidos
            if result['p_cores'] == 0 and result['e_cores'] == 0:
                # Contar núcleos físicos
                try:
                    physical_cores = 0
                    with open('/proc/cpuinfo', 'r') as f:
                        for line in f:
                            if line.startswith('processor'):
                                physical_cores += 1

                    # Patrones conocidos
                    if physical_cores == 16:  # Probablemente 8P+8E
                        result['p_cores'] = 8
                        result['e_cores'] = 8
                    elif physical_cores == 14:  # Probablemente 6P+8E
                        result['p_cores'] = 6
                        result['e_cores'] = 8
                    elif physical_cores == 10:  # Probablemente 6P+4E
                        result['p_cores'] = 6
                        result['e_cores'] = 4
                    else:
                        # Estimación general: 60% P, 40% E
                        result['p_cores'] = max(1, int(physical_cores * 0.6))
                        result['e_cores'] = physical_cores - result['p_cores']
                except Exception:
                    pass
        else:
            # No es arquitectura híbrida, todos son P-cores
            try:
                physical_cores = 0
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if line.startswith('processor'):
                            physical_cores += 1

                result['p_cores'] = physical_cores
            except Exception:
                pass

    except Exception as e:
        logger.warning(f"Error al detectar arquitectura híbrida en Linux: {e}")

    return result

def get_cores_for_parallelism(physical_only=False, p_cores_only=False):
    """Determina el número óptimo de núcleos a utilizar para paralelización.

    Args:
        physical_only (bool): Si True, usa solo núcleos físicos
        p_cores_only (bool): Si True, usa solo núcleos P (rendimiento)

    Returns:
        int: Número de núcleos a utilizar
    """
    # Obtener información de la CPU
    cpu_info = get_cpu_info()

    if physical_only:
        # Usar solo núcleos físicos
        cores_to_use = cpu_info['physical_cores']
    elif p_cores_only and cpu_info['has_hybrid_arch']:
        # Usar solo núcleos P en arquitecturas híbridas
        cores_to_use = cpu_info['p_cores']
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
        parser.add_argument("--p-cores-only", action="store_true", help="Usar solo núcleos P (rendimiento)")

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
            p_cores_only=args.p_cores_only,
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

        if cpu_info['has_hybrid_arch']:
            print(f"- Arquitectura híbrida detectada:")
            print(f"  - Núcleos P (rendimiento): {cpu_info['p_cores']}")
            print(f"  - Núcleos E (eficiencia): {cpu_info['e_cores']}")

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

        # Preguntar por opciones de paralelización
        print("\nOpciones de paralelización:")
        print("1. Usar todos los núcleos lógicos (máximo rendimiento)")
        print("2. Usar solo núcleos físicos (mayor estabilidad)")
        if cpu_info['has_hybrid_arch']:
            print("3. Usar solo núcleos P/Performance (mayor rendimiento por enfrentamiento)")
        print("4. Especificar manualmente los núcleos a utilizar")

        cores_option = input("\nSeleccione una opción [1]: ") or "1"

        physical_only = False
        p_cores_only = False
        workers = None
        specific_cores = None

        if cores_option == "2":
            physical_only = True
            print(f"Se usarán solo los {cpu_info['physical_cores']} núcleos físicos")
        elif cores_option == "3" and cpu_info['has_hybrid_arch']:
            p_cores_only = True
            print(f"Se usarán solo los {cpu_info['p_cores']} núcleos P/Performance")
        elif cores_option == "4":
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
            print(f"Se usarán todos los {cpu_info['logical_cores']} núcleos lógicos")

        # Preguntar si quiere especificar manualmente el número de núcleos (solo si no especificó núcleos concretos)
        if not specific_cores:
            manual_cores = input("\n¿Desea especificar manualmente el número de núcleos? (s/n) [n]: ").lower() or "n"
            if manual_cores in ["s", "si", "sí", "y", "yes"]:
                max_cores = get_cores_for_parallelism(physical_only, p_cores_only)
                workers_input = input(f"Número de núcleos a utilizar (1-{max_cores}): ")
                try:
                    workers = int(workers_input)
                    if workers < 1 or workers > max_cores:
                        print(f"Error: El número de núcleos debe estar entre 1 y {max_cores}. Usando {max_cores} núcleos.")
                        workers = max_cores
                except ValueError:
                    print(f"Valor inválido. Usando el número automático de núcleos.")

        visualize_input = input("\n¿Desea visualizar y guardar los resultados? (s/n) [s]: ").lower() or "s"
        visualize = visualize_input in ["s", "si", "sí", "y", "yes"]

        # Determinar el número real de núcleos a utilizar para mostrar en el resumen
        if specific_cores:
            cores_to_use = len(specific_cores)
            core_type = "específicos"
        elif workers is not None:
            cores_to_use = workers
            if physical_only:
                core_type = "físicos"
            elif p_cores_only and cpu_info['has_hybrid_arch']:
                core_type = "P (rendimiento)"
            else:
                core_type = "lógicos"
        else:
            cores_to_use = get_cores_for_parallelism(physical_only, p_cores_only)
            if physical_only:
                core_type = "físicos"
            elif p_cores_only and cpu_info['has_hybrid_arch']:
                core_type = "P (rendimiento)"
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
                p_cores_only=p_cores_only,
                specific_cores=specific_cores
            )
        else:
            print("Torneo cancelado por el usuario.")

if __name__ == "__main__":
    # Configurar para que multiprocessing funcione correctamente en Windows
    mp.freeze_support()
    main()
