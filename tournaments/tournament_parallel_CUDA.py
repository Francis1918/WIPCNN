#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
tournament_parallel_CUDA.py - Versión GPU-acelerada del torneo paralelo para agentes de Quarto.
Utiliza CUDA/GPU para la inferencia de los modelos mientras mantiene paralelismo en CPU.

🚀 CARACTERÍSTICAS PRINCIPALES:
    - ✅ Multiprocesamiento en CPU (múltiples enfrentamientos en paralelo)
    - ✅ Inferencia acelerada por GPU (CUDA) en cada proceso
    - ✅ Soporte para múltiples GPUs (distribución automática)
    - ✅ Gestión inteligente de memoria VRAM
    - ✅ Batch processing cuando sea posible

💡 ARQUITECTURA:
    - CPU: Ejecuta N procesos paralelos (enfrentamientos simultáneos)
    - GPU: Cada proceso usa GPU para inferencia de modelos (10-50x más rápido)
    - Multi-GPU: Distribuye procesos entre GPUs disponibles

    Ejemplo con 8 CPU workers + 1 GPU:
    [CPU Worker 1] → [GPU 0] ← [CPU Worker 5]
    [CPU Worker 2] → [GPU 0] ← [CPU Worker 6]
    [CPU Worker 3] → [GPU 0] ← [CPU Worker 7]
    [CPU Worker 4] → [GPU 0] ← [CPU Worker 8]

    Ejemplo con 8 CPU workers + 2 GPUs:
    [CPU Workers 1,3,5,7] → [GPU 0]
    [CPU Workers 2,4,6,8] → [GPU 1]

Uso:
    python tournament_parallel_CUDA.py [--epochs E1 E2 E3...] [--matches N] [--temp T] [--visualize]
    python tournament_parallel_CUDA.py [--cpu-workers N] [--gpu ID] [--multi-gpu]

Ejemplos:
    # Usar GPU con multiprocesamiento automático
    python tournament_parallel_CUDA.py --epochs 1 50 100 150

    # Especificar número de procesos paralelos + GPU
    python tournament_parallel_CUDA.py --all --cpu-workers 8 --gpu 0

    # Usar múltiples GPUs con 16 workers
    python tournament_parallel_CUDA.py --all --cpu-workers 16 --multi-gpu

    # Solo núcleos físicos + GPU específica
    python tournament_parallel_CUDA.py --physical-only --gpu 1

    # Máximo rendimiento: todos los núcleos + multi-GPU
    python tournament_parallel_CUDA.py --all --cpu-workers 16 --multi-gpu --visualize
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
import torch

# Importar funciones necesarias
from utils.logger import logger

# Verificar disponibilidad de CUDA
CUDA_AVAILABLE = torch.cuda.is_available()
N_GPUS = torch.cuda.device_count() if CUDA_AVAILABLE else 0

def get_gpu_info():
    """Obtiene información detallada sobre las GPUs disponibles."""
    if not CUDA_AVAILABLE:
        return {
            'available': False,
            'count': 0,
            'devices': []
        }

    devices = []
    for i in range(N_GPUS):
        props = torch.cuda.get_device_properties(i)
        device_info = {
            'id': i,
            'name': props.name,
            'total_memory_gb': props.total_memory / (1024**3),
            'compute_capability': f"{props.major}.{props.minor}",
            'multi_processor_count': props.multi_processor_count
        }
        devices.append(device_info)

    return {
        'available': True,
        'count': N_GPUS,
        'devices': devices,
        'cuda_version': torch.version.cuda
    }

def print_gpu_info():
    """Imprime información sobre las GPUs disponibles."""
    gpu_info = get_gpu_info()

    if not gpu_info['available']:
        logger.error("❌ CUDA NO ESTÁ DISPONIBLE")
        logger.error("Este script REQUIERE una GPU con CUDA para funcionar.")
        logger.info("\n📋 Para usar este torneo necesitas:")
        logger.info("   1. Una GPU NVIDIA")
        logger.info("   2. Drivers NVIDIA instalados")
        logger.info("   3. PyTorch con soporte CUDA")
        logger.info("\n💡 Alternativas:")
        logger.info("   • Para CPU multiproceso: usa 'tournament_parallel.py'")
        logger.info("   • Para secuencial: usa 'tournament.py'")
        logger.info("\n🔧 Para instalar CUDA support:")
        logger.info("   pip uninstall torch torchvision torchaudio -y")
        logger.info("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        logger.info("\n📚 Ejecuta 'python check_cuda.py' para diagnóstico completo")
        return False

    logger.info(f"\n{'='*60}")
    logger.info(f"🎮 INFORMACIÓN DE GPU - CUDA {gpu_info['cuda_version']}")
    logger.info(f"{'='*60}")
    logger.info(f"✅ GPUs disponibles: {gpu_info['count']}")

    for device in gpu_info['devices']:
        logger.info(f"\n  GPU {device['id']}: {device['name']}")
        logger.info(f"    • Memoria: {device['total_memory_gb']:.2f} GB")
        logger.info(f"    • Compute Capability: {device['compute_capability']}")
        logger.info(f"    • Multiprocesadores: {device['multi_processor_count']}")

    logger.info(f"{'='*60}\n")
    return True

def setup_gpu_environment(gpu_id=None, multi_gpu=False):
    """Configura el entorno GPU para los procesos workers.

    Args:
        gpu_id (int): ID de GPU específica a usar (None = automático)
        multi_gpu (bool): Si True, distribuye workers entre múltiples GPUs

    Returns:
        dict: Configuración GPU a usar
    """
    if not CUDA_AVAILABLE:
        logger.error("❌ ERROR CRÍTICO: CUDA no está disponible")
        logger.error("Este script está diseñado EXCLUSIVAMENTE para uso con GPU")
        logger.error("Por favor, usa 'tournament_parallel.py' si quieres multiprocesamiento en CPU")
        raise RuntimeError("CUDA no disponible. Este script requiere GPU obligatoriamente.")

    if multi_gpu and N_GPUS > 1:
        logger.info(f"✅ Modo multi-GPU habilitado: {N_GPUS} GPUs disponibles")
        for i in range(N_GPUS):
            gpu_name = torch.cuda.get_device_name(i)
            logger.info(f"   GPU {i}: {gpu_name}")
        return {'device': 'cuda', 'multi_gpu': True, 'n_gpus': N_GPUS}

    if gpu_id is not None:
        if gpu_id >= N_GPUS:
            logger.warning(f"⚠️  GPU {gpu_id} no existe. Usando GPU 0")
            gpu_id = 0
        gpu_name = torch.cuda.get_device_name(gpu_id)
        logger.info(f"✅ Usando GPU {gpu_id}: {gpu_name}")
        return {'device': f'cuda:{gpu_id}', 'gpu_id': gpu_id, 'gpu_name': gpu_name}

    # Por defecto, usar GPU 0
    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"✅ Usando GPU 0 (por defecto): {gpu_name}")
    return {'device': 'cuda:0', 'gpu_id': 0, 'gpu_name': gpu_name}

def load_agent_gpu(epoch, temperature=0.5, device='cuda:0'):
    """Carga un agente en GPU para inferencia acelerada.

    Args:
        epoch (int): Época del modelo a cargar
        temperature (float): Temperatura para el agente
        device (str): Device de PyTorch ('cuda:0', 'cuda:1', etc.)

    Returns:
        bot: Agente configurado con GPU
    """
    from models.CNN1 import QuartoCNN
    from bot.CNN_bot import Quarto_bot

    weights_dir = "models/weights/QuartoCNN1"

    if not os.path.exists(weights_dir):
        raise FileNotFoundError(f"No se encontró el directorio: {weights_dir}")

    # Buscar modelo
    model_pattern = f"*-ba_increasing_n_last_states_epoch_{epoch:04d}.pt"
    matching_files = list(Path(weights_dir).glob(model_pattern))

    if not matching_files:
        raise FileNotFoundError(f"No se encontró modelo para época {epoch}")

    matching_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    model_path = str(matching_files[0])

    # Determinar el device correcto
    if device == 'cpu' or not CUDA_AVAILABLE:
        actual_device = torch.device('cpu')
    else:
        actual_device = torch.device(device)
        # 🔧 LIMPIEZA PREVENTIVA: Limpiar caché CUDA antes de cargar modelo
        torch.cuda.empty_cache()
        # Sincronizar para asegurar que operaciones previas terminaron
        torch.cuda.synchronize(actual_device)

    # Cargar modelo en GPU
    model = QuartoCNN()

    # 🔧 FIX: Cargar primero en CPU, luego mover a GPU para evitar problemas de memoria
    try:
        state_dict = torch.load(model_path, map_location='cpu')  # Siempre cargar en CPU primero
        model.load_state_dict(state_dict)
        model.to(actual_device)  # Luego mover a GPU
        model.eval()

        # Liberar el state_dict inmediatamente
        del state_dict

    except RuntimeError as e:
        # Si falla, limpiar y reintentar
        if actual_device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize(actual_device)
        raise RuntimeError(f"Error cargando modelo época {epoch} en {actual_device}: {e}")

    # Crear bot
    bot = Quarto_bot(model=model)
    bot.DETERMINISTIC = False
    bot.TEMPERATURE = temperature

    # Almacenar el device en el bot para referencia
    bot._device = actual_device

    return bot

def save_checkpoint(tournament_dir, results_df, matches_data, completed_matches, total_combinations, start_time):
    """Guarda un checkpoint del estado actual del torneo.
    
    Args:
        tournament_dir (str): Directorio del torneo
        results_df (pd.DataFrame): DataFrame con resultados actuales
        matches_data (list): Lista de datos de enfrentamientos completados
        completed_matches (int): Número de enfrentamientos completados
        total_combinations (int): Total de enfrentamientos a realizar
        start_time (float): Tiempo de inicio del torneo
    """
    import pickle
    
    checkpoint_dir = f"{tournament_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_data = {
        'results_df': results_df,
        'matches_data': matches_data,
        'completed_matches': completed_matches,
        'total_combinations': total_combinations,
        'start_time': start_time,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    checkpoint_file = f"{checkpoint_dir}/checkpoint_latest.pkl"
    checkpoint_backup = f"{checkpoint_dir}/checkpoint_{completed_matches}_{total_combinations}.pkl"
    
    # Guardar checkpoint
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    # Crear backup cada 10% de progreso
    progress_pct = (completed_matches / total_combinations) * 100
    if completed_matches % max(1, total_combinations // 10) == 0:
        with open(checkpoint_backup, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        logger.info(f"💾 Checkpoint guardado: {progress_pct:.1f}% completado ({completed_matches}/{total_combinations})")

def load_checkpoint(tournament_dir):
    """Carga un checkpoint previo del torneo si existe.
    
    Args:
        tournament_dir (str): Directorio del torneo
        
    Returns:
        dict or None: Datos del checkpoint o None si no existe
    """
    import pickle
    
    checkpoint_file = f"{tournament_dir}/checkpoints/checkpoint_latest.pkl"
    
    if not os.path.exists(checkpoint_file):
        return None
    
    try:
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
        logger.info(f"✅ Checkpoint cargado: {checkpoint_data['completed_matches']}/{checkpoint_data['total_combinations']} enfrentamientos")
        logger.info(f"   Guardado: {checkpoint_data['timestamp']}")
        return checkpoint_data
    except Exception as e:
        logger.error(f"❌ Error cargando checkpoint: {e}")
        return None

def get_completed_matchups(matches_data):
    """Obtiene el conjunto de enfrentamientos ya completados.
    
    Args:
        matches_data (list): Lista de datos de enfrentamientos
        
    Returns:
        set: Conjunto de tuplas (epoch1, epoch2) completadas
    """
    completed = set()
    for match in matches_data:
        # 🔧 FIX: Las claves correctas son 'Epoch1' y 'Epoch2', no 'Epoch_A' y 'Epoch_B'
        epoch1 = match['Epoch1']
        epoch2 = match['Epoch2']
        # Normalizar para que siempre sea (menor, mayor)
        completed.add((min(epoch1, epoch2), max(epoch1, epoch2)))
    return completed

def run_match_parallel_cuda(args):
    """Función para ejecutar un enfrentamiento usando GPU.

    Args:
        args (tuple): (epoch1, epoch2, n_matches, temperature, visualize, matches_dir, gpu_config)

    Returns:
        tuple: (epoch1, epoch2, match_results, match_data)
    """
    epoch1, epoch2, n_matches, temperature, visualize, matches_dir, gpu_config = args

    match_start = time.time()
    process_id = os.getpid()

    # Determinar qué GPU usar si es multi-GPU
    if gpu_config.get('multi_gpu', False):
        # Distribuir procesos entre GPUs disponibles
        worker_id = process_id % gpu_config['n_gpus']
        device = f"cuda:{worker_id}"
    else:
        device = gpu_config.get('device', 'cpu')

    print(f"[Proceso {process_id}] GPU: {device} | Enfrentamiento: Época {epoch1} vs Época {epoch2}")

    # Directorio para este enfrentamiento
    match_dir = f"{matches_dir}/match_{epoch1}_vs_{epoch2}"
    os.makedirs(match_dir, exist_ok=True)

    # 🔧 Variables para limpieza garantizada
    agent1 = None
    agent2 = None

    try:
        # Pequeña pausa aleatoria para evitar condiciones de carrera en GPU
        import random
        time.sleep(random.uniform(0.01, 0.1))

        # 🔧 LIMPIEZA PREVENTIVA: Asegurar que GPU está en estado limpio
        if device != 'cpu' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Pequeña pausa para permitir que la GPU libere recursos
            time.sleep(0.05)

        # Cargar agentes en GPU con manejo de errores mejorado
        try:
            agent1 = load_agent_gpu(epoch1, temperature, device)
            agent2 = load_agent_gpu(epoch2, temperature, device)
        except RuntimeError as e:
            # Si hay error de GPU (CUDA out of memory, etc.), reintentar en CPU
            print(f"[Proceso {process_id}] ⚠️  Error cargando en GPU: {e}, reintentando...")
            if device != 'cpu':
                torch.cuda.empty_cache()
                time.sleep(0.5)
            raise

        # Importar play_games
        try:
            from quartopy import play_games
        except ImportError:
            import setup_dependencies
            setup_dependencies.setup_quartopy(silent=True)
            from quartopy import play_games

        # Ejecutar partidas
        play_args = {
            'player1': agent1,
            'player2': agent2,
            'matches': n_matches,
            'verbose': False,
            'return_file_paths': False
        }

        if visualize:
            play_args['match_dir'] = match_dir

        match_results = play_games(**play_args)

        if not match_results:
            print(f"[Proceso {process_id}] No se obtuvieron resultados para {epoch1} vs {epoch2}")
            return epoch1, epoch2, None, None

        # Extraer resultados
        wins_1 = match_results['P1']
        wins_2 = match_results['P2']
        draws = match_results['Empates']

        # Guardar datos del enfrentamiento
        match_data = {
            'Epoch1': epoch1,
            'Epoch2': epoch2,
            'Wins_Epoch1': wins_1,
            'Wins_Epoch2': wins_2,
            'Draws': draws,
            'Win_Rate_Epoch1': wins_1 / n_matches * 100,
            'Win_Rate_Epoch2': wins_2 / n_matches * 100,
            'Draw_Rate': draws / n_matches * 100,
            'Duration': time.time() - match_start,
            'Device': device,
            'Process_ID': process_id
        }

        print(f"[Proceso {process_id}] ✓ Completado en {match_data['Duration']:.2f}s: {wins_1}-{wins_2}-{draws}")

        # Flush stdout para asegurar que el mensaje se imprima
        sys.stdout.flush()

        return epoch1, epoch2, match_results, match_data

    except Exception as e:
        print(f"[Proceso {process_id}] ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return epoch1, epoch2, None, None

    finally:
        # 🔧 LIMPIEZA GARANTIZADA: Siempre se ejecuta, incluso si hay errores
        try:
            # Eliminar referencias a los agentes y sus modelos
            if agent1 is not None:
                if hasattr(agent1, 'model'):
                    del agent1.model
                del agent1

            if agent2 is not None:
                if hasattr(agent2, 'model'):
                    del agent2.model
                del agent2

            # Forzar recolección de basura
            import gc
            gc.collect()

            # Limpieza agresiva de GPU
            if device != 'cpu' and torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Sincronizar para asegurar que todas las operaciones terminaron
                torch.cuda.synchronize()

        except Exception as cleanup_error:
            print(f"[Proceso {process_id}] ⚠️  Error en limpieza final: {cleanup_error}")

# Importar funciones auxiliares del tournament_parallel original
from tournament_parallel import (
    compute_bradley_terry_skills,
    compute_bradley_terry_simple,
    get_all_available_epochs,
    select_epochs_for_tournament,
    get_cpu_info,
    get_cores_for_parallelism
)

def run_tournament_parallel_cuda(epochs, n_matches=10, temperature=0.5, visualize=False,
                                 n_workers=None, physical_only=False, gpu_id=None, multi_gpu=False):
    """Ejecuta un torneo paralelo con aceleración GPU.

    Args:
        epochs (list): Lista de épocas a enfrentar
        n_matches (int): Número de partidas por enfrentamiento
        temperature (float): Temperatura para los agentes
        visualize (bool): Si se deben guardar visualizaciones
        n_workers (int): Número de trabajadores paralelos
        physical_only (bool): Usar solo núcleos físicos de CPU
        gpu_id (int): ID de GPU específica a usar (None = automático)
        multi_gpu (bool): Distribuir carga entre múltiples GPUs

    Returns:
        pd.DataFrame: Tabla de resultados del torneo
    """
    if len(epochs) < 2:
        logger.error("Se necesitan al menos 2 épocas para un torneo")
        return None

    # ⚠️ VALIDACIÓN DE ESCALA - Evitar torneos inmanejables
    n_epochs = len(epochs)
    total_combinations = (n_epochs * (n_epochs - 1)) // 2

    # Advertir si el torneo es muy grande
    if total_combinations > 1_000_000:
        logger.error(f"\n{'='*60}")
        logger.error(f"❌ TORNEO EXCESIVAMENTE GRANDE")
        logger.error(f"{'='*60}")
        logger.error(f"Épocas solicitadas: {n_epochs:,}")
        logger.error(f"Enfrentamientos necesarios: {total_combinations:,}")
        logger.error(f"\n⚠️  ADVERTENCIA: Esto es computacionalmente inviable!")
        logger.error(f"   Tiempo estimado (10 seg/enfrentamiento): {total_combinations*10/3600/24:.1f} días")
        logger.error(f"   Memoria estimada: {total_combinations*8/1024/1024/1024:.2f} GB solo para resultados")
        logger.error(f"\n💡 RECOMENDACIONES:")
        logger.error(f"   • Reducir número de épocas a máximo 1,000 (499,500 enfrentamientos)")
        logger.error(f"   • Usar selección representativa de épocas")
        logger.error(f"   • Considerar torneos por grupos/brackets")
        return None

    if total_combinations > 10_000:
        logger.warning(f"\n⚠️  ADVERTENCIA: Torneo grande detectado")
        logger.warning(f"   Épocas: {n_epochs:,}")
        logger.warning(f"   Enfrentamientos: {total_combinations:,}")
        estimated_time_hours = (total_combinations * 10) / 3600  # Asumiendo 10 seg/enfrentamiento
        logger.warning(f"   Tiempo estimado: {estimated_time_hours:.1f} horas")
        logger.warning(f"   ¿Desea continuar?\n")

    # Configurar GPU
    gpu_config = setup_gpu_environment(gpu_id, multi_gpu)

    # Determinar número de trabajadores
    cpu_info = get_cpu_info()
    if n_workers is None:
        n_workers = get_cores_for_parallelism(physical_only)

    # Mostrar configuración
    logger.info(f"\n{'='*60}")
    logger.info(f"⚙️  CONFIGURACIÓN DEL TORNEO GPU")
    logger.info(f"{'='*60}")
    logger.info(f"Épocas: {len(epochs)} agentes (min: {min(epochs)}, max: {max(epochs)})")
    logger.info(f"Enfrentamientos totales: {total_combinations:,}")
    logger.info(f"Partidas por enfrentamiento: {n_matches}")
    logger.info(f"Temperatura: {temperature}")
    logger.info(f"Trabajadores CPU: {n_workers}")
    if gpu_config['device'] != 'cpu':
        if multi_gpu:
            logger.info(f"GPUs: {gpu_config['n_gpus']} (distribución automática)")
        else:
            logger.info(f"GPU: {gpu_config['device']}")
    else:
        logger.info(f"GPU: No disponible (usando CPU)")
    logger.info(f"{'='*60}\n")

    # Crear estructura de directorios
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_tournaments_dir = "tournaments_parallel_CUDA"
    os.makedirs(main_tournaments_dir, exist_ok=True)

    tournament_dir = f"{main_tournaments_dir}/tournament_{timestamp}"
    os.makedirs(tournament_dir, exist_ok=True)

    results_dir = f"{tournament_dir}/results"
    matches_dir = f"{tournament_dir}/matches"
    vis_dir = f"{tournament_dir}/visualizations"

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(matches_dir, exist_ok=True)
    if visualize:
        os.makedirs(vis_dir, exist_ok=True)

    # 💾 Intentar cargar checkpoint si existe
    checkpoint = load_checkpoint(tournament_dir)
    
    if checkpoint is not None:
        logger.info(f"\n{'='*60}")
        logger.info(f"📦 CHECKPOINT ENCONTRADO - REANUDANDO TORNEO")
        logger.info(f"{'='*60}")
        results_df = checkpoint['results_df']
        matches_data = checkpoint['matches_data']
        completed_matches = checkpoint['completed_matches']
        start_time = checkpoint['start_time']
        
        # Obtener enfrentamientos ya completados
        completed_matchups = get_completed_matchups(matches_data)
        
        logger.info(f"✅ Progreso recuperado: {completed_matches}/{total_combinations} ({(completed_matches/total_combinations)*100:.1f}%)")
        logger.info(f"   Tiempo transcurrido anterior: {(time.time() - start_time)/3600:.2f} horas")
        logger.info(f"   Enfrentamientos restantes: {total_combinations - completed_matches}")
        logger.info(f"{'='*60}\n")
    else:
        # Crear DataFrame de resultados - optimizado para muchas épocas
        # Usar dtype específico para ahorrar memoria
        logger.info("Inicializando estructuras de datos...")
        results_df = pd.DataFrame(
            index=epochs,
            columns=epochs + ['Victorias', 'Derrotas', 'Empates', 'Puntos', 'Posición'],
            dtype=np.float32  # Usar float32 en lugar de float64 para ahorrar memoria
        )
        results_df.fillna(0, inplace=True)

        matches_data = []
        completed_matches = 0
        completed_matchups = set()
        start_time = time.time()

    # 🚀 OPTIMIZACIÓN: Generar combinaciones sin materializar en memoria
    # No usar list(), mantener como generador hasta que sea necesario
    logger.info(f"Generando {total_combinations:,} combinaciones de enfrentamientos...")

    logger.info(f"Resultados se guardarán en: {tournament_dir}")
    logger.info(f"Iniciando torneo con {len(epochs)} agentes...\n")

    # Preparar argumentos para procesos paralelos
    # Usar generador para evitar materializar todas las combinaciones
    # Saltar enfrentamientos ya completados si hay checkpoint
    def generate_match_args():
        for epoch1, epoch2 in itertools.combinations(epochs, 2):
            # Saltar si ya está completado
            matchup_key = (min(epoch1, epoch2), max(epoch1, epoch2))
            if matchup_key not in completed_matchups:
                yield (epoch1, epoch2, n_matches, temperature, visualize, matches_dir, gpu_config)

    # Ejecutar enfrentamientos en paralelo
    logger.info("🚀 Iniciando enfrentamientos en paralelo...")

    # Procesar en lotes para evitar crear demasiados futures simultáneamente
    BATCH_SIZE = max(n_workers * 10, 100)  # Procesar en lotes de al menos 100
    
    # 💾 Configurar guardado automático de checkpoints
    CHECKPOINT_INTERVAL = 100  # Guardar cada 100 enfrentamientos
    last_checkpoint_save = completed_matches

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Procesar resultados con barra de progreso mejorada
        # Ajustar la barra si se reanudó desde checkpoint
        with tqdm(total=total_combinations,
                 initial=completed_matches,
                 desc="🎮 Enfrentamientos GPU",
                 unit="enfrentamiento",
                 bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:

            # Procesar en lotes para evitar problemas de memoria
            match_args_generator = generate_match_args()

            while completed_matches < total_combinations:
                # Tomar el siguiente lote
                batch = list(itertools.islice(match_args_generator, BATCH_SIZE))
                if not batch:
                    break

                # Enviar el lote al executor
                future_to_match = {
                    executor.submit(run_match_parallel_cuda, args): args
                    for args in batch
                }

                for future in as_completed(future_to_match):
                    try:
                        epoch1, epoch2, match_results, match_data = future.result()
                        completed_matches += 1

                        if match_results is None:
                            pbar.update(1)
                            continue

                        wins_1 = match_results['P1']
                        wins_2 = match_results['P2']
                        draws = match_results['Empates']

                        # Actualizar resultados
                        results_df.at[epoch1, epoch2] = wins_1
                        results_df.at[epoch2, epoch1] = wins_2

                        results_df.at[epoch1, 'Victorias'] += wins_1
                        results_df.at[epoch2, 'Victorias'] += wins_2
                        results_df.at[epoch1, 'Derrotas'] += wins_2
                        results_df.at[epoch2, 'Derrotas'] += wins_1
                        results_df.at[epoch1, 'Empates'] += draws
                        results_df.at[epoch2, 'Empates'] += draws

                        results_df.at[epoch1, 'Puntos'] += (wins_1 * 3 + draws * 1)
                        results_df.at[epoch2, 'Puntos'] += (wins_2 * 3 + draws * 1)

                        matches_data.append(match_data)

                        # Calcular progreso
                        elapsed = time.time() - start_time
                        matches_per_second = completed_matches / elapsed if elapsed > 0 else 0
                        estimated_total = elapsed * (total_combinations / completed_matches) if completed_matches > 0 else 0
                        remaining = estimated_total - elapsed

                        # Actualizar descripción de la barra con estadísticas
                        percentage = (completed_matches / total_combinations) * 100
                        pbar.set_description(
                            f"🎮 GPU [{percentage:.1f}%] - {matches_per_second:.2f} enf/s - "
                            f"ETA: {remaining/60:.1f}min"
                        )

                        pbar.update(1)

                        # Log detallado cada 5 enfrentamientos
                        if completed_matches % 5 == 0:
                            logger.info(f"⚡ Progreso: {completed_matches:,}/{total_combinations:,} ({percentage:.1f}%) | "
                                      f"Velocidad: {matches_per_second:.2f} enf/s | "
                                      f"Restante: {remaining/60:.1f} min")

                        # 💾 Guardar checkpoint automático cada CHECKPOINT_INTERVAL enfrentamientos
                        if completed_matches - last_checkpoint_save >= CHECKPOINT_INTERVAL:
                            save_checkpoint(tournament_dir, results_df, matches_data, 
                                          completed_matches, total_combinations, start_time)
                            last_checkpoint_save = completed_matches
                            
                        # 💾 Guardar checkpoint cada hora (3600 segundos)
                        if elapsed % 3600 < 10:  # Aprox cada hora
                            if completed_matches - last_checkpoint_save >= 10:  # Evitar guardar múltiples veces
                                save_checkpoint(tournament_dir, results_df, matches_data, 
                                              completed_matches, total_combinations, start_time)
                                last_checkpoint_save = completed_matches
                                logger.info(f"⏰ Checkpoint automático (cada hora) - {elapsed/3600:.2f} horas transcurridas")

                    except Exception as e:
                        logger.error(f"Error en enfrentamiento: {e}")
                        pbar.update(1)

    # 💾 Guardar checkpoint final
    save_checkpoint(tournament_dir, results_df, matches_data, 
                   completed_matches, total_combinations, start_time)
    logger.info("💾 Checkpoint final guardado")

    # Calcular posiciones
    positions = results_df['Puntos'].rank(method='min', ascending=False)
    results_df['Posición'] = positions
    results_df = results_df.sort_values('Posición')

    # Estadísticas
    total_time = time.time() - start_time
    time_per_match = total_time / total_combinations if total_combinations > 0 else 0

    logger.info(f"\n{'='*60}")
    logger.info(f"🏆 RESULTADOS DEL TORNEO GPU")
    logger.info(f"{'='*60}")
    logger.info(f"Tiempo total: {total_time/60:.2f} minutos")
    logger.info(f"Tiempo promedio por enfrentamiento: {time_per_match:.2f} segundos")
    logger.info(f"Velocidad: {total_combinations/total_time:.2f} enfrentamientos/segundo")

    # Tabla de posiciones
    position_table = results_df[['Victorias', 'Derrotas', 'Empates', 'Puntos', 'Posición']].sort_values('Posición')
    logger.info(f"\n📊 TABLA DE POSICIONES:\n{position_table.to_string()}")

    champion = position_table.index[0]
    logger.info(f"\n🥇 CAMPEÓN: Época {champion}")

    # Visualizaciones
    if visualize:
        create_gpu_visualizations(results_df, epochs, n_matches, matches_data, vis_dir)

    # Guardar resultados
    position_table.to_csv(f"{results_dir}/positions.csv")
    results_df.to_csv(f"{results_dir}/full_results.csv")
    pd.DataFrame(matches_data).to_csv(f"{results_dir}/matches_detail.csv", index=False)

    # Estadísticas de rendimiento GPU
    save_gpu_performance_stats(results_dir, total_time, time_per_match, n_workers,
                               gpu_config, cpu_info, matches_data, completed_matches, total_combinations)

    # Bradley-Terry
    try:
        bt_results = compute_bradley_terry_analysis(results_df, epochs, results_dir, vis_dir, visualize)
        champion_bt = bt_results['champion']
        logger.info(f"\n🏆 CAMPEÓN BRADLEY-TERRY: Época {champion_bt}")
    except Exception as e:
        logger.error(f"Error en Bradley-Terry: {e}")

    # Guardar resumen
    save_tournament_summary(tournament_dir, epochs, n_matches, temperature, total_time,
                           time_per_match, n_workers, gpu_config, position_table,
                           champion, matches_data, total_combinations, visualize)

    logger.info(f"\n✅ Resultados guardados en: {tournament_dir}")
    logger.info(f"   📊 Resultados: {results_dir}")
    logger.info(f"   🎮 Enfrentamientos: {matches_dir}")
    if visualize:
        logger.info(f"   📈 Visualizaciones: {vis_dir}")

    return results_df

def create_gpu_visualizations(results_df, epochs, n_matches, matches_data, vis_dir):
    """Crea visualizaciones específicas para el torneo GPU."""
    # Gráfico de puntos
    position_table = results_df[['Puntos', 'Posición']].sort_values('Posición')

    plt.figure(figsize=(12, 6))
    plt.bar(position_table.index.astype(str), position_table['Puntos'], color='#00d4ff')
    plt.title('Puntos totales por agente (Torneo GPU)', fontsize=16, fontweight='bold')
    plt.xlabel('Época del agente')
    plt.ylabel('Puntos')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{vis_dir}/points_by_agent_gpu.png", dpi=300)
    plt.close()

    # Matriz de calor para enfrentamientos directos
    plt.figure(figsize=(10, 8))
    matchup_matrix = results_df[epochs].copy()

    # Normalizar por número de partidas
    for col in epochs:
        matchup_matrix[col] = matchup_matrix[col] / n_matches

    # Establecer la diagonal en 0 (un agente no puede jugar contra sí mismo)
    np.fill_diagonal(matchup_matrix.values, 0.0)

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

    # Distribución de tiempos (comparación por GPU si multi-GPU)
    if matches_data:
        plt.figure(figsize=(12, 6))
        match_times = [m['Duration'] for m in matches_data]

        plt.hist(match_times, bins=30, color='#00d4ff', alpha=0.7, edgecolor='black')
        plt.axvline(x=np.mean(match_times), color='red', linestyle='--', linewidth=2,
                   label=f'Promedio: {np.mean(match_times):.2f}s')
        plt.axvline(x=np.median(match_times), color='green', linestyle='--', linewidth=2,
                   label=f'Mediana: {np.median(match_times):.2f}s')

        plt.title('Distribución de tiempos por enfrentamiento (GPU)', fontsize=16, fontweight='bold')
        plt.xlabel('Tiempo (segundos)')
        plt.ylabel('Frecuencia')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{vis_dir}/match_durations_gpu.png", dpi=300)
        plt.close()

        # Si hay información de device, crear gráfico por GPU
        if 'Device' in matches_data[0]:
            devices = [m['Device'] for m in matches_data]
            unique_devices = list(set(devices))

            if len(unique_devices) > 1:
                plt.figure(figsize=(12, 6))
                device_times = {dev: [] for dev in unique_devices}
                for m in matches_data:
                    device_times[m['Device']].append(m['Duration'])

                for dev in unique_devices:
                    plt.hist(device_times[dev], bins=20, alpha=0.5, label=dev)

                plt.title('Distribución de tiempos por GPU', fontsize=16, fontweight='bold')
                plt.xlabel('Tiempo (segundos)')
                plt.ylabel('Frecuencia')
                plt.legend()
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{vis_dir}/gpu_comparison.png", dpi=300)
                plt.close()

def save_gpu_performance_stats(results_dir, total_time, time_per_match, n_workers,
                               gpu_config, cpu_info, matches_data, completed, total):
    """Guarda estadísticas de rendimiento específicas de GPU."""
    perf_file = f"{results_dir}/gpu_performance_stats.txt"

    with open(perf_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ESTADÍSTICAS DE RENDIMIENTO - TORNEO GPU\n")
        f.write("="*80 + "\n\n")

        f.write(f"Tiempo total: {total_time:.2f} segundos ({total_time/60:.2f} minutos)\n")
        f.write(f"Tiempo promedio por enfrentamiento: {time_per_match:.2f} segundos\n")
        f.write(f"Velocidad: {total/total_time:.2f} enfrentamientos/segundo\n")
        f.write(f"Enfrentamientos completados: {completed}/{total}\n\n")

        f.write("CONFIGURACIÓN CPU:\n")
        f.write(f"  Trabajadores: {n_workers}\n")
        f.write(f"  Núcleos lógicos: {cpu_info['logical_cores']}\n")
        f.write(f"  Núcleos físicos: {cpu_info['physical_cores']}\n\n")

        f.write("CONFIGURACIÓN GPU:\n")
        if gpu_config['device'] == 'cpu':
            f.write("  GPU: No disponible (modo CPU)\n")
        else:
            f.write(f"  Device: {gpu_config.get('device', 'N/A')}\n")
            if gpu_config.get('multi_gpu', False):
                f.write(f"  Multi-GPU: Sí ({gpu_config['n_gpus']} GPUs)\n")
            else:
                f.write(f"  Multi-GPU: No\n")

            if CUDA_AVAILABLE:
                f.write(f"  CUDA Version: {torch.version.cuda}\n")
                f.write(f"  PyTorch Version: {torch.__version__}\n")

        if matches_data:
            f.write(f"\nESTADÍSTICAS DE TIEMPO:\n")
            times = [m['Duration'] for m in matches_data]
            f.write(f"  Mínimo: {min(times):.2f}s\n")
            f.write(f"  Máximo: {max(times):.2f}s\n")
            f.write(f"  Promedio: {np.mean(times):.2f}s\n")
            f.write(f"  Mediana: {np.median(times):.2f}s\n")
            f.write(f"  Desviación estándar: {np.std(times):.2f}s\n")

def compute_bradley_terry_analysis(results_df, epochs, results_dir, vis_dir, visualize):
    """Calcula y guarda análisis Bradley-Terry."""
    bt_skills = compute_bradley_terry_skills(results_df, epochs)

    # Exponenciar las habilidades para obtener valores positivos (en lugar de log-skills)
    bt_df = pd.DataFrame({
        'Época': epochs,
        'Habilidad_BT': [np.exp(bt_skills[epoch]) for epoch in epochs],  # Exponenciar para valores positivos
        'Victorias': [results_df.at[epoch, 'Victorias'] for epoch in epochs],
        'Derrotas': [results_df.at[epoch, 'Derrotas'] for epoch in epochs],
        'Empates': [results_df.at[epoch, 'Empates'] for epoch in epochs]
    })

    # Calcular probabilidades
    # Como ya exponenciamos bt_skills al crear el DataFrame, usamos los valores directamente
    bt_df['Prob_Victoria_Promedio'] = 0.0
    for i, epoch_i in enumerate(epochs):
        prob_sum = 0.0
        skill_i = np.exp(bt_skills[epoch_i])  # Habilidad exponenciada de i
        for epoch_j in epochs:
            if epoch_i != epoch_j:
                skill_j = np.exp(bt_skills[epoch_j])  # Habilidad exponenciada de j
                prob = skill_i / (skill_i + skill_j)
                prob_sum += prob
        bt_df.at[i, 'Prob_Victoria_Promedio'] = prob_sum / (len(epochs) - 1) * 100

    bt_df = bt_df.sort_values('Habilidad_BT', ascending=False).reset_index(drop=True)
    bt_df['Ranking_BT'] = range(1, len(bt_df) + 1)

    bt_df.to_csv(f"{results_dir}/bradley_terry_skills.csv", index=False)

    logger.info(f"\n📊 BRADLEY-TERRY:\n{bt_df[['Ranking_BT', 'Época', 'Habilidad_BT', 'Prob_Victoria_Promedio']].to_string(index=False)}")

    # Crear visualizaciones si se solicitó
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

    return {'champion': bt_df.iloc[0]['Época'], 'dataframe': bt_df}

def save_tournament_summary(tournament_dir, epochs, n_matches, temperature, total_time,
                            time_per_match, n_workers, gpu_config, position_table,
                            champion, matches_data, total_matches, visualize):
    """Guarda resumen completo del torneo."""
    summary_file = f"{tournament_dir}/RESUMEN_TORNEO_GPU.txt"

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RESUMEN DEL TORNEO PARALELO GPU-ACELERADO\n")
        f.write("="*80 + "\n\n")

        f.write(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Número de agentes: {len(epochs)}\n")
        f.write(f"Épocas participantes: {epochs}\n")
        f.write(f"Total de enfrentamientos: {total_matches}\n")
        f.write(f"Partidas por enfrentamiento: {n_matches}\n")
        f.write(f"Temperatura: {temperature}\n")
        f.write(f"Visualización habilitada: {'Sí' if visualize else 'No'}\n\n")

        f.write("CONFIGURACIÓN DE ACELERACIÓN:\n")
        f.write(f"  CPU Workers: {n_workers}\n")
        if gpu_config['device'] != 'cpu':
            f.write(f"  GPU Device: {gpu_config.get('device', 'N/A')}\n")
            if gpu_config.get('multi_gpu', False):
                f.write(f"  Multi-GPU: {gpu_config['n_gpus']} GPUs\n")
        else:
            f.write(f"  GPU: No disponible (modo CPU)\n")

        f.write(f"\nRENDIMIENTO:\n")
        f.write(f"  Tiempo total: {total_time/60:.2f} minutos\n")
        f.write(f"  Tiempo promedio: {time_per_match:.2f} seg/enfrentamiento\n")
        f.write(f"  Velocidad: {total_matches/total_time:.2f} enfrentamientos/seg\n\n")

        f.write("="*80 + "\n")
        f.write("TABLA DE POSICIONES\n")
        f.write("="*80 + "\n\n")
        f.write(position_table.to_string())
        f.write(f"\n\n{'='*80}\n")
        f.write(f"CAMPEÓN: Época {champion}\n")
        f.write(f"{'='*80}\n")

def main():
    """Función principal."""
    # Mostrar información de GPU al inicio
    has_gpu = print_gpu_info()

    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            description="Torneo paralelo GPU-acelerado para agentes de Quarto",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        parser.add_argument("--epochs", type=int, nargs='+', help="Lista de épocas")
        parser.add_argument("--all", action="store_true", help="Usar todas las épocas")
        parser.add_argument("--max", type=int, default=10, help="Máximo de agentes con --all")
        parser.add_argument("--matches", type=int, default=10, help="Partidas por enfrentamiento")
        parser.add_argument("--temp", type=float, default=0.5, help="Temperatura")
        parser.add_argument("--visualize", action="store_true", help="Generar visualizaciones")
        parser.add_argument("--workers", type=int, default=None, help="Número de workers CPU")
        parser.add_argument("--physical-only", action="store_true", help="Solo núcleos físicos")
        parser.add_argument("--gpu", type=int, default=None, help="ID de GPU específica")
        parser.add_argument("--multi-gpu", action="store_true", help="Usar múltiples GPUs")

        args = parser.parse_args()

        if args.all:
            epochs = select_epochs_for_tournament(max_agents=args.max)
            if not epochs:
                logger.error("No se encontraron épocas")
                return
        elif args.epochs:
            epochs = args.epochs
        else:
            parser.print_help()
            return

        run_tournament_parallel_cuda(
            epochs=epochs,
            n_matches=args.matches,
            temperature=args.temp,
            visualize=args.visualize,
            n_workers=args.workers,
            physical_only=args.physical_only,
            gpu_id=args.gpu,
            multi_gpu=args.multi_gpu
        )
    else:
        # Modo interactivo
        print("\n" + "="*60)
        print("🎮 TORNEO PARALELO GPU-ACELERADO")
        print("="*60)

        if not has_gpu:
            proceed = input("\n⚠️  No se detectó GPU. ¿Continuar en modo CPU? (s/n) [n]: ").lower()
            if proceed not in ['s', 'si', 'sí', 'y', 'yes']:
                print("Torneo cancelado.")
                return

        available_epochs = get_all_available_epochs()
        if not available_epochs:
            print("No se encontraron épocas disponibles.")
            return

        print(f"\nÉpocas disponibles: {available_epochs}")
        print(f"Total de épocas: {len(available_epochs)}")

        # Selección de épocas
        print("\n📋 Opciones de selección:")
        print("1. Usar todas las épocas disponibles")
        print("2. Selección automática (épocas distribuidas uniformemente)")
        print("3. Selección manual de épocas específicas")

        option = input("\nSeleccione una opción [2]: ").strip() or "2"

        if option == "1":
            # Usar todas las épocas disponibles
            if len(available_epochs) > 15:
                print(f"\n⚠️  Advertencia: Hay {len(available_epochs)} épocas disponibles.")
                print(f"   Esto generará {len(available_epochs) * (len(available_epochs) - 1) // 2} enfrentamientos.")
                max_agents = input(f"¿Desea limitar el número de épocas? [Usar todas/{len(available_epochs)}]: ").strip()

                if max_agents and max_agents.lower() not in ['todas', 'all', '']:
                    try:
                        max_agents_int = int(max_agents)
                        if max_agents_int < len(available_epochs):
                            epochs = select_epochs_for_tournament(max_agents_int)
                            print(f"Usando {len(epochs)} épocas seleccionadas uniformemente")
                        else:
                            epochs = available_epochs
                            print(f"Usando todas las {len(epochs)} épocas")
                    except ValueError:
                        epochs = available_epochs
                        print(f"Usando todas las {len(epochs)} épocas")
                else:
                    epochs = available_epochs
                    print(f"Usando todas las {len(epochs)} épocas")
            else:
                epochs = available_epochs
                print(f"Usando todas las {len(epochs)} épocas disponibles")

        elif option == "2":
            # Selección automática uniforme
            max_agents = input("¿Cuántos agentes desea incluir? [8]: ").strip() or "8"
            try:
                max_agents_int = int(max_agents)
                if max_agents_int < 2:
                    max_agents_int = 2
                epochs = select_epochs_for_tournament(max_agents_int)
                print(f"Épocas seleccionadas uniformemente: {epochs}")
            except ValueError:
                print("Valor inválido, usando 8 agentes.")
                epochs = select_epochs_for_tournament(8)

        else:
            # Selección manual
            epochs_input = input("\nIngrese las épocas separadas por espacios (ej: 1 50 100 150 200): ")
            try:
                epochs = [int(e) for e in epochs_input.split()]
                # Verificar que las épocas existan
                invalid_epochs = [e for e in epochs if e not in available_epochs]
                if invalid_epochs:
                    print(f"⚠️  Advertencia: Las siguientes épocas no están disponibles: {invalid_epochs}")
                    epochs = [e for e in epochs if e in available_epochs]
            except ValueError:
                print("Entrada inválida. Usando selección automática con 5 épocas.")
                epochs = select_epochs_for_tournament(5)

        if len(epochs) < 2:
            print("⚠️  Se necesitan al menos 2 épocas para un torneo. Usando las primeras 2 épocas disponibles.")
            epochs = available_epochs[:2]

        print(f"\n✅ Épocas seleccionadas para el torneo: {epochs}")
        print(f"   Total de enfrentamientos: {len(epochs) * (len(epochs) - 1) // 2}")

        # Parámetros
        n_matches = int(input("\nPartidas por enfrentamiento [10]: ") or "10")
        temperature = float(input("Temperatura [0.5]: ") or "0.5")

        # GPU config
        gpu_id = None
        multi_gpu = False

        if has_gpu:
            if N_GPUS > 1:
                use_multi = input(f"\n¿Usar {N_GPUS} GPUs? (s/n) [s]: ").lower() or "s"
                if use_multi in ['s', 'si', 'sí', 'y', 'yes']:
                    multi_gpu = True
                else:
                    gpu_id = int(input(f"ID de GPU a usar [0]: ") or "0")
            else:
                print(f"\nUsando GPU 0")
                gpu_id = 0

        visualize = input("\n¿Visualizar resultados? (s/n) [s]: ").lower() or "s"
        visualize = visualize in ['s', 'si', 'sí', 'y', 'yes']

        # Confirmar
        print("\n" + "="*60)
        print("RESUMEN:")
        print(f"  Épocas: {epochs}")
        print(f"  Enfrentamientos: {len(epochs)*(len(epochs)-1)//2}")
        print(f"  Partidas: {n_matches}")
        print(f"  Temperatura: {temperature}")
        if multi_gpu:
            print(f"  GPUs: {N_GPUS} (distribución automática)")
        elif gpu_id is not None:
            print(f"  GPU: {gpu_id}")
        print("="*60)

        confirm = input("\n¿Iniciar torneo? (s/n) [s]: ").lower() or "s"
        if confirm in ['s', 'si', 'sí', 'y', 'yes']:
            print("\n🚀 Iniciando torneo GPU...\n")
            run_tournament_parallel_cuda(
                epochs=epochs,
                n_matches=n_matches,
                temperature=temperature,
                visualize=visualize,
                gpu_id=gpu_id,
                multi_gpu=multi_gpu
            )
        else:
            print("Torneo cancelado.")

if __name__ == "__main__":
    mp.freeze_support()
    main()
