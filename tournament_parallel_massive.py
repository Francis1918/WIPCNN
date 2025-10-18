#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
tournament_parallel_massive.py - Versión interactiva y ultra-robusta para torneos masivos
Soporta millones de enfrentamientos con interfaz interactiva en tiempo real.

Uso:
    python tournament_parallel_massive.py [--epochs E1 E2 E3...] [--matches N] [--temp T] [--workers W]
    python tournament_parallel_massive.py --all --max 10000

Características:
    - Interfaz interactiva en tiempo real con estadísticas
    - Sistema de reintentos infinitos con cooldown
    - Detección y recuperación automática de bloqueos
    - Monitoreo de recursos en tiempo real
    - Comandos interactivos durante la ejecución
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import itertools
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import time
import platform
import subprocess
import json
import pickle
import gzip
from scipy import sparse
import psutil
import signal
import traceback
import threading
import queue
from collections import deque

# Importar funciones desde compare_agents.py
from compare_agents import compare_agents
from utils.logger import logger

# Importar funciones auxiliares del archivo original
from tournament_parallel import (
    get_cpu_info,
    get_cores_for_parallelism,
    initialize_worker,
    get_all_available_epochs
)

# Variables globales para control
INTERRUPTED = False
PAUSE_REQUESTED = False
STATS_QUEUE = queue.Queue()


def signal_handler(signum, frame):
    """Manejador de señales para interrupciones controladas."""
    global INTERRUPTED
    INTERRUPTED = True
    logger.warning("\n⚠️  Interrupción recibida. Guardando progreso y finalizando de forma segura...")


# Registrar manejador de señales
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def run_match_parallel_safe(args):
    """Función ultra-robusta para ejecutar un enfrentamiento.

    Args:
        args (tuple): (epoch1, epoch2, n_matches, temperature, visualize, tournament_dir, attempt_num)

    Returns:
        tuple: (epoch1, epoch2, match_results, match_data, error_msg)
    """
    epoch1, epoch2, n_matches, temperature, visualize, tournament_dir, attempt_num = args

    match_start = time.time()
    process_id = os.getpid()

    try:
        # Importar aquí para evitar problemas de serialización
        from compare_agents import compare_agents
        import gc

        # Verificar si torch está disponible
        try:
            import torch
            has_torch = True
        except ImportError:
            has_torch = False

        # Realizar el enfrentamiento con timeout interno
        match_results = compare_agents(
            epoch1,
            epoch2,
            n_matches=n_matches,
            temperature=temperature,
            visualize=False
        )

        if not match_results:
            return epoch1, epoch2, None, None, "No se obtuvieron resultados"

        # Extraer resultados
        wins_1 = match_results.get('P1', 0)
        wins_2 = match_results.get('P2', 0)
        draws = match_results.get('Empates', 0)

        # Validar resultados
        total = wins_1 + wins_2 + draws
        if total != n_matches:
            return epoch1, epoch2, None, None, f"Resultados inconsistentes: {total} != {n_matches}"

        # Guardar detalles del enfrentamiento
        match_data = {
            'Epoch1': epoch1,
            'Epoch2': epoch2,
            'Wins_Epoch1': wins_1,
            'Wins_Epoch2': wins_2,
            'Draws': draws,
            'Win_Rate_Epoch1': wins_1 / n_matches * 100 if n_matches > 0 else 0,
            'Win_Rate_Epoch2': wins_2 / n_matches * 100 if n_matches > 0 else 0,
            'Draw_Rate': draws / n_matches * 100 if n_matches > 0 else 0,
            'Duration': time.time() - match_start,
            'Attempt': attempt_num,
            'ProcessID': process_id
        }

        # Limpiar memoria agresivamente
        if has_torch:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except:
                pass

        gc.collect()

        return epoch1, epoch2, match_results, match_data, None

    except KeyboardInterrupt:
        raise
    except Exception as e:
        error_msg = f"Intento {attempt_num} - Error: {str(e)[:200]}"
        return epoch1, epoch2, None, None, error_msg
    finally:
        # Limpieza final agresiva
        try:
            import gc
            gc.collect()
        except:
            pass


class InteractiveStatsMonitor:
    """Monitor interactivo de estadísticas en tiempo real."""

    def __init__(self, total_matches):
        self.total_matches = total_matches
        self.completed = 0
        self.failed = 0
        self.retrying = 0
        self.start_time = time.time()
        self.recent_times = deque(maxlen=100)
        self.lock = threading.Lock()
        self.running = True

    def update(self, completed=0, failed=0, retrying=0, duration=None):
        """Actualiza las estadísticas."""
        with self.lock:
            self.completed += completed
            self.failed += failed
            self.retrying = retrying
            if duration:
                self.recent_times.append(duration)

    def get_stats(self):
        """Obtiene estadísticas actuales."""
        with self.lock:
            elapsed = time.time() - self.start_time
            rate = self.completed / elapsed if elapsed > 0 else 0
            remaining = self.total_matches - self.completed
            eta = remaining / rate if rate > 0 else 0
            avg_time = sum(self.recent_times) / len(self.recent_times) if self.recent_times else 0

            return {
                'completed': self.completed,
                'failed': self.failed,
                'retrying': self.retrying,
                'total': self.total_matches,
                'progress': (self.completed / self.total_matches * 100) if self.total_matches > 0 else 0,
                'rate': rate,
                'eta': eta,
                'elapsed': elapsed,
                'avg_time': avg_time
            }

    def display_loop(self):
        """Loop de visualización interactiva."""
        while self.running and not INTERRUPTED:
            try:
                stats = self.get_stats()

                # Limpiar pantalla (compatible con Windows y Unix)
                os.system('cls' if os.name == 'nt' else 'clear')

                print("=" * 80)
                print("🏆 TORNEO MASIVO - MONITOR EN TIEMPO REAL")
                print("=" * 80)
                print()

                # Barra de progreso ASCII
                bar_width = 50
                filled = int(bar_width * stats['progress'] / 100)
                bar = '█' * filled + '░' * (bar_width - filled)
                print(f"Progreso: [{bar}] {stats['progress']:.2f}%")
                print()

                # Estadísticas principales
                print(f"✅ Completados:  {stats['completed']:,} / {stats['total']:,}")
                print(f"❌ Fallidos:     {stats['failed']:,}")
                print(f"🔄 Reintentando: {stats['retrying']:,}")
                print()

                # Velocidad y tiempo
                print(f"⚡ Velocidad:    {stats['rate']:.2f} enfrentamientos/seg")
                print(f"⏱️  Tiempo prom:  {stats['avg_time']:.2f} seg/enfrentamiento")
                print(f"⏰ Transcurrido: {timedelta(seconds=int(stats['elapsed']))}")
                print(f"🎯 ETA:          {timedelta(seconds=int(stats['eta']))}")
                print()

                # Recursos del sistema
                cpu_percent = psutil.cpu_percent(interval=0.1)
                mem = psutil.virtual_memory()
                print(f"💻 CPU:          {cpu_percent:.1f}%")
                print(
                    f"🧠 RAM:          {mem.percent:.1f}% ({mem.used / (1024 ** 3):.1f} GB / {mem.total / (1024 ** 3):.1f} GB)")
                print()

                print("=" * 80)
                print("Comandos: [Ctrl+C] Detener y guardar | [Espera automática]")
                print("=" * 80)

                time.sleep(2)  # Actualizar cada 2 segundos

            except Exception as e:
                logger.error(f"Error en monitor: {e}")
                time.sleep(5)

    def stop(self):
        """Detiene el monitor."""
        self.running = False


class MassiveTournamentManager:
    """Gestor de torneos masivos con checkpoints y optimización de memoria."""

    def __init__(self, tournament_dir, epochs, n_matches=10, temperature=0.5):
        self.tournament_dir = tournament_dir
        self.epochs = sorted(epochs)
        self.n_matches = n_matches
        self.temperature = temperature
        self.n_agents = len(epochs)

        # Archivos de checkpoint
        self.checkpoint_file = f"{tournament_dir}/checkpoint.pkl.gz"
        self.results_file = f"{tournament_dir}/results_incremental.pkl.gz"
        self.matches_file = f"{tournament_dir}/matches_incremental.jsonl.gz"
        self.errors_file = f"{tournament_dir}/errors.log"

        # Índice de épocas para acceso rápido
        self.epoch_to_idx = {epoch: i for i, epoch in enumerate(epochs)}

        # Matriz sparse para resultados (ahorra memoria)
        self.wins_matrix = sparse.lil_matrix((self.n_agents, self.n_agents), dtype=np.int32)
        self.draws_array = np.zeros(self.n_agents, dtype=np.int32)

        # Estadísticas
        self.completed_matches = set()
        self.failed_matches = {}
        self.retry_counts = {}  # Contador de reintentos por enfrentamiento
        self.start_time = None
        self.last_save_time = time.time()

        # Lock para operaciones thread-safe
        self.lock = threading.Lock()

    def estimate_resources(self):
        """Estima tiempo y recursos necesarios para el torneo."""
        total_matches = self.n_agents * (self.n_agents - 1) // 2

        # Estimación de memoria
        estimated_memory_mb = (self.n_agents ** 2 * 8) / (1024 * 1024)

        # Estimación de tiempo (asumiendo 2 segundos por enfrentamiento)
        cpu_info = get_cpu_info()
        n_workers = cpu_info['logical_cores']
        estimated_time_hours = (total_matches * 2) / (n_workers * 3600)

        # Estimación de espacio en disco
        estimated_disk_mb = (total_matches * 500) / (1024 * 1024)

        return {
            'total_matches': total_matches,
            'estimated_memory_mb': estimated_memory_mb,
            'estimated_time_hours': estimated_time_hours,
            'estimated_disk_mb': estimated_disk_mb,
            'n_workers': n_workers,
            'available_memory_gb': psutil.virtual_memory().available / (1024 ** 3)
        }

    def load_checkpoint(self):
        """Carga el estado desde un checkpoint si existe."""
        if os.path.exists(self.checkpoint_file):
            try:
                with gzip.open(self.checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)

                self.wins_matrix = checkpoint['wins_matrix']
                self.draws_array = checkpoint['draws_array']
                self.completed_matches = checkpoint['completed_matches']
                self.failed_matches = checkpoint.get('failed_matches', {})
                self.retry_counts = checkpoint.get('retry_counts', {})
                self.start_time = checkpoint.get('start_time', time.time())

                logger.info(f"✅ Checkpoint cargado: {len(self.completed_matches)} enfrentamientos completados")
                return True
            except Exception as e:
                logger.error(f"Error al cargar checkpoint: {e}")
                logger.debug(traceback.format_exc())
                return False
        return False

    def save_checkpoint(self, force=False):
        """Guarda el estado actual en un checkpoint."""
        current_time = time.time()
        if not force and (current_time - self.last_save_time) < 30:
            return

        with self.lock:
            try:
                checkpoint = {
                    'wins_matrix': self.wins_matrix,
                    'draws_array': self.draws_array,
                    'completed_matches': self.completed_matches,
                    'failed_matches': self.failed_matches,
                    'retry_counts': self.retry_counts,
                    'start_time': self.start_time,
                    'timestamp': datetime.now().isoformat()
                }

                # Guardar con compresión
                temp_file = f"{self.checkpoint_file}.tmp"
                with gzip.open(temp_file, 'wb') as f:
                    pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)

                # Reemplazar archivo original de forma atómica
                if os.path.exists(self.checkpoint_file):
                    backup_file = f"{self.checkpoint_file}.bak"
                    os.replace(self.checkpoint_file, backup_file)
                os.replace(temp_file, self.checkpoint_file)

                self.last_save_time = current_time
                logger.debug(f"Checkpoint guardado: {len(self.completed_matches)} enfrentamientos")
            except Exception as e:
                logger.error(f"Error al guardar checkpoint: {e}")
                logger.debug(traceback.format_exc())

    def save_match_result(self, epoch1, epoch2, match_results, match_data):
        """Guarda el resultado de un enfrentamiento de forma incremental."""
        with self.lock:
            try:
                # Actualizar matriz de victorias
                idx1 = self.epoch_to_idx[epoch1]
                idx2 = self.epoch_to_idx[epoch2]

                wins_1 = match_results.get('P1', 0)
                wins_2 = match_results.get('P2', 0)
                draws = match_results.get('Empates', 0)

                self.wins_matrix[idx1, idx2] = wins_1
                self.wins_matrix[idx2, idx1] = wins_2
                self.draws_array[idx1] += draws
                self.draws_array[idx2] += draws

                # Marcar como completado
                match_key = tuple(sorted([epoch1, epoch2]))
                self.completed_matches.add(match_key)

                # Remover de fallidos si estaba ahí
                if match_key in self.failed_matches:
                    del self.failed_matches[match_key]
                if match_key in self.retry_counts:
                    del self.retry_counts[match_key]

                # Guardar detalles en archivo JSONL comprimido
                with gzip.open(self.matches_file, 'at', encoding='utf-8') as f:
                    json.dump(match_data, f)
                    f.write('\n')

            except Exception as e:
                logger.error(f"Error al guardar resultado de {epoch1} vs {epoch2}: {e}")
                logger.debug(traceback.format_exc())

    def log_error(self, epoch1, epoch2, error_msg):
        """Registra un error en el archivo de errores."""
        try:
            with open(self.errors_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().isoformat()
                f.write(f"[{timestamp}] {epoch1} vs {epoch2}: {error_msg}\n")
        except Exception as e:
            logger.error(f"Error al escribir log de errores: {e}")

    def get_pending_matches(self):
        """Obtiene la lista de enfrentamientos pendientes."""
        all_matches = set(itertools.combinations(self.epochs, 2))
        pending = all_matches - self.completed_matches
        return list(pending)

    def should_retry(self, match_key):
        """Determina si un enfrentamiento debe reintentarse."""
        # Siempre reintentar, pero con cooldown progresivo
        return True

    def get_retry_delay(self, match_key):
        """Calcula el delay antes de reintentar un enfrentamiento."""
        retry_count = self.retry_counts.get(match_key, 0)
        # Cooldown progresivo: 5, 10, 20, 40, 60 (máximo) segundos
        delay = min(5 * (2 ** retry_count), 60)
        return delay

    def increment_retry(self, match_key):
        """Incrementa el contador de reintentos."""
        with self.lock:
            self.retry_counts[match_key] = self.retry_counts.get(match_key, 0) + 1

    def compute_final_results(self):
        """Calcula los resultados finales del torneo."""
        logger.info("Calculando resultados finales...")

        # Convertir matriz sparse a densa para cálculos finales
        wins_dense = self.wins_matrix.toarray()

        # Calcular estadísticas por agente
        results = []
        for i, epoch in enumerate(self.epochs):
            wins = np.sum(wins_dense[i, :])
            losses = np.sum(wins_dense[:, i])
            draws = self.draws_array[i]
            total_games = wins + losses + draws
            points = wins * 3 + draws * 1

            results.append({
                'Época': epoch,
                'Victorias': int(wins),
                'Derrotas': int(losses),
                'Empates': int(draws),
                'Puntos': int(points),
                'Partidas': int(total_games)
            })

        # Crear DataFrame y ordenar
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Puntos', ascending=False).reset_index(drop=True)
        results_df['Posición'] = range(1, len(results_df) + 1)

        return results_df

    def save_final_results(self, results_df):
        """Guarda los resultados finales."""
        results_dir = f"{self.tournament_dir}/results"
        os.makedirs(results_dir, exist_ok=True)

        try:
            # Guardar tabla de posiciones
            results_df.to_csv(f"{results_dir}/positions.csv", index=False)

            # Guardar matriz de victorias completa (comprimida)
            with gzip.open(f"{results_dir}/wins_matrix.pkl.gz", 'wb') as f:
                pickle.dump({
                    'wins_matrix': self.wins_matrix,
                    'draws_array': self.draws_array,
                    'epochs': self.epochs
                }, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Guardar resumen de errores si hay
            if self.failed_matches:
                with open(f"{results_dir}/failed_matches.txt", 'w', encoding='utf-8') as f:
                    f.write(f"Total de enfrentamientos con errores: {len(self.failed_matches)}\n\n")
                    for match_key, error in self.failed_matches.items():
                        retries = self.retry_counts.get(match_key, 0)
                        f.write(f"{match_key[0]} vs {match_key[1]} (Reintentos: {retries}): {error}\n")

            logger.info(f"✅ Resultados guardados en {results_dir}")
        except Exception as e:
            logger.error(f"Error al guardar resultados finales: {e}")
            logger.debug(traceback.format_exc())


def run_massive_tournament(epochs, n_matches=10, temperature=0.5, visualize=False,
                           n_workers=None, physical_only=False, specific_cores=None,
                           checkpoint_interval=50, interactive=True):
    """
    Ejecuta un torneo masivo optimizado para miles de épocas.
    """
    global INTERRUPTED

    if len(epochs) < 2:
        logger.error("Se necesitan al menos 2 épocas para un torneo")
        return None

    # Determinar número de trabajadores
    if specific_cores is not None:
        n_workers = len(specific_cores)
    elif n_workers is None:
        n_workers = get_cores_for_parallelism(physical_only)

    # Crear estructura de directorios
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    program_name = Path(__file__).stem
    main_program_dir = program_name
    os.makedirs(main_program_dir, exist_ok=True)

    tournament_dir = f"{main_program_dir}/tournament_{timestamp}"
    os.makedirs(tournament_dir, exist_ok=True)

    matches_dir = f"{tournament_dir}/matches"
    os.makedirs(matches_dir, exist_ok=True)

    # Inicializar gestor de torneo
    manager = MassiveTournamentManager(tournament_dir, epochs, n_matches, temperature)

    # Estimar recursos
    estimates = manager.estimate_resources()
    total_matches = estimates['total_matches']

    # Cargar checkpoint si existe
    resumed = manager.load_checkpoint()
    if resumed:
        logger.info(f"🔄 Reanudando torneo desde checkpoint")
    else:
        manager.start_time = time.time()

    # Inicializar monitor interactivo
    monitor = InteractiveStatsMonitor(total_matches)
    monitor_thread = None

    if interactive:
        monitor_thread = threading.Thread(target=monitor.display_loop, daemon=True)
        monitor_thread.start()
    else:
        logger.info("\n" + "=" * 70)
        logger.info("ESTIMACIÓN DE RECURSOS")
        logger.info("=" * 70)
        logger.info(f"Número de agentes: {len(epochs)}")
        logger.info(f"Total de enfrentamientos: {total_matches:,}")
        logger.info(f"Trabajadores paralelos: {n_workers}")

    # Obtener enfrentamientos pendientes
    pending_matches = manager.get_pending_matches()
    completed_before = len(manager.completed_matches)

    if len(pending_matches) == 0:
        logger.info("✅ Todos los enfrentamientos ya están completados")
        results_df = manager.compute_final_results()
        manager.save_final_results(results_df)
        if monitor_thread:
            monitor.stop()
        return results_df

    # Ejecutar enfrentamientos en paralelo con reintentos infinitos
    completed_count = 0
    last_checkpoint = 0
    active_futures = {}
    retry_queue = deque()

    try:
        with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=initialize_worker if specific_cores else None,
                initargs=(specific_cores,) if specific_cores else ()
        ) as executor:

            # Función para enviar un trabajo
            def submit_match(match_pair, attempt_num=1):
                epoch1, epoch2 = match_pair
                args = (epoch1, epoch2, n_matches, temperature, False, matches_dir, attempt_num)
                future = executor.submit(run_match_parallel_safe, args)
                active_futures[future] = (match_pair, attempt_num, time.time())
                return future

            # Enviar trabajos iniciales
            for match_pair in pending_matches[:n_workers * 2]:  # Buffer inicial
                submit_match(match_pair)

            pending_to_submit = pending_matches[n_workers * 2:]

            # Loop principal de procesamiento
            while (active_futures or pending_to_submit or retry_queue) and not INTERRUPTED:

                # Procesar reintentos con cooldown
                current_time = time.time()
                while retry_queue:
                    match_pair, retry_time, attempt_num = retry_queue[0]
                    if current_time >= retry_time:
                        retry_queue.popleft()
                        submit_match(match_pair, attempt_num)
                    else:
                        break

                # Enviar más trabajos si hay espacio
                while len(active_futures) < n_workers * 2 and pending_to_submit:
                    match_pair = pending_to_submit.pop(0)
                    submit_match(match_pair)

                # Procesar resultados completados
                done_futures = []
                for future in list(active_futures.keys()):
                    if future.done():
                        done_futures.append(future)

                for future in done_futures:
                    match_pair, attempt_num, submit_time = active_futures.pop(future)
                    epoch1, epoch2 = match_pair
                    match_key = tuple(sorted([epoch1, epoch2]))

                    try:
                        # Timeout de 300 segundos por enfrentamiento
                        epoch1, epoch2, match_results, match_data, error_msg = future.result(timeout=300)

                        if match_results is not None:
                            # Éxito - guardar resultado
                            manager.save_match_result(epoch1, epoch2, match_results, match_data)
                            completed_count += 1
                            monitor.update(completed=1, duration=match_data['Duration'])

                            # Guardar checkpoint periódicamente
                            if completed_count - last_checkpoint >= checkpoint_interval:
                                manager.save_checkpoint()
                                last_checkpoint = completed_count
                        else:
                            # Fallo - programar reintento
                            manager.log_error(epoch1, epoch2, error_msg or "Error desconocido")
                            manager.increment_retry(match_key)

                            delay = manager.get_retry_delay(match_key)
                            retry_time = time.time() + delay
                            retry_queue.append((match_pair, retry_time, attempt_num + 1))

                            monitor.update(failed=1, retrying=len(retry_queue))

                    except TimeoutError:
                        # Timeout - reintentar
                        logger.warning(f"Timeout en {epoch1} vs {epoch2} (intento {attempt_num})")
                        manager.log_error(epoch1, epoch2, f"Timeout en intento {attempt_num}")
                        manager.increment_retry(match_key)

                        delay = manager.get_retry_delay(match_key)
                        retry_time = time.time() + delay
                        retry_queue.append((match_pair, retry_time, attempt_num + 1))

                        monitor.update(failed=1, retrying=len(retry_queue))

                    except Exception as e:
                        # Error inesperado - reintentar
                        logger.error(f"Error procesando {epoch1} vs {epoch2}: {e}")
                        manager.log_error(epoch1, epoch2, str(e))
                        manager.increment_retry(match_key)

                        delay = manager.get_retry_delay(match_key)
                        retry_time = time.time() + delay
                        retry_queue.append((match_pair, retry_time, attempt_num + 1))

                        monitor.update(failed=1, retrying=len(retry_queue))

                # Pequeña pausa para no saturar el CPU
                time.sleep(0.1)

            # Cancelar trabajos pendientes si se interrumpió
            if INTERRUPTED:
                logger.warning("Cancelando trabajos pendientes...")
                for future in active_futures.keys():
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)

    except KeyboardInterrupt:
        INTERRUPTED = True
        logger.warning("Interrupción detectada en el bucle principal...")

    finally:
        # Detener monitor
        if monitor_thread:
            monitor.stop()
            monitor_thread.join(timeout=2)

        # Guardar checkpoint final siempre
        logger.info("Guardando checkpoint final...")
        manager.save_checkpoint(force=True)

    # Calcular y guardar resultados finales
    logger.info("\n" + "=" * 70)
    logger.info("CALCULANDO RESULTADOS FINALES")
    logger.info("=" * 70)

    results_df = manager.compute_final_results()
    manager.save_final_results(results_df)

    # Estadísticas finales
    total_time = time.time() - manager.start_time
    logger.info("\n" + "=" * 70)
    if INTERRUPTED:
        logger.info("TORNEO INTERRUMPIDO (Progreso guardado)")
    else:
        logger.info("TORNEO COMPLETADO")
    logger.info("=" * 70)
    logger.info(f"Tiempo total: {timedelta(seconds=int(total_time))}")
    logger.info(f"Enfrentamientos completados: {len(manager.completed_matches):,}")
    logger.info(f"Enfrentamientos con errores: {len(manager.failed_matches):,}")
    if total_time > 0:
        logger.info(f"Velocidad promedio: {len(manager.completed_matches) / total_time:.2f} enfrentamientos/segundo")

    # Mostrar top 10
    if len(results_df) > 0:
        logger.info("\n🏆 TOP 10 AGENTES:")
        logger.info("\n" + results_df.head(10).to_string(index=False))

        # Campeón
        champion = results_df.iloc[0]
        logger.info(f"\n🥇 CAMPEÓN: Época {champion['Época']}")
        logger.info(f"   Victorias: {champion['Victorias']}")
        logger.info(f"   Puntos: {champion['Puntos']}")

    if INTERRUPTED:
        logger.info(f"\n💾 Para reanudar, ejecuta el mismo comando nuevamente")

    return results_df


def main():
    """Función principal para ejecutar el torneo masivo."""
    parser = argparse.ArgumentParser(
        description="Torneo paralelo masivo interactivo para agentes de Quarto",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--epochs", type=int, nargs='+', help="Lista de épocas para el torneo")
    parser.add_argument("--all", action="store_true", help="Usar todas las épocas disponibles")
    parser.add_argument("--max", type=int, default=10000, help="Número máximo de agentes (default: 10000)")
    parser.add_argument("--matches", type=int, default=10, help="Partidas por enfrentamiento (default: 10)")
    parser.add_argument("--temp", type=float, default=0.5, help="Temperatura (default: 0.5)")
    parser.add_argument("--workers", type=int, default=None, help="Número de trabajadores paralelos")
    parser.add_argument("--cores", type=str, help="Núcleos específicos (ej: 0,1,2,5)")
    parser.add_argument("--physical-only", action="store_true", help="Usar solo núcleos físicos")
    parser.add_argument("--checkpoint-interval", type=int, default=50,
                        help="Guardar checkpoint cada N enfrentamientos (default: 50)")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Desactivar interfaz interactiva")
    parser.add_argument("--estimate-only", action="store_true",
                        help="Solo mostrar estimación de recursos sin ejecutar")

    args = parser.parse_args()

    try:
        # Obtener épocas
        if args.all:
            all_epochs = get_all_available_epochs()
            epochs = all_epochs[:args.max] if len(all_epochs) > args.max else all_epochs
            if not epochs:
                logger.error("No se encontraron épocas disponibles")
                return 1
            logger.info(f"Usando {len(epochs)} épocas de {len(all_epochs)} disponibles")
        elif args.epochs:
            epochs = args.epochs
        else:
            parser.print_help()
            return 0

        # Procesar núcleos específicos
        specific_cores = None
        if args.cores:
            try:
                specific_cores = [int(c.strip()) for c in args.cores.split(',')]
            except ValueError:
                logger.error(f"Formato inválido para núcleos: {args.cores}")
                return 1

        # Solo estimación
        if args.estimate_only:
            manager = MassiveTournamentManager("temp", epochs, args.matches, args.temp)
            estimates = manager.estimate_resources()

            print("\n" + "=" * 70)
            print("ESTIMACIÓN DE RECURSOS PARA EL TORNEO")
            print("=" * 70)
            print(f"Número de agentes: {len(epochs):,}")
            print(f"Total de enfrentamientos: {estimates['total_matches']:,}")
            print(f"Memoria estimada: {estimates['estimated_memory_mb']:.1f} MB")
            print(f"Tiempo estimado: {estimates['estimated_time_hours']:.1f} horas")
            print(f"Espacio en disco estimado: {estimates['estimated_disk_mb']:.1f} MB")
            print(f"Memoria disponible: {estimates['available_memory_gb']:.1f} GB")
            print(f"Trabajadores paralelos: {estimates['n_workers']}")
            print("=" * 70)
            return 0

        # Ejecutar torneo
        result = run_massive_tournament(
            epochs=epochs,
            n_matches=args.matches,
            temperature=args.temp,
            visualize=False,
            n_workers=args.workers,
            physical_only=args.physical_only,
            specific_cores=specific_cores,
            checkpoint_interval=args.checkpoint_interval,
            interactive=not args.no_interactive
        )

        return 0 if result is not None else 1

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Programa interrumpido por el usuario")
        return 130
    except Exception as e:
        logger.error(f"Error fatal: {e}")
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())