#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
tournament_interactive_console.py - Versión 100% interactiva con menú de consola
Interfaz amigable para configurar y ejecutar torneos masivos.

Características:
    - Menú interactivo completo
    - Configuración paso a paso
    - Validación de entradas
    - Visualización en tiempo real
    - Sistema de reintentos infinitos
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

# Intentar importar funciones auxiliares del archivo original
try:
    from tournament_parallel import (
        get_cores_for_parallelism,
        initialize_worker,
        get_all_available_epochs
    )
except ImportError:
    # Definir funciones básicas si no están disponibles
    def get_cores_for_parallelism(physical_only=False):
        if physical_only:
            return psutil.cpu_count(logical=False) or 1
        return psutil.cpu_count(logical=True) or 1


    def initialize_worker(cores=None):
        pass


    def get_all_available_epochs():
        """Busca todas las épocas disponibles en el directorio de modelos."""
        models_dir = Path("models")
        if not models_dir.exists():
            return []

        epochs = []
        for file in models_dir.glob("*.pth"):
            try:
                epoch_num = int(file.stem.split('_')[-1])
                epochs.append(epoch_num)
            except (ValueError, IndexError):
                continue

        return sorted(epochs)


def get_cpu_info():
    """Obtiene información detallada de la CPU de forma robusta."""
    try:
        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)

        # Intentar obtener el nombre de la CPU
        cpu_brand = "CPU Desconocida"

        try:
            if platform.system() == "Windows":
                cpu_brand = platform.processor()
            elif platform.system() == "Darwin":  # macOS
                command = "sysctl -n machdep.cpu.brand_string"
                cpu_brand = subprocess.check_output(command, shell=True).decode().strip()
            elif platform.system() == "Linux":
                command = "cat /proc/cpuinfo | grep 'model name' | uniq"
                output = subprocess.check_output(command, shell=True).decode().strip()
                if output:
                    cpu_brand = output.split(':')[1].strip()
        except Exception:
            # Si falla, usar información básica
            cpu_brand = f"{platform.processor() or platform.machine() or 'CPU'}"

        # Si aún no tenemos un nombre válido, usar uno genérico
        if not cpu_brand or cpu_brand.strip() == "":
            cpu_brand = f"CPU con {logical_cores} núcleos"

        return {
            'brand': cpu_brand,
            'physical_cores': physical_cores or 1,
            'logical_cores': logical_cores or 1,
            'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else 0
        }
    except Exception as e:
        # Fallback completo
        return {
            'brand': 'CPU Desconocida',
            'physical_cores': 1,
            'logical_cores': 1,
            'frequency': 0
        }


# Variables globales para control
INTERRUPTED = False
PAUSE_REQUESTED = False


def signal_handler(signum, frame):
    """Manejador de señales para interrupciones controladas."""
    global INTERRUPTED
    INTERRUPTED = True
    logger.warning("\n⚠️  Interrupción recibida. Guardando progreso y finalizando de forma segura...")


# Registrar manejador de señales
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def clear_screen():
    """Limpia la pantalla de la consola."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(title):
    """Imprime un encabezado decorado."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_menu(title, options):
    """Imprime un menú con opciones numeradas."""
    print_header(title)
    for i, option in enumerate(options, 1):
        print(f"  [{i}] {option}")
    print(f"  [0] Salir")
    print()


def get_input(prompt, input_type=str, default=None, validator=None):
    """Obtiene entrada del usuario con validación."""
    while True:
        try:
            if default is not None:
                user_input = input(f"{prompt} [{default}]: ").strip()
                if not user_input:
                    return default
            else:
                user_input = input(f"{prompt}: ").strip()

            if not user_input and default is None:
                print("❌ Este campo es obligatorio. Intenta de nuevo.")
                continue

            # Convertir al tipo deseado
            if input_type == int:
                value = int(user_input)
            elif input_type == float:
                value = float(user_input)
            elif input_type == bool:
                value = user_input.lower() in ['s', 'si', 'sí', 'y', 'yes', '1', 'true']
            elif input_type == list:
                # Para listas de enteros separados por comas
                value = [int(x.strip()) for x in user_input.split(',')]
            else:
                value = user_input

            # Validar si hay validador
            if validator and not validator(value):
                print("❌ Valor inválido. Intenta de nuevo.")
                continue

            return value

        except ValueError:
            print(f"❌ Entrada inválida. Se esperaba {input_type.__name__}. Intenta de nuevo.")
        except KeyboardInterrupt:
            print("\n⚠️  Operación cancelada.")
            return None


def run_match_parallel_safe(args):
    """Función ultra-robusta para ejecutar un enfrentamiento."""
    epoch1, epoch2, n_matches, temperature, visualize, tournament_dir, attempt_num = args

    match_start = time.time()
    process_id = os.getpid()

    try:
        from compare_agents import compare_agents
        import gc

        try:
            import torch
            has_torch = True
        except ImportError:
            has_torch = False

        match_results = compare_agents(
            epoch1,
            epoch2,
            n_matches=n_matches,
            temperature=temperature,
            visualize=False
        )

        if not match_results:
            return epoch1, epoch2, None, None, "No se obtuvieron resultados"

        wins_1 = match_results.get('P1', 0)
        wins_2 = match_results.get('P2', 0)
        draws = match_results.get('Empates', 0)

        total = wins_1 + wins_2 + draws
        if total != n_matches:
            return epoch1, epoch2, None, None, f"Resultados inconsistentes: {total} != {n_matches}"

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

                clear_screen()

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
                print("Comandos: [Ctrl+C] Detener y guardar")
                print("=" * 80)

                time.sleep(2)

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

        self.checkpoint_file = f"{tournament_dir}/checkpoint.pkl.gz"
        self.results_file = f"{tournament_dir}/results_incremental.pkl.gz"
        self.matches_file = f"{tournament_dir}/matches_incremental.jsonl.gz"
        self.errors_file = f"{tournament_dir}/errors.log"

        self.epoch_to_idx = {epoch: i for i, epoch in enumerate(epochs)}

        self.wins_matrix = sparse.lil_matrix((self.n_agents, self.n_agents), dtype=np.int32)
        self.draws_array = np.zeros(self.n_agents, dtype=np.int32)

        self.completed_matches = set()
        self.failed_matches = {}
        self.retry_counts = {}
        self.start_time = None
        self.last_save_time = time.time()

        self.lock = threading.Lock()

    def estimate_resources(self):
        """Estima tiempo y recursos necesarios para el torneo."""
        total_matches = self.n_agents * (self.n_agents - 1) // 2

        estimated_memory_mb = (self.n_agents ** 2 * 8) / (1024 * 1024)

        cpu_info = get_cpu_info()
        n_workers = cpu_info['logical_cores']
        estimated_time_hours = (total_matches * 2) / (n_workers * 3600)

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

                temp_file = f"{self.checkpoint_file}.tmp"
                with gzip.open(temp_file, 'wb') as f:
                    pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)

                if os.path.exists(self.checkpoint_file):
                    backup_file = f"{self.checkpoint_file}.bak"
                    os.replace(self.checkpoint_file, backup_file)
                os.replace(temp_file, self.checkpoint_file)

                self.last_save_time = current_time
            except Exception as e:
                logger.error(f"Error al guardar checkpoint: {e}")

    def save_match_result(self, epoch1, epoch2, match_results, match_data):
        """Guarda el resultado de un enfrentamiento de forma incremental."""
        with self.lock:
            try:
                idx1 = self.epoch_to_idx[epoch1]
                idx2 = self.epoch_to_idx[epoch2]

                wins_1 = match_results.get('P1', 0)
                wins_2 = match_results.get('P2', 0)
                draws = match_results.get('Empates', 0)

                self.wins_matrix[idx1, idx2] = wins_1
                self.wins_matrix[idx2, idx1] = wins_2
                self.draws_array[idx1] += draws
                self.draws_array[idx2] += draws

                match_key = tuple(sorted([epoch1, epoch2]))
                self.completed_matches.add(match_key)

                if match_key in self.failed_matches:
                    del self.failed_matches[match_key]
                if match_key in self.retry_counts:
                    del self.retry_counts[match_key]

                with gzip.open(self.matches_file, 'at', encoding='utf-8') as f:
                    json.dump(match_data, f)
                    f.write('\n')

            except Exception as e:
                logger.error(f"Error al guardar resultado de {epoch1} vs {epoch2}: {e}")

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

    def get_retry_delay(self, match_key):
        """Calcula el delay antes de reintentar un enfrentamiento."""
        retry_count = self.retry_counts.get(match_key, 0)
        delay = min(5 * (2 ** retry_count), 60)
        return delay

    def increment_retry(self, match_key):
        """Incrementa el contador de reintentos."""
        with self.lock:
            self.retry_counts[match_key] = self.retry_counts.get(match_key, 0) + 1

    def compute_final_results(self):
        """Calcula los resultados finales del torneo."""
        logger.info("Calculando resultados finales...")

        wins_dense = self.wins_matrix.toarray()

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

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Puntos', ascending=False).reset_index(drop=True)
        results_df['Posición'] = range(1, len(results_df) + 1)

        return results_df

    def save_final_results(self, results_df):
        """Guarda los resultados finales."""
        results_dir = f"{self.tournament_dir}/results"
        os.makedirs(results_dir, exist_ok=True)

        try:
            results_df.to_csv(f"{results_dir}/positions.csv", index=False)

            with gzip.open(f"{results_dir}/wins_matrix.pkl.gz", 'wb') as f:
                pickle.dump({
                    'wins_matrix': self.wins_matrix,
                    'draws_array': self.draws_array,
                    'epochs': self.epochs
                }, f, protocol=pickle.HIGHEST_PROTOCOL)

            if self.failed_matches:
                with open(f"{results_dir}/failed_matches.txt", 'w', encoding='utf-8') as f:
                    f.write(f"Total de enfrentamientos con errores: {len(self.failed_matches)}\n\n")
                    for match_key, error in self.failed_matches.items():
                        retries = self.retry_counts.get(match_key, 0)
                        f.write(f"{match_key[0]} vs {match_key[1]} (Reintentos: {retries}): {error}\n")

            logger.info(f"✅ Resultados guardados en {results_dir}")
        except Exception as e:
            logger.error(f"Error al guardar resultados finales: {e}")


def run_massive_tournament(epochs, n_matches=10, temperature=0.5, n_workers=None,
                           physical_only=False, specific_cores=None, checkpoint_interval=50):
    """Ejecuta un torneo masivo optimizado."""
    global INTERRUPTED

    if len(epochs) < 2:
        logger.error("Se necesitan al menos 2 épocas para un torneo")
        return None

    if specific_cores is not None:
        n_workers = len(specific_cores)
    elif n_workers is None:
        n_workers = get_cores_for_parallelism(physical_only)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    program_name = Path(__file__).stem
    main_program_dir = program_name
    os.makedirs(main_program_dir, exist_ok=True)

    tournament_dir = f"{main_program_dir}/tournament_{timestamp}"
    os.makedirs(tournament_dir, exist_ok=True)

    matches_dir = f"{tournament_dir}/matches"
    os.makedirs(matches_dir, exist_ok=True)

    manager = MassiveTournamentManager(tournament_dir, epochs, n_matches, temperature)

    estimates = manager.estimate_resources()
    total_matches = estimates['total_matches']

    resumed = manager.load_checkpoint()
    if resumed:
        logger.info(f"🔄 Reanudando torneo desde checkpoint")
    else:
        manager.start_time = time.time()

    monitor = InteractiveStatsMonitor(total_matches)
    monitor_thread = threading.Thread(target=monitor.display_loop, daemon=True)
    monitor_thread.start()

    pending_matches = manager.get_pending_matches()

    if len(pending_matches) == 0:
        logger.info("✅ Todos los enfrentamientos ya están completados")
        results_df = manager.compute_final_results()
        manager.save_final_results(results_df)
        monitor.stop()
        return results_df

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

            def submit_match(match_pair, attempt_num=1):
                epoch1, epoch2 = match_pair
                args = (epoch1, epoch2, n_matches, temperature, False, matches_dir, attempt_num)
                future = executor.submit(run_match_parallel_safe, args)
                active_futures[future] = (match_pair, attempt_num, time.time())
                return future

            for match_pair in pending_matches[:n_workers * 2]:
                submit_match(match_pair)

            pending_to_submit = pending_matches[n_workers * 2:]

            while (active_futures or pending_to_submit or retry_queue) and not INTERRUPTED:

                current_time = time.time()
                while retry_queue:
                    match_pair, retry_time, attempt_num = retry_queue[0]
                    if current_time >= retry_time:
                        retry_queue.popleft()
                        submit_match(match_pair, attempt_num)
                    else:
                        break

                while len(active_futures) < n_workers * 2 and pending_to_submit:
                    match_pair = pending_to_submit.pop(0)
                    submit_match(match_pair)

                done_futures = []
                for future in list(active_futures.keys()):
                    if future.done():
                        done_futures.append(future)

                for future in done_futures:
                    match_pair, attempt_num, submit_time = active_futures.pop(future)
                    epoch1, epoch2 = match_pair
                    match_key = tuple(sorted([epoch1, epoch2]))

                    try:
                        epoch1, epoch2, match_results, match_data, error_msg = future.result(timeout=300)

                        if match_results is not None:
                            manager.save_match_result(epoch1, epoch2, match_results, match_data)
                            completed_count += 1
                            monitor.update(completed=1, duration=match_data['Duration'])

                            if completed_count - last_checkpoint >= checkpoint_interval:
                                manager.save_checkpoint()
                                last_checkpoint = completed_count
                        else:
                            manager.log_error(epoch1, epoch2, error_msg or "Error desconocido")
                            manager.increment_retry(match_key)

                            delay = manager.get_retry_delay(match_key)
                            retry_time = time.time() + delay
                            retry_queue.append((match_pair, retry_time, attempt_num + 1))

                            monitor.update(failed=1, retrying=len(retry_queue))

                    except TimeoutError:
                        logger.warning(f"Timeout en {epoch1} vs {epoch2} (intento {attempt_num})")
                        manager.log_error(epoch1, epoch2, f"Timeout en intento {attempt_num}")
                        manager.increment_retry(match_key)

                        delay = manager.get_retry_delay(match_key)
                        retry_time = time.time() + delay
                        retry_queue.append((match_pair, retry_time, attempt_num + 1))

                        monitor.update(failed=1, retrying=len(retry_queue))

                    except Exception as e:
                        logger.error(f"Error procesando {epoch1} vs {epoch2}: {e}")
                        manager.log_error(epoch1, epoch2, str(e))
                        manager.increment_retry(match_key)

                        delay = manager.get_retry_delay(match_key)
                        retry_time = time.time() + delay
                        retry_queue.append((match_pair, retry_time, attempt_num + 1))

                        monitor.update(failed=1, retrying=len(retry_queue))

                time.sleep(0.1)

            if INTERRUPTED:
                logger.warning("Cancelando trabajos pendientes...")
                for future in active_futures.keys():
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)

    except KeyboardInterrupt:
        INTERRUPTED = True
        logger.warning("Interrupción detectada en el bucle principal...")

    finally:
        monitor.stop()
        monitor_thread.join(timeout=2)

        logger.info("Guardando checkpoint final...")
        manager.save_checkpoint(force=True)

    logger.info("\n" + "=" * 70)
    logger.info("CALCULANDO RESULTADOS FINALES")
    logger.info("=" * 70)

    results_df = manager.compute_final_results()
    manager.save_final_results(results_df)

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

    if len(results_df) > 0:
        logger.info("\n🏆 TOP 10 AGENTES:")
        logger.info("\n" + results_df.head(10).to_string(index=False))

        champion = results_df.iloc[0]
        logger.info(f"\n🥇 CAMPEÓN: Época {champion['Época']}")
        logger.info(f"   Victorias: {champion['Victorias']}")
        logger.info(f"   Puntos: {champion['Puntos']}")

    return results_df


def interactive_menu():
    """Menú interactivo principal."""
    clear_screen()

    print("=" * 80)
    print("  🏆 TORNEO MASIVO DE AGENTES - MODO INTERACTIVO")
    print("=" * 80)
    print()
    print("  Bienvenido al sistema de torneos masivos para agentes de Quarto.")
    print("  Este asistente te guiará paso a paso en la configuración del torneo.")
    print()
    input("  Presiona ENTER para continuar...")

    # Paso 1: Seleccionar épocas
    clear_screen()
    print_header("PASO 1: SELECCIÓN DE ÉPOCAS")

    print("¿Cómo deseas seleccionar las épocas?")
    print()
    print("  [1] Usar todas las épocas disponibles")
    print("  [2] Ingresar épocas manualmente")
    print("  [3] Usar un rango de épocas")
    print()

    selection_mode = get_input("Selecciona una opción", int, validator=lambda x: 1 <= x <= 3)
    if selection_mode is None:
        return

    epochs = []

    if selection_mode == 1:
        all_epochs = get_all_available_epochs()
        if not all_epochs:
            print("\n❌ No se encontraron épocas disponibles.")
            input("Presiona ENTER para salir...")
            return

        print(f"\n✅ Se encontraron {len(all_epochs)} épocas disponibles.")
        print(f"   Rango: {min(all_epochs)} - {max(all_epochs)}")

        max_epochs = get_input("\n¿Cuántas épocas deseas usar? (0 = todas)", int, default=0,
                               validator=lambda x: x >= 0)
        if max_epochs is None:
            return

        if max_epochs == 0 or max_epochs >= len(all_epochs):
            epochs = all_epochs
        else:
            epochs = all_epochs[:max_epochs]

        print(f"\n✅ Se usarán {len(epochs)} épocas")

    elif selection_mode == 2:
        print("\nIngresa las épocas separadas por comas (ej: 100,200,300,400)")
        epochs = get_input("Épocas", list)
        if epochs is None:
            return

        print(f"\n✅ Se usarán {len(epochs)} épocas: {epochs}")

    elif selection_mode == 3:
        print("\nIngresa el rango de épocas:")
        start = get_input("  Época inicial", int, validator=lambda x: x > 0)
        if start is None:
            return

        end = get_input("  Época final", int, validator=lambda x: x >= start)
        if end is None:
            return

        step = get_input("  Paso (incremento)", int, default=100, validator=lambda x: x > 0)
        if step is None:
            return

        epochs = list(range(start, end + 1, step))
        print(f"\n✅ Se usarán {len(epochs)} épocas: {epochs[:5]}{'...' if len(epochs) > 5 else ''}")

    if len(epochs) < 2:
        print("\n❌ Se necesitan al menos 2 épocas para un torneo.")
        input("Presiona ENTER para salir...")
        return

    input("\nPresiona ENTER para continuar...")

    # Paso 2: Configurar parámetros del torneo
    clear_screen()
    print_header("PASO 2: CONFIGURACIÓN DEL TORNEO")

    print(f"Épocas seleccionadas: {len(epochs)}")
    print()

    n_matches = get_input("¿Cuántas partidas por enfrentamiento?", int, default=10,
                          validator=lambda x: x > 0)
    if n_matches is None:
        return

    temperature = get_input("¿Qué temperatura usar? (0.0 - 1.0)", float, default=0.5,
                            validator=lambda x: 0.0 <= x <= 1.0)
    if temperature is None:
        return

    # Paso 3: Configurar paralelismo
    clear_screen()
    print_header("PASO 3: CONFIGURACIÓN DE PARALELISMO")

    cpu_info = get_cpu_info()
    print(f"CPU detectada: {cpu_info['brand']}")
    print(f"Núcleos físicos: {cpu_info['physical_cores']}")
    print(f"Núcleos lógicos: {cpu_info['logical_cores']}")
    print()

    print("¿Cómo deseas configurar el paralelismo?")
    print()
    print("  [1] Automático (recomendado)")
    print("  [2] Todos los núcleos lógicos (máximo rendimiento)")
    print("  [3] Solo núcleos físicos (más estable)")
    print("  [4] Selección manual (especificar número)")
    print()

    parallel_mode = get_input("Selecciona una opción", int, default=1, validator=lambda x: 1 <= x <= 4)
    if parallel_mode is None:
        return

    n_workers = None
    physical_only = False

    if parallel_mode == 2:
        # Todos los núcleos lógicos
        n_workers = cpu_info['logical_cores']
        print(f"\n✅ Se usarán {n_workers} trabajadores (todos los núcleos lógicos)")
        print(f"   ⚠️  Esto usará el 100% de la CPU")
    elif parallel_mode == 3:
        # Solo núcleos físicos
        physical_only = True
        n_workers = cpu_info['physical_cores']
        print(f"\n✅ Se usarán {n_workers} trabajadores (solo núcleos físicos)")
        print(f"   ℹ️  Configuración más estable, deja recursos para otras tareas")
    elif parallel_mode == 4:
        # Selección manual
        n_workers = get_input(f"¿Cuántos trabajadores? (1-{cpu_info['logical_cores']})", int,
                              validator=lambda x: 1 <= x <= cpu_info['logical_cores'])
        if n_workers is None:
            return
        print(f"\n✅ Se usarán {n_workers} trabajadores (selección manual)")

        # Mostrar porcentaje de uso
        percent_logical = (n_workers / cpu_info['logical_cores']) * 100
        percent_physical = (n_workers / cpu_info['physical_cores']) * 100
        print(f"   📊 Uso de CPU: {percent_logical:.1f}% de núcleos lógicos")
        if percent_physical <= 100:
            print(f"   📊 Equivalente a {percent_physical:.1f}% de núcleos físicos")
    else:
        # Automático
        n_workers = None
        print(f"\n✅ Se usará configuración automática")
        print(f"   ℹ️  El sistema decidirá el mejor número de trabajadores")

    checkpoint_interval = get_input("\n¿Cada cuántos enfrentamientos guardar checkpoint?", int, default=50,
                                    validator=lambda x: x > 0)
    if checkpoint_interval is None:
        return

    # Paso 4: Resumen y confirmación
    clear_screen()
    print_header("PASO 4: RESUMEN Y CONFIRMACIÓN")

    print("Configuración del torneo:")
    print()
    print(f"  📊 Épocas:                    {len(epochs)}")
    print(f"  🎮 Partidas por enfrentamiento: {n_matches}")
    print(f"  🌡️  Temperatura:               {temperature}")
    print(f"  ⚙️  Trabajadores:              {n_workers if n_workers else 'Automático'}")
    print(f"  💾 Intervalo de checkpoint:   Cada {checkpoint_interval} enfrentamientos")
    print()

    # Estimación
    temp_manager = MassiveTournamentManager("temp", epochs, n_matches, temperature)
    estimates = temp_manager.estimate_resources()

    print("Estimación de recursos:")
    print()
    print(f"  🔢 Total de enfrentamientos:  {estimates['total_matches']:,}")
    print(f"  ⏱️  Tiempo estimado:           {estimates['estimated_time_hours']:.1f} horas")
    print(f"  🧠 Memoria estimada:          {estimates['estimated_memory_mb']:.1f} MB")
    print(f"  💽 Espacio en disco:          {estimates['estimated_disk_mb']:.1f} MB")
    print()

    confirm = get_input("¿Deseas iniciar el torneo? (s/n)", bool, default=True)
    if not confirm:
        print("\n❌ Torneo cancelado.")
        input("Presiona ENTER para salir...")
        return

    # Iniciar torneo
    clear_screen()
    print_header("INICIANDO TORNEO")
    print()
    print("El torneo está iniciando...")
    print("Se mostrará una interfaz interactiva con el progreso en tiempo real.")
    print()
    print("⚠️  Para detener el torneo de forma segura, presiona Ctrl+C")
    print()
    input("Presiona ENTER para comenzar...")

    try:
        result = run_massive_tournament(
            epochs=epochs,
            n_matches=n_matches,
            temperature=temperature,
            n_workers=n_workers,
            physical_only=physical_only,
            specific_cores=None,
            checkpoint_interval=checkpoint_interval
        )

        if result is not None:
            clear_screen()
            print_header("🎉 TORNEO COMPLETADO")
            print()
            print("Resultados finales:")
            print()
            print(result.head(10).to_string(index=False))
            print()
            input("Presiona ENTER para salir...")

    except Exception as e:
        clear_screen()
        print_header("❌ ERROR")
        print()
        print(f"Ocurrió un error durante el torneo: {e}")
        print()
        input("Presiona ENTER para salir...")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Torneo paralelo masivo interactivo para agentes de Quarto",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--interactive", action="store_true", default=True,
                        help="Modo interactivo con menú (default)")
    parser.add_argument("--epochs", type=int, nargs='+', help="Lista de épocas para el torneo")
    parser.add_argument("--all", action="store_true", help="Usar todas las épocas disponibles")
    parser.add_argument("--max", type=int, default=10000, help="Número máximo de agentes")
    parser.add_argument("--matches", type=int, default=10, help="Partidas por enfrentamiento")
    parser.add_argument("--temp", type=float, default=0.5, help="Temperatura")
    parser.add_argument("--workers", type=int, default=None, help="Número de trabajadores")

    args = parser.parse_args()

    try:
        # Si no hay argumentos, usar modo interactivo
        if len(sys.argv) == 1 or args.interactive:
            interactive_menu()
        else:
            # Modo línea de comandos
            if args.all:
                all_epochs = get_all_available_epochs()
                epochs = all_epochs[:args.max] if len(all_epochs) > args.max else all_epochs
                if not epochs:
                    logger.error("No se encontraron épocas disponibles")
                    return 1
            elif args.epochs:
                epochs = args.epochs
            else:
                parser.print_help()
                return 0

            result = run_massive_tournament(
                epochs=epochs,
                n_matches=args.matches,
                temperature=args.temp,
                n_workers=args.workers,
                checkpoint_interval=50
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