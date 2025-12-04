"""
torneo_paralelo.py - Sistema de torneos PARALELO para agentes de Quarto
Permite torneos todos contra todos con multiproceso
"""

import os
import json
import csv
from datetime import datetime
from pathlib import Path
from itertools import combinations
from collections import defaultdict
from typing import Literal, Optional
import torch
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import psutil

from models.CNN1 import QuartoCNN
from bot.CNN_bot import Quarto_bot
from quartopy import play_games


# =============================================================================
# SISTEMA DE PUNTUACIÓN ELO
# =============================================================================
class SistemaElo:
    """Sistema de puntuación Elo para torneos de Quarto."""

    def __init__(self, k_factor: float = 32, elo_inicial: float = 1500):
        self.k_factor = k_factor
        self.elo_inicial = elo_inicial
        self.ratings = defaultdict(lambda: elo_inicial)
        self.historial = []

    def probabilidad_esperada(self, elo_a: float, elo_b: float) -> float:
        """Calcula probabilidad de que A gane contra B."""
        return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

    def actualizar(self, jugador_a: str, jugador_b: str, resultado_a: float):
        """Actualiza ratings después de una partida."""
        elo_a = self.ratings[jugador_a]
        elo_b = self.ratings[jugador_b]

        esperado_a = self.probabilidad_esperada(elo_a, elo_b)
        esperado_b = 1 - esperado_a

        nuevo_elo_a = elo_a + self.k_factor * (resultado_a - esperado_a)
        nuevo_elo_b = elo_b + self.k_factor * ((1 - resultado_a) - esperado_b)

        self.ratings[jugador_a] = nuevo_elo_a
        self.ratings[jugador_b] = nuevo_elo_b

        self.historial.append({
            "jugador_a": jugador_a,
            "jugador_b": jugador_b,
            "resultado": resultado_a,
            "elo_a_antes": elo_a,
            "elo_b_antes": elo_b,
            "elo_a_despues": nuevo_elo_a,
            "elo_b_despues": nuevo_elo_b
        })

    def procesar_enfrentamiento(self, nombre1: str, nombre2: str,
                                 wins1: int, wins2: int, empates: int):
        """Procesa todas las partidas de un enfrentamiento."""
        for _ in range(wins1):
            self.actualizar(nombre1, nombre2, 1.0)
        for _ in range(wins2):
            self.actualizar(nombre1, nombre2, 0.0)
        for _ in range(empates):
            self.actualizar(nombre1, nombre2, 0.5)

    def ranking(self) -> list[tuple[str, float]]:
        """Retorna ranking ordenado por Elo."""
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)

    def get_elo(self, jugador: str) -> float:
        """Obtiene el Elo actual de un jugador."""
        return self.ratings[jugador]


# =============================================================================
# FUNCIÓN PARA EJECUTAR UN ENFRENTAMIENTO (para multiprocessing)
# =============================================================================
def ejecutar_enfrentamiento(args: tuple) -> dict:
    """
    Ejecuta un enfrentamiento entre dos agentes.
    Esta función se ejecuta en un proceso separado.

    Args:
        args: Tupla con (checkpoint1, checkpoint2, partidas_por_enfrentamiento, temperatura, output_dir)

    Returns:
        dict con resultados del enfrentamiento
    """
    checkpoint1, checkpoint2, partidas_por_enfrentamiento, temperatura, output_dir = args

    # Cargar modelos
    model1 = QuartoCNN()
    model1.load_state_dict(torch.load(checkpoint1, weights_only=True))
    model1.eval()
    bot1 = Quarto_bot(model=model1)
    bot1.DETERMINISTIC = True
    bot1.TEMPERATURE = temperatura

    model2 = QuartoCNN()
    model2.load_state_dict(torch.load(checkpoint2, weights_only=True))
    model2.eval()
    bot2 = Quarto_bot(model=model2)
    bot2.DETERMINISTIC = True
    bot2.TEMPERATURE = temperatura

    nombre1 = Path(checkpoint1).stem
    nombre2 = Path(checkpoint2).stem

    partidas_dir = Path(output_dir) / "partidas" / f"{nombre1}_vs_{nombre2}"
    partidas_dir.mkdir(parents=True, exist_ok=True)

    # Jugar mitad como P1, mitad como P2
    n_mitad = partidas_por_enfrentamiento // 2

    # Bot1 como P1
    res1 = play_games(
        matches=n_mitad,
        player1=bot1,
        player2=bot2,
        verbose=False,
        match_dir=str(partidas_dir / "bot1_p1"),
        return_file_paths=False
    )

    # Bot1 como P2
    res2 = play_games(
        matches=n_mitad,
        player1=bot2,
        player2=bot1,
        verbose=False,
        match_dir=str(partidas_dir / "bot1_p2"),
        return_file_paths=False
    )

    # Calcular resultados para bot1
    wins1 = res1["P1"] + res2["P2"]
    wins2 = res1["P2"] + res2["P1"]
    draws = res1["Empates"] + res2["Empates"]

    return {
        "agente1": nombre1,
        "agente2": nombre2,
        "wins_agente1": wins1,
        "wins_agente2": wins2,
        "empates": draws,
        "total_partidas": partidas_por_enfrentamiento,
        "checkpoint1": checkpoint1,
        "checkpoint2": checkpoint2
    }


# =============================================================================
# SISTEMA DE TORNEOS PARALELO
# =============================================================================
class TorneoQuartoParalelo:
    """Sistema de torneos paralelos todos contra todos para agentes de Quarto."""

    def __init__(
            self,
            checkpoints_dir: str = r"C:\Users\bravo\Documents\Metodos Numericos Pycharm\Mech Interp\DatosEntrenamientoDev\checkpoints\QuartoCNN1",
            output_dir: str = r"C:\Users\bravo\Documents\Metodos Numericos Pycharm\Mech Interp\torneomasivo\TorneoParalelo",
            partidas_por_enfrentamiento: int = 10,
            temperatura: float = 0.05,
            k_factor_elo: float = 32,
            elo_inicial: float = 1500,
            num_workers: Optional[int] = None,
            usar_nucleos_fisicos: bool = True
    ):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.partidas_por_enfrentamiento = partidas_por_enfrentamiento
        self.temperatura = temperatura
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Configuración de workers
        self.num_workers = self._configurar_workers(num_workers, usar_nucleos_fisicos)

        # Resultados tradicionales (puntos)
        self.resultados = defaultdict(lambda: {"wins": 0, "draws": 0, "losses": 0, "puntos": 0})
        self.matriz_enfrentamientos = {}

        # Sistema Elo
        self.elo = SistemaElo(k_factor=k_factor_elo, elo_inicial=elo_inicial)

    def _configurar_workers(self, num_workers: Optional[int], usar_nucleos_fisicos: bool) -> int:
        """Configura el número de workers según las opciones."""
        nucleos_totales = cpu_count()
        nucleos_fisicos = psutil.cpu_count(logical=False) or (nucleos_totales // 2)

        if num_workers is not None:
            # Usuario especificó número exacto
            workers = min(num_workers, nucleos_totales)
        elif usar_nucleos_fisicos:
            # Usar solo núcleos físicos (evita hyperthreading)
            workers = nucleos_fisicos
        else:
            # Usar todos los núcleos (incluyendo lógicos)
            workers = nucleos_totales

        return max(1, workers)

    @staticmethod
    def obtener_info_cpu() -> dict:
        """Obtiene información sobre los CPUs disponibles."""
        nucleos_totales = cpu_count()
        nucleos_fisicos = psutil.cpu_count(logical=False) or (nucleos_totales // 2)
        return {
            "nucleos_totales": nucleos_totales,
            "nucleos_fisicos": nucleos_fisicos,
            "nucleos_logicos": nucleos_totales - nucleos_fisicos
        }

    def listar_checkpoints(self, recursivo: bool = True) -> list[str]:
        """Lista todos los checkpoints disponibles."""
        if not self.checkpoints_dir.exists():
            print(f"⚠️ ERROR: El directorio no existe: {self.checkpoints_dir}")
            return []

        pattern_pt = "**/*.pt" if recursivo else "*.pt"
        pattern_pth = "**/*.pth" if recursivo else "*.pth"

        checkpoints_pt = list(self.checkpoints_dir.glob(pattern_pt))
        checkpoints_pth = list(self.checkpoints_dir.glob(pattern_pth))
        checkpoints = checkpoints_pt + checkpoints_pth

        print(f"\n📁 Buscando checkpoints en: {self.checkpoints_dir}")
        print(f"   Búsqueda recursiva: {recursivo}")
        print(f"   Archivos .pt encontrados: {len(checkpoints_pt)}")
        print(f"   Archivos .pth encontrados: {len(checkpoints_pth)}")

        if len(checkpoints) == 0:
            all_files = list(self.checkpoints_dir.iterdir())
            print(f"   ⚠️ No se encontraron checkpoints!")
            print(f"   Contenido del directorio ({len(all_files)} items):")
            for f in all_files[:10]:
                print(f"      - {f.name} {'(carpeta)' if f.is_dir() else ''}")
            if len(all_files) > 10:
                print(f"      ... y {len(all_files) - 10} más")

        checkpoints.sort(key=lambda x: x.stem)
        return [str(cp) for cp in checkpoints]

    def seleccionar_agentes(
            self,
            modo: Literal["todos", "grupo", "duo"],
            seleccion: list[str] = None
    ) -> list[str]:
        """Selecciona agentes según el modo especificado."""
        if modo == "todos":
            return self.listar_checkpoints()
        elif modo in ["grupo", "duo"]:
            if seleccion is None:
                raise ValueError(f"Modo '{modo}' requiere lista de selección")
            if modo == "duo" and len(seleccion) != 2:
                raise ValueError("Modo 'duo' requiere exactamente 2 agentes")
            return seleccion
        else:
            raise ValueError(f"Modo desconocido: {modo}")

    def ejecutar_torneo(
            self,
            modo: Literal["todos", "grupo", "duo"],
            seleccion: list[str] = None
    ) -> dict:
        """Ejecuta el torneo completo en paralelo."""
        agentes = self.seleccionar_agentes(modo, seleccion)

        # Información del sistema
        info_cpu = self.obtener_info_cpu()

        print(f"\n{'=' * 70}")
        print(f"🏆 TORNEO DE QUARTO (PARALELO) - {self.timestamp}")
        print(f"{'=' * 70}")
        print(f"Agentes participantes: {len(agentes)}")
        print(f"Partidas por enfrentamiento: {self.partidas_por_enfrentamiento}")
        print(f"Total enfrentamientos: {len(list(combinations(agentes, 2)))}")
        print(f"{'=' * 70}")
        print(f"⚡ CONFIGURACIÓN PARALELA:")
        print(f"   Workers activos: {self.num_workers}")
        print(f"   Núcleos físicos: {info_cpu['nucleos_fisicos']}")
        print(f"   Núcleos totales: {info_cpu['nucleos_totales']}")
        print(f"{'=' * 70}\n")

        # Generar todos los enfrentamientos
        enfrentamientos = list(combinations(agentes, 2))

        # Preparar argumentos para multiprocessing
        args_list = [
            (cp1, cp2, self.partidas_por_enfrentamiento, self.temperatura, str(self.output_dir))
            for cp1, cp2 in enfrentamientos
        ]

        # Ejecutar en paralelo con barra de progreso
        resultados_enfrentamientos = []

        with Pool(processes=self.num_workers) as pool:
            for resultado in tqdm(
                pool.imap_unordered(ejecutar_enfrentamiento, args_list),
                total=len(args_list),
                desc=f"Enfrentamientos ({self.num_workers} workers)"
            ):
                resultados_enfrentamientos.append(resultado)

                # Procesar resultado inmediatamente
                nombre1 = resultado["agente1"]
                nombre2 = resultado["agente2"]

                # Actualizar estadísticas agente 1
                self.resultados[nombre1]["wins"] += resultado["wins_agente1"]
                self.resultados[nombre1]["draws"] += resultado["empates"]
                self.resultados[nombre1]["losses"] += resultado["wins_agente2"]
                self.resultados[nombre1]["puntos"] += resultado["wins_agente1"] * 3 + resultado["empates"]

                # Actualizar estadísticas agente 2
                self.resultados[nombre2]["wins"] += resultado["wins_agente2"]
                self.resultados[nombre2]["draws"] += resultado["empates"]
                self.resultados[nombre2]["losses"] += resultado["wins_agente1"]
                self.resultados[nombre2]["puntos"] += resultado["wins_agente2"] * 3 + resultado["empates"]

                # Actualizar sistema Elo
                self.elo.procesar_enfrentamiento(
                    nombre1, nombre2,
                    resultado["wins_agente1"],
                    resultado["wins_agente2"],
                    resultado["empates"]
                )

                # Guardar en matriz
                self.matriz_enfrentamientos[f"{nombre1}_vs_{nombre2}"] = resultado

        # Guardar resultados
        self._guardar_resultados()

        return dict(self.resultados)

    def _guardar_resultados(self):
        """Guarda todos los archivos de resultados."""
        ranking_elo = self.elo.ranking()

        # 1. JSON completo
        json_path = self.output_dir / f"resultados_torneo_{self.timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": self.timestamp,
                "num_workers": self.num_workers,
                "resultados": dict(self.resultados),
                "ranking_elo": {nombre: elo for nombre, elo in ranking_elo},
                "historial_elo": self.elo.historial,
                "enfrentamientos": self.matriz_enfrentamientos
            }, f, indent=2, ensure_ascii=False)

        # 2. Tabla de posiciones CSV
        csv_path = self.output_dir / f"tabla_posiciones_{self.timestamp}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Posicion", "Agente", "Elo", "Puntos", "Victorias", "Empates", "Derrotas", "WinRate"])
            for pos, (nombre, elo) in enumerate(ranking_elo, 1):
                stats = self.resultados[nombre]
                total = stats["wins"] + stats["draws"] + stats["losses"]
                wr = (stats["wins"] + 0.5 * stats["draws"]) / total if total > 0 else 0
                writer.writerow([
                    pos, nombre, f"{elo:.1f}", stats["puntos"], stats["wins"],
                    stats["draws"], stats["losses"], f"{wr:.2%}"
                ])

        # 3. Resumen de texto
        resumen_path = self.output_dir / f"resumen_torneo_{self.timestamp}.txt"
        with open(resumen_path, "w", encoding="utf-8") as f:
            f.write(f"{'=' * 70}\n")
            f.write(f"🏆 RESULTADOS DEL TORNEO (PARALELO) - {self.timestamp}\n")
            f.write(f"{'=' * 70}\n")
            f.write(f"Workers utilizados: {self.num_workers}\n\n")

            f.write("🥇🥈🥉 TOP 3 (por Elo):\n")
            for pos, (nombre, elo) in enumerate(ranking_elo[:3], 1):
                medalla = ["🥇", "🥈", "🥉"][pos - 1]
                stats = self.resultados[nombre]
                f.write(f"{medalla} {pos}. {nombre} - Elo: {elo:.1f} | {stats['puntos']} pts\n")

            f.write(f"\n{'=' * 70}\n")
            f.write("RANKING COMPLETO (ordenado por Elo):\n")
            f.write(f"{'Pos':<4} {'Agente':<45} {'Elo':<8} {'Pts':<5} {'V':<4} {'E':<4} {'D':<4} {'WR':<7}\n")
            f.write(f"{'-' * 70}\n")
            for pos, (nombre, elo) in enumerate(ranking_elo, 1):
                stats = self.resultados[nombre]
                total = stats["wins"] + stats["draws"] + stats["losses"]
                wr = (stats["wins"] + 0.5 * stats["draws"]) / total if total > 0 else 0
                f.write(f"{pos:<4} {nombre:<45} {elo:<8.1f} {stats['puntos']:<5} "
                       f"{stats['wins']:<4} {stats['draws']:<4} {stats['losses']:<4} {wr:<7.2%}\n")

        # 4. CSV solo de Elo
        elo_csv_path = self.output_dir / f"ranking_elo_{self.timestamp}.csv"
        with open(elo_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Posicion", "Agente", "Elo"])
            for pos, (nombre, elo) in enumerate(ranking_elo, 1):
                writer.writerow([pos, nombre, f"{elo:.1f}"])

        print(f"\n✅ Resultados guardados en: {self.output_dir}")
        print(f"   - {json_path.name}")
        print(f"   - {csv_path.name}")
        print(f"   - {resumen_path.name}")
        print(f"   - {elo_csv_path.name}")

        # Mostrar top 5 en consola
        print(f"\n🏆 TOP 5 (por Elo):")
        for pos, (nombre, elo) in enumerate(ranking_elo[:5], 1):
            stats = self.resultados[nombre]
            print(f"   {pos}. {nombre}: Elo {elo:.1f} | {stats['puntos']} pts")


def menu_interactivo():
    """Menú interactivo para ejecutar torneos paralelos."""
    CHECKPOINTS_DIR = r"C:\Users\bravo\Documents\Metodos Numericos Pycharm\Mech Interp\DatosEntrenamientoDev\checkpoints\QuartoCNN1"

    # Obtener info de CPU
    info_cpu = TorneoQuartoParalelo.obtener_info_cpu()

    print("\n" + "=" * 70)
    print("🎮 SISTEMA DE TORNEOS QUARTO (PARALELO)")
    print("=" * 70)
    print(f"\n💻 Información del sistema:")
    print(f"   Núcleos físicos: {info_cpu['nucleos_fisicos']}")
    print(f"   Núcleos totales (lógicos): {info_cpu['nucleos_totales']}")

    print("\n⚡ Configuración de paralelismo:")
    print("1. Usar solo núcleos físicos (recomendado)")
    print("2. Usar todos los núcleos (físicos + lógicos)")
    print("3. Especificar número de núcleos manualmente")
    print("0. Salir")

    opcion_nucleos = input("\nSelecciona configuración de núcleos: ").strip()

    if opcion_nucleos == "0":
        print("Saliendo...")
        return

    # Configurar workers
    num_workers = None
    usar_nucleos_fisicos = True

    if opcion_nucleos == "1":
        usar_nucleos_fisicos = True
        print(f"   ✓ Usando {info_cpu['nucleos_fisicos']} núcleos físicos")
    elif opcion_nucleos == "2":
        usar_nucleos_fisicos = False
        print(f"   ✓ Usando {info_cpu['nucleos_totales']} núcleos totales")
    elif opcion_nucleos == "3":
        num_workers = int(input(f"   Número de núcleos (1-{info_cpu['nucleos_totales']}): ").strip())
        num_workers = max(1, min(num_workers, info_cpu['nucleos_totales']))
        print(f"   ✓ Usando {num_workers} núcleos")
    else:
        print("Opción no válida")
        return

    # Crear torneo con configuración
    torneo = TorneoQuartoParalelo(
        checkpoints_dir=CHECKPOINTS_DIR,
        num_workers=num_workers,
        usar_nucleos_fisicos=usar_nucleos_fisicos
    )

    print("\n" + "-" * 70)
    print("Modos de torneo disponibles:")
    print("1. Todos contra todos (todos los checkpoints)")
    print("2. Grupo específico (seleccionar varios)")
    print("3. Duelo (exactamente 2 agentes)")
    print("4. Listar checkpoints disponibles")
    print("0. Salir")

    opcion = input("\nSelecciona modo de torneo: ").strip()

    if opcion == "1":
        torneo.ejecutar_torneo(modo="todos")

    elif opcion == "2":
        checkpoints = torneo.listar_checkpoints()
        if not checkpoints:
            print("No hay checkpoints disponibles.")
            return
        print("\nCheckpoints disponibles:")
        for i, cp in enumerate(checkpoints):
            print(f"  {i}: {Path(cp).stem}")

        indices = input("\nIngresa índices separados por coma (ej: 0,5,10,15): ").strip()
        try:
            seleccion = [checkpoints[int(i.strip())] for i in indices.split(",")]
            torneo.ejecutar_torneo(modo="grupo", seleccion=seleccion)
        except (ValueError, IndexError) as e:
            print(f"Error en la selección: {e}")

    elif opcion == "3":
        checkpoints = torneo.listar_checkpoints()
        if not checkpoints:
            print("No hay checkpoints disponibles.")
            return
        print("\nCheckpoints disponibles:")
        for i, cp in enumerate(checkpoints):
            print(f"  {i}: {Path(cp).stem}")

        try:
            idx1 = int(input("\nÍndice agente 1: ").strip())
            idx2 = int(input("Índice agente 2: ").strip())
            torneo.ejecutar_torneo(modo="duo", seleccion=[checkpoints[idx1], checkpoints[idx2]])
        except (ValueError, IndexError) as e:
            print(f"Error en la selección: {e}")

    elif opcion == "4":
        checkpoints = torneo.listar_checkpoints()
        print(f"\n{len(checkpoints)} checkpoints encontrados:")
        for cp in checkpoints:
            print(f"  - {Path(cp).stem}")

    elif opcion == "0":
        print("Saliendo...")
    else:
        print("Opción no válida")


if __name__ == "__main__":
    # Necesario para Windows con multiprocessing
    mp.freeze_support()
    menu_interactivo()

