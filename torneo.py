"""
torneo.py - Sistema de torneos para agentes de Quarto
Permite torneos todos contra todos con diferentes modos de selección
"""

import os
import json
import csv
from datetime import datetime
from pathlib import Path
from itertools import combinations
from collections import defaultdict
from typing import Literal
import torch
from tqdm import tqdm

from models.CNN1 import QuartoCNN
from bot.CNN_bot import Quarto_bot
from quartopy import play_games


# =============================================================================
# SISTEMA DE PUNTUACIÓN ELO
# =============================================================================
class SistemaElo:
    """Sistema de puntuación Elo para torneos de Quarto."""

    def __init__(self, k_factor: float = 32, elo_inicial: float = 1500):
        """
        Args:
            k_factor: Factor K que determina cuánto cambia el rating por partida.
                      Mayor K = cambios más rápidos. Típico: 16-32
            elo_inicial: Rating inicial para nuevos jugadores
        """
        self.k_factor = k_factor
        self.elo_inicial = elo_inicial
        self.ratings = defaultdict(lambda: elo_inicial)
        self.historial = []

    def probabilidad_esperada(self, elo_a: float, elo_b: float) -> float:
        """Calcula probabilidad de que A gane contra B."""
        return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

    def actualizar(self, jugador_a: str, jugador_b: str, resultado_a: float):
        """
        Actualiza ratings después de una partida.

        Args:
            jugador_a: Nombre del jugador A
            jugador_b: Nombre del jugador B
            resultado_a: 1.0 = victoria A, 0.5 = empate, 0.0 = derrota A
        """
        elo_a = self.ratings[jugador_a]
        elo_b = self.ratings[jugador_b]

        esperado_a = self.probabilidad_esperada(elo_a, elo_b)
        esperado_b = 1 - esperado_a

        # Actualizar ratings
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
        # Victorias de jugador 1
        for _ in range(wins1):
            self.actualizar(nombre1, nombre2, 1.0)

        # Victorias de jugador 2
        for _ in range(wins2):
            self.actualizar(nombre1, nombre2, 0.0)

        # Empates
        for _ in range(empates):
            self.actualizar(nombre1, nombre2, 0.5)

    def ranking(self) -> list[tuple[str, float]]:
        """Retorna ranking ordenado por Elo."""
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)

    def get_elo(self, jugador: str) -> float:
        """Obtiene el Elo actual de un jugador."""
        return self.ratings[jugador]


# =============================================================================
# SISTEMA DE TORNEOS
# =============================================================================


class TorneoQuarto:
    """Sistema de torneos todos contra todos para agentes de Quarto."""

    def __init__(
            self,
            checkpoints_dir: str = r"C:\Users\bravo\Documents\Metodos Numericos Pycharm\Mech Interp\DatosEntrenamientoDev\checkpoints\QuartoCNN1",
            output_dir: str = r"C:\Users\bravo\Documents\Metodos Numericos Pycharm\Mech Interp\torneomasivo\torneo",
            partidas_por_enfrentamiento: int = 10,
            temperatura: float = 0.05,
            k_factor_elo: float = 32,
            elo_inicial: float = 1500
    ):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.partidas_por_enfrentamiento = partidas_por_enfrentamiento
        self.temperatura = temperatura
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Resultados tradicionales (puntos)
        self.resultados = defaultdict(lambda: {"wins": 0, "draws": 0, "losses": 0, "puntos": 0})
        self.matriz_enfrentamientos = {}

        # Sistema Elo
        self.elo = SistemaElo(k_factor=k_factor_elo, elo_inicial=elo_inicial)

    def listar_checkpoints(self) -> list[str]:
        """Lista todos los checkpoints disponibles."""
        # Buscar archivos .pt y .pth
        checkpoints = list(self.checkpoints_dir.glob("*.pt")) + list(self.checkpoints_dir.glob("*.pth"))
        checkpoints.sort(key=lambda x: x.stem)
        return [str(cp) for cp in checkpoints]

    def seleccionar_agentes(
            self,
            modo: Literal["todos", "grupo", "duo"],
            seleccion: list[str] = None
    ) -> list[str]:
        """
        Selecciona agentes según el modo especificado.

        Args:
            modo: "todos" - todos los checkpoints
                  "grupo" - lista específica de checkpoints
                  "duo" - exactamente 2 agentes para enfrentamiento directo
            seleccion: Lista de rutas de checkpoints (requerido para "grupo" y "duo")
        """
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

    def cargar_agente(self, checkpoint_path: str) -> Quarto_bot:
        """Carga un agente desde un checkpoint."""
        model = QuartoCNN()
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        model.eval()
        bot = Quarto_bot(model=model)
        bot.DETERMINISTIC = True
        bot.TEMPERATURE = self.temperatura
        return bot

    def enfrentar(self, checkpoint1: str, checkpoint2: str) -> dict:
        """Enfrenta dos agentes y retorna resultados."""
        bot1 = self.cargar_agente(checkpoint1)
        bot2 = self.cargar_agente(checkpoint2)

        nombre1 = Path(checkpoint1).stem
        nombre2 = Path(checkpoint2).stem

        partidas_dir = self.output_dir / "partidas" / f"{nombre1}_vs_{nombre2}"
        partidas_dir.mkdir(parents=True, exist_ok=True)

        # Jugar mitad como P1, mitad como P2
        n_mitad = self.partidas_por_enfrentamiento // 2

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
            "total_partidas": self.partidas_por_enfrentamiento
        }

    def ejecutar_torneo(
            self,
            modo: Literal["todos", "grupo", "duo"],
            seleccion: list[str] = None
    ) -> dict:
        """Ejecuta el torneo completo."""
        agentes = self.seleccionar_agentes(modo, seleccion)

        print(f"\n{'=' * 60}")
        print(f"🏆 TORNEO DE QUARTO - {self.timestamp}")
        print(f"{'=' * 60}")
        print(f"Agentes participantes: {len(agentes)}")
        print(f"Partidas por enfrentamiento: {self.partidas_por_enfrentamiento}")
        print(f"Total enfrentamientos: {len(list(combinations(agentes, 2)))}")
        print(f"{'=' * 60}\n")

        # Generar todos los enfrentamientos
        enfrentamientos = list(combinations(agentes, 2))

        for cp1, cp2 in tqdm(enfrentamientos, desc="Enfrentamientos"):
            resultado = self.enfrentar(cp1, cp2)

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
        # Obtener ranking Elo
        ranking_elo = self.elo.ranking()

        # 1. JSON completo (incluye Elo)
        json_path = self.output_dir / f"resultados_torneo_{self.timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": self.timestamp,
                "resultados": dict(self.resultados),
                "ranking_elo": {nombre: elo for nombre, elo in ranking_elo},
                "historial_elo": self.elo.historial,
                "enfrentamientos": self.matriz_enfrentamientos
            }, f, indent=2, ensure_ascii=False)

        # 2. Tabla de posiciones CSV (con Elo)
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

        # 3. Resumen de texto (con Elo)
        resumen_path = self.output_dir / f"resumen_torneo_{self.timestamp}.txt"
        with open(resumen_path, "w", encoding="utf-8") as f:
            f.write(f"{'=' * 70}\n")
            f.write(f"🏆 RESULTADOS DEL TORNEO - {self.timestamp}\n")
            f.write(f"{'=' * 70}\n\n")

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
    """Menú interactivo para ejecutar torneos."""
    CHECKPOINTS_DIR = r"C:\Users\bravo\Documents\Metodos Numericos Pycharm\Mech Interp\DatosEntrenamientoDev\checkpoints\QuartoCNN1"

    torneo = TorneoQuarto(checkpoints_dir=CHECKPOINTS_DIR)

    print("\n" + "=" * 60)
    print("🎮 SISTEMA DE TORNEOS QUARTO")
    print("=" * 60)
    print("\nModos disponibles:")
    print("1. Todos contra todos (todos los checkpoints)")
    print("2. Grupo específico (seleccionar varios)")
    print("3. Duelo (exactamente 2 agentes)")
    print("4. Listar checkpoints disponibles")
    print("0. Salir")

    opcion = input("\nSelecciona modo: ").strip()

    if opcion == "1":
        torneo.ejecutar_torneo(modo="todos")

    elif opcion == "2":
        checkpoints = torneo.listar_checkpoints()
        print("\nCheckpoints disponibles:")
        for i, cp in enumerate(checkpoints):
            print(f"  {i}: {Path(cp).stem}")

        indices = input("\nIngresa índices separados por coma (ej: 0,5,10,15): ").strip()
        seleccion = [checkpoints[int(i)] for i in indices.split(",")]
        torneo.ejecutar_torneo(modo="grupo", seleccion=seleccion)

    elif opcion == "3":
        checkpoints = torneo.listar_checkpoints()
        print("\nCheckpoints disponibles:")
        for i, cp in enumerate(checkpoints):
            print(f"  {i}: {Path(cp).stem}")

        idx1 = int(input("\nÍndice agente 1: ").strip())
        idx2 = int(input("Índice agente 2: ").strip())
        torneo.ejecutar_torneo(modo="duo", seleccion=[checkpoints[idx1], checkpoints[idx2]])

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
    menu_interactivo()
