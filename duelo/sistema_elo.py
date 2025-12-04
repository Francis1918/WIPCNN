"""
Sistema de puntuación ELO para duelos.
"""
from typing import Tuple
import math


class SistemaELO:
    """Implementación del sistema de puntuación ELO."""

    def __init__(self, k_factor: float = 32.0, elo_inicial: float = 1500.0):
        """
        Args:
            k_factor: Factor K que determina la volatilidad de los cambios de ELO
            elo_inicial: Puntuación ELO inicial para nuevos jugadores
        """
        self.k_factor = k_factor
        self.elo_inicial = elo_inicial

    def probabilidad_esperada(self, elo_a: float, elo_b: float) -> float:
        """
        Calcula la probabilidad esperada de que A gane contra B.

        Args:
            elo_a: ELO del jugador A
            elo_b: ELO del jugador B

        Returns:
            Probabilidad esperada de victoria para A (0-1)
        """
        return 1.0 / (1.0 + math.pow(10, (elo_b - elo_a) / 400.0))

    def calcular_nuevo_elo(
        self,
        elo_a: float,
        elo_b: float,
        resultado_a: float
    ) -> Tuple[float, float]:
        """
        Calcula los nuevos ELOs después de una partida.

        Args:
            elo_a: ELO actual del jugador A
            elo_b: ELO actual del jugador B
            resultado_a: Resultado para A (1.0=victoria, 0.5=empate, 0.0=derrota)

        Returns:
            Tuple con (nuevo_elo_a, nuevo_elo_b)
        """
        esperado_a = self.probabilidad_esperada(elo_a, elo_b)
        esperado_b = 1.0 - esperado_a
        resultado_b = 1.0 - resultado_a

        nuevo_elo_a = elo_a + self.k_factor * (resultado_a - esperado_a)
        nuevo_elo_b = elo_b + self.k_factor * (resultado_b - esperado_b)

        return round(nuevo_elo_a, 2), round(nuevo_elo_b, 2)

    def actualizar_elos_serie(
        self,
        elo_a: float,
        elo_b: float,
        victorias_a: int,
        victorias_b: int,
        empates: int = 0
    ) -> Tuple[float, float]:
        """
        Actualiza ELOs después de una serie de partidas.

        Args:
            elo_a: ELO inicial del jugador A
            elo_b: ELO inicial del jugador B
            victorias_a: Número de victorias de A
            victorias_b: Número de victorias de B
            empates: Número de empates

        Returns:
            Tuple con (nuevo_elo_a, nuevo_elo_b)
        """
        # Procesar victorias de A
        for _ in range(victorias_a):
            elo_a, elo_b = self.calcular_nuevo_elo(elo_a, elo_b, 1.0)

        # Procesar victorias de B
        for _ in range(victorias_b):
            elo_a, elo_b = self.calcular_nuevo_elo(elo_a, elo_b, 0.0)

        # Procesar empates
        for _ in range(empates):
            elo_a, elo_b = self.calcular_nuevo_elo(elo_a, elo_b, 0.5)

        return elo_a, elo_b

    def resumen_estadisticas(
        self,
        nombre_a: str,
        nombre_b: str,
        elo_inicial_a: float,
        elo_inicial_b: float,
        elo_final_a: float,
        elo_final_b: float,
        victorias_a: int,
        victorias_b: int,
        empates: int
    ) -> dict:
        """Genera un resumen estadístico del duelo."""
        total_partidas = victorias_a + victorias_b + empates

        return {
            'jugadores': {
                nombre_a: {
                    'elo_inicial': elo_inicial_a,
                    'elo_final': elo_final_a,
                    'cambio_elo': round(elo_final_a - elo_inicial_a, 2),
                    'victorias': victorias_a,
                    'derrotas': victorias_b,
                    'empates': empates,
                    'porcentaje_victorias': round(victorias_a / total_partidas * 100, 2) if total_partidas > 0 else 0
                },
                nombre_b: {
                    'elo_inicial': elo_inicial_b,
                    'elo_final': elo_final_b,
                    'cambio_elo': round(elo_final_b - elo_inicial_b, 2),
                    'victorias': victorias_b,
                    'derrotas': victorias_a,
                    'empates': empates,
                    'porcentaje_victorias': round(victorias_b / total_partidas * 100, 2) if total_partidas > 0 else 0
                }
            },
            'total_partidas': total_partidas,
            'empates': empates,
            'ganador': nombre_a if elo_final_a > elo_final_b else (nombre_b if elo_final_b > elo_final_a else 'Empate')
        }