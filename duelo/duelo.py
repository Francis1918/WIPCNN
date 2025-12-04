"""
Sistema de duelo entre agentes de Quarto.
Carga agentes desde: C:/Users/bravo/Documents/Metodos Numericos Pycharm/Mech Interp/Comparacion entre agentes/Agentes
Guarda resultados en: C:/Users/bravo/Documents/Metodos Numericos Pycharm/Mech Interp/Comparacion entre agentes/Resultados
"""
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import random

# Agregar el directorio raíz del proyecto al path
ROOT_DIR = Path(__file__).parent.parent
DUELO_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(DUELO_DIR))

# Imports condicionales para soportar ejecución directa y como módulo
try:
    from .cargador_agentes import CargadorAgentes, AgenteQuarto
    from .sistema_elo import SistemaELO
    from .utilidades import (
        asegurar_directorios,
        guardar_resultado,
        RUTA_AGENTES,
        RUTA_RESULTADOS
    )
except ImportError:
    from cargador_agentes import CargadorAgentes, AgenteQuarto
    from sistema_elo import SistemaELO
    from utilidades import (
        asegurar_directorios,
        guardar_resultado,
        RUTA_AGENTES,
        RUTA_RESULTADOS
    )

# Importar el juego de Quarto del proyecto principal
try:
    from quartopy import Quarto
except ImportError:
    # Intentar importar desde ubicación alternativa
    try:
        from bot.CNN_bot import Quarto
    except ImportError:
        print("Advertencia: No se pudo importar Quarto, usando implementación local")
        Quarto = None


class MotorQuarto:
    """Motor simplificado de Quarto para duelos."""

    def __init__(self):
        self.reiniciar()

    def reiniciar(self):
        """Reinicia el estado del juego."""
        self.tablero = [-1] * 16  # -1 = vacío
        self.piezas_disponibles = list(range(16))
        self.pieza_actual = None
        self.turno = 0  # 0 o 1
        self.terminado = False
        self.ganador = None

    def seleccionar_pieza_inicial(self) -> int:
        """Selecciona una pieza aleatoria para comenzar."""
        self.pieza_actual = random.choice(self.piezas_disponibles)
        self.piezas_disponibles.remove(self.pieza_actual)
        return self.pieza_actual

    def hacer_movimiento(self, posicion: int, pieza_siguiente: Optional[int]) -> bool:
        """
        Ejecuta un movimiento.

        Args:
            posicion: Casilla donde colocar la pieza actual (0-15)
            pieza_siguiente: Pieza para el oponente (None si es último movimiento)

        Returns:
            True si el movimiento fue válido
        """
        # Validar posición
        if posicion < 0 or posicion >= 16:
            return False
        if self.tablero[posicion] != -1:
            return False

        # Colocar pieza
        self.tablero[posicion] = self.pieza_actual

        # Verificar victoria
        if self._verificar_victoria():
            self.terminado = True
            self.ganador = self.turno
            return True

        # Verificar empate
        if not self.piezas_disponibles and all(c != -1 for c in self.tablero):
            self.terminado = True
            self.ganador = -1  # Empate
            return True

        # Seleccionar pieza para oponente
        if pieza_siguiente is not None and pieza_siguiente in self.piezas_disponibles:
            self.piezas_disponibles.remove(pieza_siguiente)
            self.pieza_actual = pieza_siguiente
        elif self.piezas_disponibles:
            self.pieza_actual = self.piezas_disponibles[0]
            self.piezas_disponibles.remove(self.pieza_actual)
        else:
            self.pieza_actual = None

        # Cambiar turno
        self.turno = 1 - self.turno

        return True

    def _verificar_victoria(self) -> bool:
        """Verifica si hay una línea ganadora."""
        # Líneas a verificar: filas, columnas, diagonales
        lineas = []

        # Filas
        for i in range(4):
            lineas.append([i * 4 + j for j in range(4)])

        # Columnas
        for j in range(4):
            lineas.append([i * 4 + j for i in range(4)])

        # Diagonales
        lineas.append([0, 5, 10, 15])
        lineas.append([3, 6, 9, 12])

        for linea in lineas:
            piezas = [self.tablero[pos] for pos in linea]
            if -1 in piezas:
                continue

            # Verificar atributos comunes
            for bit in range(4):
                if all((p >> bit) & 1 == (piezas[0] >> bit) & 1 for p in piezas):
                    return True

        return False

    def obtener_estado(self) -> dict:
        """Retorna el estado actual del juego."""
        return {
            'tablero': self.tablero.copy(),
            'piezas_disponibles': self.piezas_disponibles.copy(),
            'pieza_actual': self.pieza_actual,
            'turno': self.turno,
            'terminado': self.terminado,
            'ganador': self.ganador
        }


def jugar_partida(agente_1: AgenteQuarto, agente_2: AgenteQuarto, motor: MotorQuarto = None) -> dict:
    """
    Juega una partida entre dos agentes.

    Returns:
        Diccionario con resultado de la partida
    """
    if motor is None:
        motor = MotorQuarto()
    else:
        motor.reiniciar()

    agentes = [agente_1, agente_2]

    # Pieza inicial aleatoria
    motor.seleccionar_pieza_inicial()

    movimientos = []

    while not motor.terminado:
        agente_actual = agentes[motor.turno]

        # Obtener movimiento del agente
        try:
            # Convertir tablero a formato 4x4 para el agente
            tablero_2d = [motor.tablero[i*4:(i+1)*4] for i in range(4)]

            posicion, pieza = agente_actual.seleccionar_movimiento(
                tablero_2d,
                motor.piezas_disponibles,
                motor.pieza_actual
            )

            movimientos.append({
                'agente': agente_actual.nombre,
                'posicion': posicion,
                'pieza_colocada': motor.pieza_actual,
                'pieza_siguiente': pieza
            })

            motor.hacer_movimiento(posicion, pieza)

        except Exception as e:
            print(f"Error en movimiento de {agente_actual.nombre}: {e}")
            # Movimiento aleatorio de emergencia
            posiciones_libres = [i for i in range(16) if motor.tablero[i] == -1]
            if posiciones_libres:
                pos = random.choice(posiciones_libres)
                pieza = motor.piezas_disponibles[0] if motor.piezas_disponibles else None
                motor.hacer_movimiento(pos, pieza)

    return {
        'ganador': agentes[motor.ganador].nombre if motor.ganador >= 0 else 'Empate',
        'ganador_idx': motor.ganador,
        'movimientos': len(movimientos),
        'historial': movimientos
    }


def ejecutar_duelo(num_partidas: int = 100, verbose: bool = True) -> dict:
    """
    Ejecuta un duelo completo entre los agentes cargados.

    Args:
        num_partidas: Número de partidas a jugar
        verbose: Si mostrar progreso

    Returns:
        Diccionario con resultados del duelo
    """
    asegurar_directorios()

    print("=" * 60)
    print("SISTEMA DE DUELO - QUARTO")
    print("=" * 60)
    print(f"Ruta de agentes: {RUTA_AGENTES}")
    print(f"Ruta de resultados: {RUTA_RESULTADOS}")
    print("=" * 60)

    # Cargar agentes
    cargador = CargadorAgentes()
    agente_1, agente_2 = cargador.obtener_agentes_duelo()

    print(f"\nAgentes del duelo:")
    print(f"  1. {agente_1.nombre}")
    print(f"  2. {agente_2.nombre}")
    print(f"\nPartidas a jugar: {num_partidas}")
    print("-" * 60)

    # Sistema ELO
    sistema_elo = SistemaELO()
    elo_inicial_1 = agente_1.elo
    elo_inicial_2 = agente_2.elo

    # Estadísticas
    victorias_1 = 0
    victorias_2 = 0
    empates = 0
    historial_partidas = []

    motor = MotorQuarto()

    for i in range(num_partidas):
        # Alternar quién empieza
        if i % 2 == 0:
            resultado = jugar_partida(agente_1, agente_2, motor)
        else:
            resultado = jugar_partida(agente_2, agente_1, motor)
            # Ajustar índices
            if resultado['ganador_idx'] == 0:
                resultado['ganador_idx'] = 1
            elif resultado['ganador_idx'] == 1:
                resultado['ganador_idx'] = 0

        # Actualizar estadísticas
        if resultado['ganador'] == agente_1.nombre:
            victorias_1 += 1
            agente_1.elo, agente_2.elo = sistema_elo.calcular_nuevo_elo(
                agente_1.elo, agente_2.elo, 1.0
            )
        elif resultado['ganador'] == agente_2.nombre:
            victorias_2 += 1
            agente_1.elo, agente_2.elo = sistema_elo.calcular_nuevo_elo(
                agente_1.elo, agente_2.elo, 0.0
            )
        else:
            empates += 1
            agente_1.elo, agente_2.elo = sistema_elo.calcular_nuevo_elo(
                agente_1.elo, agente_2.elo, 0.5
            )

        historial_partidas.append({
            'partida': i + 1,
            'ganador': resultado['ganador'],
            'movimientos': resultado['movimientos'],
            'elo_1': agente_1.elo,
            'elo_2': agente_2.elo
        })

        if verbose and (i + 1) % 10 == 0:
            print(f"Partida {i+1}/{num_partidas} | "
                  f"{agente_1.nombre}: {victorias_1} | "
                  f"{agente_2.nombre}: {victorias_2} | "
                  f"Empates: {empates}")

    # Generar resumen
    resumen = sistema_elo.resumen_estadisticas(
        agente_1.nombre, agente_2.nombre,
        elo_inicial_1, elo_inicial_2,
        agente_1.elo, agente_2.elo,
        victorias_1, victorias_2, empates
    )

    resultado_final = {
        'timestamp': datetime.now().isoformat(),
        'configuracion': {
            'num_partidas': num_partidas,
            'ruta_agentes': str(RUTA_AGENTES),
            'ruta_resultados': str(RUTA_RESULTADOS)
        },
        'agentes': {
            agente_1.nombre: agente_1.metadata,
            agente_2.nombre: agente_2.metadata
        },
        'resumen': resumen,
        'historial': historial_partidas
    }

    # Guardar resultados
    guardar_resultado(resultado_final)

    # Mostrar resumen final
    print("\n" + "=" * 60)
    print("RESULTADOS FINALES")
    print("=" * 60)
    print(f"\n{agente_1.nombre}:")
    print(f"  Victorias: {victorias_1} ({victorias_1/num_partidas*100:.1f}%)")
    print(f"  ELO: {elo_inicial_1} → {agente_1.elo} ({agente_1.elo - elo_inicial_1:+.2f})")

    print(f"\n{agente_2.nombre}:")
    print(f"  Victorias: {victorias_2} ({victorias_2/num_partidas*100:.1f}%)")
    print(f"  ELO: {elo_inicial_2} → {agente_2.elo} ({agente_2.elo - elo_inicial_2:+.2f})")

    print(f"\nEmpates: {empates} ({empates/num_partidas*100:.1f}%)")
    print(f"\n🏆 GANADOR: {resumen['ganador']}")
    print("=" * 60)

    return resultado_final


if __name__ == "__main__":
    # Ejecutar duelo con 100 partidas por defecto
    import argparse

    parser = argparse.ArgumentParser(description='Sistema de duelo entre agentes de Quarto')
    parser.add_argument('-n', '--partidas', type=int, default=100,
                        help='Número de partidas (default: 100)')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Modo silencioso')

    args = parser.parse_args()

    ejecutar_duelo(num_partidas=args.partidas, verbose=not args.quiet)