"""
Cargador de agentes para el sistema de duelo.
"""
import sys
from pathlib import Path

# Agregar directorio padre al path para imports
_DUELO_DIR = Path(__file__).parent
if str(_DUELO_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_DUELO_DIR.parent))

import torch
import torch.nn as nn
from typing import List, Dict

# Imports condicionales para soportar ejecución directa y como módulo
try:
    from .adaptador_modelo import AdaptadorModelo
    from .utilidades import RUTA_AGENTES, listar_agentes
except ImportError:
    from adaptador_modelo import AdaptadorModelo
    from utilidades import RUTA_AGENTES, listar_agentes


class AgenteQuarto:
    """Representa un agente de Quarto cargado."""

    def __init__(self, nombre: str, modelo: nn.Module, metadata: dict, device: str):
        self.nombre = nombre
        self.modelo = modelo
        self.metadata = metadata
        self.device = torch.device(device)
        self.elo = 1500  # ELO inicial

    def seleccionar_movimiento(self, tablero, piezas_disponibles, pieza_actual=None) -> tuple:
        """
        Selecciona el mejor movimiento dado el estado actual.
        Retorna: (posicion, pieza_para_oponente)
        """
        self.modelo.eval()

        with torch.no_grad():
            # Preparar entrada del tablero
            x = self._preparar_estado(tablero)

            # Intentar llamar al modelo con pieza si lo soporta
            try:
                salida = self.modelo(x, pieza_actual)
            except TypeError:
                # El modelo no acepta pieza como argumento
                salida = self.modelo(x)

            # Interpretar salida según dimensiones
            posicion, pieza = self._interpretar_salida(salida, tablero, piezas_disponibles)

        return posicion, pieza

    def _preparar_estado(self, tablero) -> torch.Tensor:
        """Prepara el estado del juego como tensor."""
        if isinstance(tablero, torch.Tensor):
            x = tablero.clone().float()
        else:
            x = torch.tensor(tablero, dtype=torch.float32)

        # Normalizar dimensiones [batch, channels, 4, 4]
        if x.dim() == 1:
            x = x.view(4, 4)
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)

        return x.to(self.device)

    def _interpretar_salida(self, salida, tablero, piezas_disponibles) -> tuple:
        """Interpreta la salida del modelo."""
        if isinstance(salida, tuple) and len(salida) >= 2:
            # Modelo con salidas separadas para posición y pieza (QuartoCNN)
            pos_logits = salida[0].view(-1)
            pieza_logits = salida[1].view(-1)
        else:
            # Modelo con salida única
            if isinstance(salida, tuple):
                salida = salida[0]
            salida = salida.view(-1)

            if salida.shape[0] >= 32:
                # Asumimos 16 para posición + 16 para pieza
                pos_logits = salida[:16]
                pieza_logits = salida[16:32]
            else:
                # Solo posición
                pos_logits = salida[:16] if salida.shape[0] >= 16 else salida
                pieza_logits = torch.zeros(16, device=self.device)

        # Convertir tablero a tensor plano
        if isinstance(tablero, list):
            if isinstance(tablero[0], list):
                tablero_flat = [item for row in tablero for item in row]
            else:
                tablero_flat = tablero
        else:
            tablero_flat = tablero.view(-1).tolist()

        tablero_tensor = torch.tensor(tablero_flat, device=self.device)

        # Máscara de posiciones válidas (casillas vacías = -1)
        mascara_pos = (tablero_tensor == -1)

        # Aplicar máscara a posiciones
        pos_logits = pos_logits[:16].clone()
        pos_logits[~mascara_pos] = float('-inf')

        # Verificar que hay al menos una posición válida
        if mascara_pos.sum() == 0:
            posicion = 0  # Fallback
        else:
            posicion = pos_logits.argmax().item()

        # Seleccionar pieza para el oponente
        pieza_logits = pieza_logits[:16].clone()

        if piezas_disponibles:
            # Crear máscara para piezas disponibles
            mascara_pieza = torch.zeros(16, dtype=torch.bool, device=self.device)
            for p in piezas_disponibles:
                if 0 <= p < 16:
                    mascara_pieza[p] = True

            pieza_logits[~mascara_pieza] = float('-inf')
            pieza = pieza_logits.argmax().item()
        else:
            pieza = None

        return posicion, pieza

    def __repr__(self):
        return f"AgenteQuarto(nombre='{self.nombre}', elo={self.elo})"


class CargadorAgentes:
    """Carga y gestiona agentes desde la carpeta de agentes."""

    def __init__(self, device: str = None):
        self.adaptador = AdaptadorModelo(device)
        self.device = self.adaptador.device
        self.agentes_cargados: Dict[str, AgenteQuarto] = {}

    def cargar_todos(self) -> List[AgenteQuarto]:
        """Carga todos los agentes de la carpeta de agentes."""
        archivos = listar_agentes()

        if len(archivos) < 2:
            raise ValueError(
                f"Se necesitan al menos 2 agentes para un duelo. "
                f"Encontrados: {len(archivos)} en {RUTA_AGENTES}"
            )

        agentes = []
        for archivo in archivos:
            try:
                agente = self.cargar_agente(archivo)
                agentes.append(agente)
                print(f"✓ Agente cargado: {agente.nombre}")
            except Exception as e:
                print(f"✗ Error cargando {archivo.name}: {e}")

        return agentes

    def cargar_agente(self, ruta: Path) -> AgenteQuarto:
        """Carga un agente individual."""
        ruta = Path(ruta)

        if ruta.stem in self.agentes_cargados:
            return self.agentes_cargados[ruta.stem]

        modelo, metadata = self.adaptador.cargar_modelo(ruta)

        agente = AgenteQuarto(
            nombre=ruta.stem,
            modelo=modelo,
            metadata=metadata,
            device=str(self.device)
        )

        self.agentes_cargados[ruta.stem] = agente
        return agente

    def obtener_agentes_duelo(self) -> tuple:
        """Obtiene exactamente 2 agentes para el duelo."""
        agentes = self.cargar_todos()

        if len(agentes) < 2:
            raise ValueError("Se necesitan exactamente 2 agentes para el duelo")

        if len(agentes) > 2:
            print(f"Advertencia: Hay {len(agentes)} agentes, usando los primeros 2")

        return agentes[0], agentes[1]