"""
Módulo de duelo entre agentes de diferentes arquitecturas CNN.
"""
from .cargador_agentes import CargadorAgentes
from .adaptador_modelo import AdaptadorModelo
from .sistema_elo import SistemaELO
from .duelo import ejecutar_duelo

__all__ = ['CargadorAgentes', 'AdaptadorModelo', 'SistemaELO', 'ejecutar_duelo']