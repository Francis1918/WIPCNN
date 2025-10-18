#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_compare_agents.py - Script para probar la funcionalidad del comparador de agentes.

Este script verifica que el comparador de agentes funcione correctamente, probando
las funciones principales y asegurando que pueda cargar agentes y ejecutar comparaciones.
"""

import os
import sys
from pathlib import Path

# Asegurar que podemos importar desde el directorio principal
sys.path.insert(0, str(Path(__file__).parent))

# Cargar variables de entorno desde .env
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)
        print(f"✅ Cargado archivo .env desde: {env_file}")
    except ImportError:
        # Si no tiene python-dotenv, instalar automáticamente
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        from dotenv import load_dotenv
        load_dotenv(env_file)
        print(f"✅ Instalado python-dotenv y cargado .env desde: {env_file}")
else:
    print(f"⚠️ No se encontró el archivo .env en: {env_file}")

# Importar funciones del comparador
from compare_agents_external import (
    load_agent_from_path,
    list_available_agents,
    compare_agents,
    DEFAULT_AGENTS_PATH,
    DEFAULT_RESULTS_PATH
)

def test_list_agents():
    """Prueba la función para listar agentes disponibles."""
    print("\n=== Probando listado de agentes ===")
    
    # Ruta de agentes desde .env o predeterminada
    agents_dir = DEFAULT_AGENTS_PATH
    
    # Verificar si el directorio existe
    if not os.path.exists(agents_dir):
        print(f"⚠️ El directorio de agentes no existe: {agents_dir}")
        print("Creando directorio para pruebas...")
        os.makedirs(agents_dir, exist_ok=True)
        print(f"✅ Directorio creado: {agents_dir}")
    
    # Listar agentes
    agents = list_available_agents(agents_dir)
    
    if agents:
        print(f"✅ Se encontraron {len(agents)} agentes en el directorio.")
        return True
    else:
        print("⚠️ No se encontraron agentes. Esto es normal si no hay archivos .pt en el directorio.")
        return False

def test_agent_loading():
    """Prueba la carga de agentes desde archivos."""
    print("\n=== Probando carga de agentes ===")
    
    # Ruta de agentes desde .env o predeterminada
    agents_dir = DEFAULT_AGENTS_PATH
    
    # Listar agentes disponibles
    agents = list_available_agents(agents_dir)
    
    if not agents:
        print("⚠️ No hay agentes disponibles para probar la carga.")
        print("Esta prueba se omitirá.")
        return None
    
    # Intentar cargar el primer agente
    try:
        agent_path = agents[0]
        print(f"Intentando cargar el agente: {agent_path}")
        
        # Primero intentar como CNN
        try:
            agent = load_agent_from_path(agent_path, agent_type="CNN", temperature=0.1)
            print(f"✅ Agente CNN cargado correctamente: {agent.name}")
            return True
        except Exception as e:
            print(f"⚠️ Error al cargar como CNN: {e}")
            
            # Intentar como CNN_F
            try:
                agent = load_agent_from_path(agent_path, agent_type="CNN_F", temperature=0.1)
                print(f"✅ Agente CNN_F cargado correctamente")
                return True
            except Exception as e:
                print(f"⚠️ Error al cargar como CNN_F: {e}")
                return False
    
    except Exception as e:
        print(f"❌ Error al cargar el agente: {e}")
        return False

def test_comparison():
    """Prueba la comparación entre dos agentes."""
    print("\n=== Probando comparación de agentes ===")
    
    # Ruta de agentes desde .env o predeterminada
    agents_dir = DEFAULT_AGENTS_PATH
    
    # Listar agentes disponibles
    agents = list_available_agents(agents_dir)
    
    if len(agents) < 2:
        print("⚠️ Se necesitan al menos dos agentes para probar la comparación.")
        print("Esta prueba se omitirá.")
        return None
    
    # Seleccionar dos agentes diferentes
    agent1_path = agents[0]
    agent2_path = agents[1] if len(agents) > 1 else agents[0]
    
    # Directorio de resultados desde .env o predeterminado
    results_dir = DEFAULT_RESULTS_PATH
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Comparando agentes:")
    print(f"Agente 1: {agent1_path}")
    print(f"Agente 2: {agent2_path}")
    print(f"Resultados en: {results_dir}")
    
    try:
        # Ejecutar una comparación con pocas partidas para prueba
        results = compare_agents(
            agent1_path=agent1_path,
            agent2_path=agent2_path,
            agent1_type="CNN",
            agent2_type="CNN",
            agent1_name="TestAgent1",
            agent2_name="TestAgent2",
            n_matches=5,  # Pocas partidas para prueba
            temperature=0.1,
            deterministic=False,
            output_dir=results_dir
        )
        
        if results:
            print("✅ Comparación completada con éxito.")
            print(f"Resultados: {results}")
            return True
        else:
            print("❌ La comparación no devolvió resultados.")
            return False
    
    except Exception as e:
        print(f"❌ Error durante la comparación: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    """Función principal para ejecutar todas las pruebas."""
    print("=== Iniciando pruebas del comparador de agentes ===")
    
    # Ejecutar pruebas
    list_result = test_list_agents()
    load_result = test_agent_loading()
    compare_result = test_comparison()
    
    # Resumen de resultados
    print("\n=== Resumen de pruebas ===")
    print(f"Listado de agentes: {'✅ Pasó' if list_result else '❌ Falló'}")
    print(f"Carga de agentes: {'✅ Pasó' if load_result else '❌ Falló' if load_result is not None else '⚠️ Omitida'}")
    print(f"Comparación de agentes: {'✅ Pasó' if compare_result else '❌ Falló' if compare_result is not None else '⚠️ Omitida'}")
    
    if list_result and (load_result is None or load_result) and (compare_result is None or compare_result):
        print("\n✅ Todas las pruebas completadas con éxito o omitidas correctamente.")
        print("El comparador de agentes está listo para usar.")
    else:
        print("\n⚠️ Algunas pruebas fallaron. Revise los mensajes de error para más detalles.")

if __name__ == "__main__":
    main()