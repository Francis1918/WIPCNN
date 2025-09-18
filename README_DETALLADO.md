# Proyecto de Aprendizaje por Refuerzo para el Juego Quarto

## Descripción General

Este proyecto implementa un sistema de aprendizaje por refuerzo profundo (Deep Reinforcement Learning) para entrenar agentes inteligentes que jueguen al juego de mesa Quarto. El sistema utiliza redes neuronales convolucionales (CNN) y algoritmos de aprendizaje por refuerzo como Deep Q-Network (DQN) con diversas optimizaciones para mejorar el rendimiento del agente mediante autojuego (self-play).

## El Juego Quarto

Quarto es un juego de mesa abstracto para dos jugadores inventado por Blaise Müller. Se juega en un tablero de 4×4 con 16 piezas únicas. Cada pieza tiene cuatro características binarias:
- Alto/bajo
- Claro/oscuro
- Cuadrado/redondo
- Sólido/hueco

En cada turno, un jugador selecciona una pieza para que su oponente la coloque en el tablero. El objetivo es formar una línea de cuatro piezas que compartan al menos una característica común. Esta línea puede ser horizontal, vertical o diagonal.

## Estructura del Proyecto

```
hierarchical-SAE/
│
├── trainRL.py                   # Script principal para entrenar el agente de RL
├── requirements.txt             # Dependencias del proyecto
├── setup_dependencies.py        # Configuración inicial de dependencias
│
├── QuartoRL/                    # Módulo con funcionalidades de RL para Quarto
│   ├── __init__.py
│   ├── contest.py               # Implementación de torneos entre agentes
│   └── RL_functions.py          # Funciones de aprendizaje por refuerzo
│
├── models/                      # Arquitecturas de redes neuronales
│   ├── __init__.py
│   ├── CNN1.py                  # Implementación de la CNN para Quarto
│   ├── NN_abstract.py           # Clase abstracta para redes neuronales
│   ├── best_models/             # Almacena los mejores modelos entrenados
│   ├── checkpoints/             # Puntos de control durante el entrenamiento
│   └── weights/                 # Pesos de los modelos
│
├── bot/                         # Implementación de diferentes bots para Quarto
│   ├── __init__.py
│   ├── CNN_bot.py               # Bot basado en CNN
│   ├── human.py                 # Interfaz para jugador humano
│   ├── random_bot.py            # Bot que realiza movimientos aleatorios
│   └── readme.md                # Documentación de los bots
│
├── utils/                       # Utilidades generales
│   └── __init__.py
│
├── tests/                       # Pruebas del sistema
│   ├── test_architecture.ipynb
│   ├── test_engine_CNN.ipynb
│   ├── test_engine.ipynb
│   ├── test_Quarto.ipynb
│   ├── test_RL.ipynb
│   ├── test_RL2.ipynb
│   └── test_temperature.ipynb
│
├── analysis/                    # Análisis de resultados
│   └── view_results_2last_states.ipynb
│
├── tools/                       # Herramientas auxiliares
│   ├── format_matches.py
│   ├── question_deepseek.ipynb
│   └── view_training.py
│
└── partidas_guardadas/          # Registro de partidas jugadas durante el entrenamiento
    ├── ba_increasing_n_last_states/
    └── epoch_*/                 # Partidas organizadas por época
```

## Componentes Principales

### 1. Redes Neuronales (models/)

El directorio `models/` contiene las implementaciones de las redes neuronales utilizadas:

- **CNN1.py**: Implementa `QuartoCNN`, una red neuronal convolucional diseñada específicamente para el juego Quarto. Esta red:
  - Procesa el estado del tablero y la información de las piezas disponibles
  - Tiene una arquitectura dual que produce dos salidas:
    - Predicción de la posición para colocar una pieza en el tablero
    - Predicción de la pieza a seleccionar para el oponente

- **NN_abstract.py**: Define una clase abstracta para todas las redes neuronales del proyecto, garantizando una interfaz común.

### 2. Bots (bot/)

El directorio `bot/` contiene diferentes implementaciones de agentes:

- **CNN_bot.py**: Implementa `Quarto_bot`, un agente que utiliza la CNN entrenada para tomar decisiones. Incluye parámetros como temperatura para controlar la exploración/explotación.

- **random_bot.py**: Un agente que realiza movimientos aleatorios, útil como línea base para comparación.

- **human.py**: Interfaz para permitir que un jugador humano juegue contra los bots entrenados.

### 3. Funcionalidades de RL (QuartoRL/)

El directorio `QuartoRL/` contiene la implementación del aprendizaje por refuerzo:

- **RL_functions.py**: Implementa funciones como `gen_experience()` para generar experiencias de juego mediante autojuego, utilizando la política actual del agente.

- **contest.py**: Implementa `run_contest()` para evaluar el rendimiento del agente contra versiones anteriores o diferentes oponentes.

### 4. Script Principal (trainRL.py)

El archivo `trainRL.py` es el núcleo del proyecto, orquestando todo el proceso de entrenamiento:

- **Configuración de Hiperparámetros**: Define parámetros clave como tamaño de lote, número de épocas, tamaño del buffer de repetición, etc.

- **Inicialización de Redes y Optimizador**: Configura la red de política, red objetivo, optimizador y programador de tasa de aprendizaje.

- **Bucle de Entrenamiento**: Para cada época:
  1. Genera experiencia mediante autojuego
  2. Actualiza la red de política utilizando el buffer de repetición
  3. Actualiza periódicamente la red objetivo
  4. Evalúa el rendimiento contra versiones anteriores
  5. Guarda el modelo y visualiza resultados

- **Características Avanzadas**:
  - Replay Buffer para almacenar y muestrear experiencias
  - Target Network para estabilizar el entrenamiento
  - Temperatura ajustable para balance exploración/explotación
  - Aumento progresivo del número de estados considerados
  - Visualización de tasas de victoria contra versiones anteriores

## Optimizaciones Técnicas

1. **Gradiente Clipping**: Previene explosiones de gradiente durante el entrenamiento.

2. **Programación de Tasa de Aprendizaje**: Utiliza un programador de tipo coseno para ajustar la tasa de aprendizaje a lo largo del entrenamiento.

3. **Función de Pérdida de Huber**: Más robusta a valores atípicos que el error cuadrático medio.

4. **Actualización Suave de Red Objetivo**: Usa interpolación TAU para actualizar gradualmente la red objetivo.

5. **Filtrado de Acciones Inválidas**: Evita que el agente aprenda de experiencias con acciones inválidas.

6. **Normalización de Estado**: Procesa los estados del juego para facilitar el aprendizaje de la red.

## Entrenamiento y Evaluación

El proceso de entrenamiento se basa en:

1. **Autojuego (Self-Play)**: El agente juega contra sí mismo para generar experiencias.

2. **Evaluación Periódica**: Después de cada época, el agente se evalúa contra versiones anteriores para medir su mejora.

3. **Almacenamiento de Checkpoints**: Se guardan modelos después de cada época para su posterior evaluación.

4. **Visualización de Resultados**: Se generan gráficos de tasa de victoria para monitorear el progreso del entrenamiento.

## Características Innovadoras

1. **Escalado de Complejidad**: El sistema aumenta progresivamente el número de estados considerados (N_LAST_STATES) a medida que avanza el entrenamiento, permitiendo que el agente aprenda gradualmente a considerar más contexto.

2. **Ajuste de Temperatura**: La exploración se controla mediante un parámetro de temperatura, con valores altos para la exploración inicial y valores más bajos para la explotación durante la evaluación.

3. **Torneos Contra Versiones Anteriores**: El sistema evalúa el agente actual contra versiones anteriores, proporcionando una medida clara de mejora.

## Análisis y Herramientas

El proyecto incluye varias herramientas para análisis:

- **view_training.py**: Visualiza métricas de entrenamiento como tasa de victoria, pérdida, etc.

- **test_*.ipynb**: Cuadernos Jupyter para probar diferentes aspectos del sistema.

- **view_results_2last_states.ipynb**: Analiza resultados específicos de entrenamientos con 2 últimos estados.

## Uso del Sistema

Para entrenar un nuevo agente:

1. Instalar dependencias: `pip install -r requirements.txt`
2. Ejecutar el script de entrenamiento: `python trainRL.py`

Los parámetros de entrenamiento se pueden ajustar en el script `trainRL.py`, con dos modos disponibles:
- Modo de entrenamiento real (DEBUG_PARAMS = False)
- Modo de depuración con parámetros reducidos (DEBUG_PARAMS = True)

## Resultados

Los resultados del entrenamiento se almacenan en:

- **Modelos**: `models/checkpoints/` y `models/best_models/`
- **Partidas**: `partidas_guardadas/`
- **Datos de entrenamiento**: Archivos `.pkl` en el directorio principal

La visualización de resultados muestra cómo el agente mejora con el tiempo, aumentando su tasa de victoria contra versiones anteriores a medida que aprende mejores estrategias para jugar Quarto.

## Conclusiones

Este proyecto demuestra la aplicación de técnicas avanzadas de aprendizaje por refuerzo profundo al juego Quarto, logrando entrenar agentes que mejoran continuamente mediante autojuego. La arquitectura modular y las diversas optimizaciones implementadas permiten un entrenamiento eficiente y estable, resultando en agentes cada vez más competentes en el juego.
