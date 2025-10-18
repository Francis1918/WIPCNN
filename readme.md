# Hierarchical-SAE - Documentacion Consolidada

**Generado automaticamente:** 2025-10-17 22:29:01

**Nota:** Esta documentacion consolida todos los archivos README del proyecto.

---


# README Principal

*Fuente: readme.md*

# Hierarchical-SAE

Proyecto de aprendizaje por refuerzo para el juego Quarto utilizando redes neuronales convolucionales (CNN).

## Descripción

Este proyecto implementa un agente de IA para jugar Quarto usando aprendizaje por refuerzo con arquitectura CNN. Incluye herramientas para entrenamiento, evaluación y torneos entre diferentes versiones del agente.

## Características principales

- **Modelo CNN**: Arquitectura de red neuronal convolucional para toma de decisiones en Quarto
- **Entrenamiento RL**: Sistema de aprendizaje por refuerzo para mejorar el agente
- **Torneos paralelos**: Sistema de torneos que utiliza multiprocesamiento para evaluar múltiples agentes
- **Ranking Bradley-Terry**: Sistema avanzado de puntuación para evaluar habilidades relativas
- **Soporte GPU**: Optimizado para ejecutarse en CPU o GPU (CUDA)

## Estructura del proyecto

```
hierarchical-SAE/
├── trainRL.py                    # Script principal de entrenamiento
├── trainRL_resume.py             # Reanudar entrenamiento
├── requirements.txt              # Dependencias del proyecto
├── readme.md                     # Este archivo
│
├── 📁 install/                   # Sistema de instalación de dependencias
│   ├── install_requirements.py  # Script universal de instalación
│   ├── install.bat              # Helper para Windows
│   ├── install.sh               # Helper para Linux/macOS
│   └── README.md                # Documentación del sistema
│
├── 📁 scripts/                   # Scripts de utilidad y herramientas
│   ├── auto_checkpoint_monitor.py
│   ├── check_cuda.py
│   ├── cleanup_project.py
│   ├── diagnostico_proyecto.py
│   ├── organize_project.py
│   └── README.md
│
├── 📁 tournaments/               # Scripts de torneos y competencias
│   ├── compare_agents.py        # Comparación directa entre agentes
│   ├── tournament.py            # Torneo básico
│   ├── tournament_parallel.py   # Torneo paralelo optimizado
│   ├── tournament_parallel_CUDA.py
│   └── README.md
│
├── 📁 monitoring/                # Scripts de monitoreo
│   ├── epoch_group_monitor.py
│   └── README.md
│
├── 📁 docs/                      # Documentación del proyecto
│   ├── README_CONSOLIDATED.md
│   ├── README_DETALLADO.md
│   └── README.md
│
├── 📁 bot/                       # Implementaciones de bots
│   ├── CNN_bot.py               # Bot basado en CNN
│   ├── random_bot.py            # Bot aleatorio de referencia
│   ├── human.py                 # Interfaz para jugador humano
│   └── readme.md
│
├── 📁 models/                    # Arquitecturas de modelos
│   ├── CNN1.py                  # Modelo CNN principal
│   ├── NN_abstract.py           # Clase abstracta base
│   └── weights/                 # Pesos de modelos entrenados
│
├── 📁 QuartoRL/                  # Funciones de aprendizaje por refuerzo
│   ├── RL_functions.py          # Funciones de generación de experiencia
│   └── contest.py               # Sistema de torneos contra rivales
│
├── 📁 utils/                     # Utilidades generales
│   ├── logger.py                # Sistema de logging con colores
│   └── checkpoint_manager.py
│
├── 📁 checkpoint_monitor/        # Sistema de monitoreo de checkpoints
├── 📁 tests/                     # Tests y notebooks
├── 📁 tools/                     # Herramientas adicionales
└── 📁 analysis/                  # Análisis de resultados
```

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/tuusuario/hierarchical-SAE.git
cd hierarchical-SAE
```

2. Instalar dependencias usando el sistema universal:

**Windows:**
```cmd
python install/install_requirements.py --install
# O usar el helper interactivo
install\install.bat
```

**Linux/macOS:**
```bash
python install/install_requirements.py --install
# O usar el helper interactivo
./install/install.sh
```

**Opciones adicionales:**
```bash
# Crear entorno virtual e instalar
python install/install_requirements.py --create-venv --install

# Actualizar todas las dependencias
python install/install_requirements.py --upgrade-all

# Ver ayuda completa
python install/install_requirements.py --help
```

3. Configurar quartopy (dependencia externa):
   - Clona el proyecto quartopy en una ubicación accesible
   - Crea un archivo `.env` con la ruta: `QUARTOPY_PATH=/ruta/a/quartopy`

Para más información sobre el sistema de instalación, ver [`install/INSTALL_SYSTEM_README.md`](install/INSTALL_SYSTEM_README.md)

## Uso

### Entrenamiento

```bash
# Entrenar desde cero
python trainRL.py

# Reanudar entrenamiento
python trainRL_resume.py
```

```bash
# Comparación básica
python tournaments/compare_agents.py 1 100

# Con más partidas
python tournaments/compare_agents.py 1 100 --matches 50

# Con visualización
python tournaments/compare_agents.py 1 100 --visualize

# Con temperatura ajustada
python tournaments/compare_agents.py 1 100 --temp 0.1
```

### Ejecutar torneo paralelo

```bash
# Modo interactivo
python tournaments/tournament_parallel.py

# Con parámetros específicos
python tournaments/tournament_parallel.py --epochs 1 50 100 150 200 --matches 10 --visualize

# Usar todos los agentes disponibles
python tournaments/tournament_parallel.py --all --workers 8

# Usar solo núcleos físicos
python tournaments/tournament_parallel.py --physical-only
```

### Scripts de utilidad

```bash
# Verificar configuración CUDA
python scripts/check_cuda.py

# Diagnosticar proyecto
python scripts/diagnostico_proyecto.py

# Limpiar archivos innecesarios
python scripts/cleanup_project.py
```

### Monitoreo

```bash
# Monitorear checkpoints automáticamente
python scripts/auto_checkpoint_monitor.py

# Monitorear grupos de épocas
python monitoring/epoch_group_monitor.py
```

## Requisitos del sistema

- Python 3.8+
- PyTorch (con o sin CUDA)
- 8GB RAM mínimo (16GB recomendado para torneos grandes)
- GPU NVIDIA opcional (para aceleración)

## Licencia

Ver archivo LICENSE para más detalles.

## Autor

@z_tjona

## Agradecimientos

Este proyecto utiliza el juego Quarto implementado en el proyecto quartopy.

---


# Documentacion Detallada

*Fuente: README_DETALLADO.md*

# Proyecto de Aprendizaje por Refuerzo para el Juego Quarto
*Última actualización: 20 de septiembre de 2025*

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
├── auto_checkpoint_monitor.py   # Monitor automático de puntos de control
├── epoch_group_monitor.py       # Monitor de grupos de épocas
├── run_checkpoint_monitor.py    # Ejecutor de monitoreo de puntos de control
├── compare_agents.py            # Herramienta para comparar agentes de diferentes épocas
├── tournament.py                # Torneo "todos contra todos" entre agentes
├── tournament_parallel.py       # Versión paralela del torneo usando multiprocesamiento
├── test_collector.py            # Pruebas del sistema de recolección de datos
├── try_collector.py             # Experimentos con recolectores de TorchRL
├── debugging.py                 # Script de depuración y pruebas rápidas
├── actions.py                   # Definición de especificaciones de acciones para TorchRL
├── cart_p0ole.py               # Ejemplo/prueba con el entorno CartPole
├── board.csv                   # Datos del tablero de Quarto
├── piece_map.csv               # Mapeo de características de las piezas
├── .env                        # Variables de entorno
├── .gitignore                  # Archivos ignorados por Git
├── LICENSE                     # Licencia del proyecto
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
│   ├── best_models_auto/        # Mejores modelos seleccionados automáticamente
│   ├── best_models_by_group/    # Mejores modelos por grupo de épocas
│   ├── checkpoints/             # Puntos de control durante el entrenamiento
│   ├── checkpoints_monitored/   # Puntos de control bajo monitoreo
│   ├── evaluation_results/      # Resultados de evaluación de modelos
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
│   ├── __init__.py
│   ├── logger.py                # Sistema de registro personalizado
│   └── checkpoint_manager.py    # Gestión avanzada de checkpoints de modelos
│
├── checkpoint_monitor/          # Sistema de monitoreo de puntos de control
│   ├── __init__.py
│   ├── checkpoint_manager.py    # Gestión de puntos de control
│   ├── model_evaluator.py       # Evaluación de modelos
│   ├── monitor.py               # Monitor principal
│   ├── monitor.log              # Registro de monitoreo
│   ├── visualize.py             # Visualización de resultados
│   ├── logs/                    # Registros detallados
│   └── visualizations/          # Visualizaciones generadas
│
├── chat/                        # Componente de chat o interfaz conversacional
│   └── architecture.md          # Documentación de la arquitectura de chat
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
│   ├── view_results_2last_states.ipynb
│   └── agent_comparisons/       # Análisis comparativos entre agentes
│
├── tools/                       # Herramientas auxiliares
│   ├── format_matches.py
│   ├── question_deepseek.ipynb
│   └── view_training.py
│
└── partidas_guardadas/          # Registro de partidas jugadas durante el entrenamiento
    ├── ba_increasing_n_last_states/
    ├── compare_*/               # Resultados de comparaciones entre agentes
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

### 5. Sistema de Monitoreo de Checkpoints (checkpoint_monitor/)

El directorio `checkpoint_monitor/` implementa un sistema avanzado para monitorear, evaluar y gestionar los puntos de control del modelo:

- **checkpoint_manager.py**: Gestiona el ciclo de vida de los checkpoints, incluyendo su creación, selección y eliminación.

- **model_evaluator.py**: Proporciona funcionalidades para evaluar el rendimiento de los modelos guardados mediante métricas como tasa de victoria.

- **monitor.py**: Implementa el monitor principal que supervisa el proceso de entrenamiento y activa evaluaciones periódicas.

- **visualize.py**: Genera visualizaciones para analizar el rendimiento de los modelos a lo largo del tiempo.

### 6. Scripts de Monitoreo (archivos en la raíz)

Varios scripts en el directorio raíz permiten diferentes modos de monitoreo:

- **auto_checkpoint_monitor.py**: Implementa un monitoreo automático que selecciona los mejores checkpoints basado en criterios predefinidos.

- **epoch_group_monitor.py**: Monitorea y evalúa grupos de épocas para identificar tendencias en el rendimiento.

- **run_checkpoint_monitor.py**: Script para ejecutar el monitor de checkpoints de forma manual o programada.

### 7. Herramientas de Comparación y Torneos

El proyecto incluye herramientas avanzadas para evaluar y comparar agentes:

- **compare_agents.py**: Herramienta de línea de comandos para enfrentar agentes de diferentes épocas. Permite:
  - Comparar el rendimiento entre dos épocas específicas
  - Configurar número de partidas y parámetros de temperatura
  - Guardar partidas para análisis posterior
  - Generar visualizaciones de los resultados

- **tournament.py**: Implementa torneos "todos contra todos" entre múltiples agentes:
  - Modo interactivo para seleccionar épocas
  - Enfrentamientos exhaustivos entre todos los participantes
  - Generación de tablas de clasificación
  - Identificación del agente campeón

- **tournament_parallel.py**: Versión optimizada del torneo que utiliza multiprocesamiento:
  - Paralelización de enfrentamientos para mayor velocidad
  - Soporte para configuración de núcleos físicos vs lógicos
  - Optimización para sistemas con núcleos P y E (rendimiento y eficiencia)
  - Escalabilidad para torneos grandes

Para comparar agentes y realizar torneos:

1. **Comparar dos agentes específicos**:
   ```cmd
   python compare_agents.py 1 100                   # Enfrentar época 1 vs época 100
   python compare_agents.py 1 100 --matches 50      # Con 50 partidas
   python compare_agents.py 1 100 --visualize       # Guardar partidas y generar visualización
   python compare_agents.py 1 100 --temp 0.1        # Usar temperatura baja
   ```

2. **Realizar torneos entre múltiples agentes**:
   ```cmd
   python tournament.py                              # Modo interactivo
   python tournament.py --epochs 1 50 100 150 200   # Épocas específicas
   python tournament.py --all                        # Todas las épocas disponibles
   ```

3. **Torneos paralelos (más rápidos)**:
   ```cmd
   python tournament_parallel.py --all --workers 4  # 4 trabajadores
   python tournament_parallel.py --physical-only    # Solo núcleos físicos
   python tournament_parallel.py --p-cores-only     # Solo núcleos P (rendimiento)
   ```
### 8. Utilidades y Herramientas de Desarrollo

- **debugging.py**: Script de depuración para pruebas rápidas y diagnósticos del sistema.

- **test_collector.py**: Pruebas del sistema de recolección de datos usando TorchRL, validando la correcta integración con entornos de Gymnasium.

- **try_collector.py**: Experimentos con diferentes configuraciones de recolectores de datos para optimizar la generación de experiencias.

- **actions.py**: Define las especificaciones de acciones para TorchRL, estableciendo el espacio de acciones dual (selección de pieza y posición en tablero).

- **cart_p0ole.py**: Ejemplo de implementación con el entorno CartPole para validar el pipeline de entrenamiento.

- **utils/checkpoint_manager.py**: Clase `ModelCheckpointer` que proporciona gestión avanzada de checkpoints:
  - Mantenimiento automático de los mejores modelos
  - Limitación del número de checkpoints regulares
  - Gestión de directorios y metadatos

### 9. Archivos de Configuración y Datos

- **board.csv** y **piece_map.csv**: Archivos de datos que contienen información estructurada sobre el tablero de Quarto y las características de las piezas.

- **.env**: Variables de entorno para configuración del proyecto.

- **.gitignore**: Configuración para excluir archivos temporales y datos sensibles del control de versiones.

- **LICENSE**: Licencia del proyecto.

## Dependencias del Proyecto

El proyecto utiliza las siguientes bibliotecas principales:

- **PyTorch**: Framework principal para implementación y entrenamiento de redes neuronales
- **TorchRL**: Biblioteca específica para aprendizaje por refuerzo con PyTorch
- **TensorDict**: Manejo eficiente de tensores para experiencias de RL
- **Gymnasium**: Entornos estandarizados para aprendizaje por refuerzo
- **Matplotlib**: Visualización de resultados
- **Numpy & Pandas**: Manipulación de datos y análisis
- **TQDM**: Barras de progreso para monitoreo de entrenamiento
- **Colorama**: Salida en consola con colores para mejor legibilidad

Para instalar todas las dependencias: `pip install -r requirements.txt`

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

4. **Sistema de Monitoreo Automático**: El proyecto incluye un sistema de monitoreo de checkpoints que evalúa continuamente los modelos guardados, selecciona los mejores según diversos criterios y genera visualizaciones para seguir el progreso.

5. **Agrupación por Épocas**: La funcionalidad de monitoreo por grupos de épocas permite identificar tendencias y patrones en el aprendizaje a diferentes escalas temporales.

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

Para monitorear el entrenamiento y evaluar modelos:

1. Durante el entrenamiento: `python auto_checkpoint_monitor.py`
2. Para evaluar grupos de épocas: `python epoch_group_monitor.py`
3. Para una evaluación manual: `python run_checkpoint_monitor.py`

Para comparar agentes y realizar torneos:

1. **Comparar dos agentes específicos**:
   ```cmd
   python compare_agents.py 1 100                   # Enfrentar época 1 vs época 100
   python compare_agents.py 1 100 --matches 50      # Con 50 partidas
   python compare_agents.py 1 100 --visualize       # Guardar partidas y generar visualización
   python compare_agents.py 1 100 --temp 0.1        # Usar temperatura baja
   ```

2. **Realizar torneos entre múltiples agentes**:
   ```cmd
   python tournament.py                              # Modo interactivo
   python tournament.py --epochs 1 50 100 150 200   # Épocas específicas
   python tournament.py --all                        # Todas las épocas disponibles
   ```

3. **Torneos paralelos (más rápidos)**:
   ```cmd
   python tournament_parallel.py --all --workers 4  # 4 trabajadores
   python tournament_parallel.py --physical-only    # Solo núcleos físicos
   python tournament_parallel.py --p-cores-only     # Solo núcleos P (rendimiento)
   ```
## Resultados

Los resultados del entrenamiento se almacenan en:

- **Modelos**: `models/checkpoints/` y `models/best_models/`
- **Partidas**: `partidas_guardadas/`
- **Datos de entrenamiento**: Archivos `.pkl` en el directorio principal

La visualización de resultados muestra cómo el agente mejora con el tiempo, aumentando su tasa de victoria contra versiones anteriores a medida que aprende mejores estrategias para jugar Quarto.

## Estado Actual del Proyecto

El proyecto se encuentra en desarrollo activo, con mejoras continuas en:

1. **Arquitectura de Red**: Refinamiento de la CNN para capturar mejor las características del juego.
2. **Estrategias de Entrenamiento**: Ajuste de hiperparámetros y experimentación con diferentes enfoques de autojuego.
3. **Monitoreo y Evaluación**: Desarrollo de herramientas más sofisticadas para analizar el rendimiento de los modelos.
4. **Interfaz de Usuario**: Mejora de la interfaz para facilitar las partidas contra los bots entrenados.

## Contribuciones y Desarrollo Futuro

Se están considerando las siguientes mejoras para futuras versiones:

1. **Paralelización del Entrenamiento**: Implementar generación de experiencia en paralelo para acelerar el entrenamiento.
2. **Técnicas de Aprendizaje más Avanzadas**: Explorar algoritmos como PPO, A3C o SAC.
3. **Interpretabilidad**: Añadir herramientas para visualizar qué características del juego está aprendiendo la red.
4. **Interfaz Web**: Desarrollar una interfaz web para jugar contra los bots entrenados.

## Conclusiones

Este proyecto demuestra la aplicación de técnicas avanzadas de aprendizaje por refuerzo profundo al juego Quarto, logrando entrenar agentes que mejoran continuamente mediante autojuego. La arquitectura modular y las diversas optimizaciones implementadas permiten un entrenamiento eficiente y estable, resultando en agentes cada vez más competentes en el juego.


---
