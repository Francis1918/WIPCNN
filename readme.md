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
├── bot/                    # Implementaciones de bots
│   ├── CNN_bot.py         # Bot basado en CNN
│   ├── random_bot.py      # Bot aleatorio de referencia
│   └── human.py           # Interfaz para jugador humano
├── models/                # Arquitecturas de modelos
│   ├── CNN1.py           # Modelo CNN principal
│   └── NN_abstract.py    # Clase abstracta base
├── QuartoRL/             # Funciones de aprendizaje por refuerzo
│   ├── RL_functions.py   # Funciones de generación de experiencia
│   └── contest.py        # Sistema de torneos contra rivales
├── utils/                # Utilidades
│   └── logger.py         # Sistema de logging con colores
├── tournament_parallel.py # Torneo paralelo optimizado
├── compare_agents.py     # Comparación directa entre agentes
└── requirements.txt      # Dependencias del proyecto
```

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/tuusuario/hierarchical-SAE.git
cd hierarchical-SAE
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. (Opcional) Actualizar dependencias a versiones más recientes:
```bash
python update_requirements.py
```

4. Configurar quartopy (dependencia externa):
   - Clona el proyecto quartopy en una ubicación accesible
   - Crea un archivo `.env` con la ruta: `QUARTOPY_PATH=/ruta/a/quartopy`

## Uso

### Actualizar dependencias

El proyecto incluye una herramienta para mantener las dependencias actualizadas:

```bash
# Solo verificar versiones (sin actualizar)
python update_requirements.py --check-only

# Actualizar todas las dependencias automáticamente
python update_requirements.py --force

# Modo interactivo (pregunta antes de actualizar)
python update_requirements.py
```

### Comparar dos agentes

```bash
# Comparación básica
python compare_agents.py 1 100

# Con más partidas
python compare_agents.py 1 100 --matches 50

# Con visualización
python compare_agents.py 1 100 --visualize

# Con temperatura ajustada
python compare_agents.py 1 100 --temp 0.1
```

### Ejecutar torneo paralelo

```bash
# Modo interactivo
python tournament_parallel.py

# Con parámetros específicos
python tournament_parallel.py --epochs 1 50 100 150 200 --matches 10 --visualize

# Usar todos los agentes disponibles
python tournament_parallel.py --all --workers 8

# Usar solo núcleos físicos
python tournament_parallel.py --physical-only
```

### Verificar configuración CUDA

```bash
python check_cuda.py
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