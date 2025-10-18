# Comparador de Agentes para Quarto

Este proyecto proporciona herramientas para comparar agentes de diferentes épocas y arquitecturas en el juego Quarto. Permite cargar agentes desde una ruta externa específica, enfrentarlos en partidas y analizar su rendimiento relativo.

## Requisitos

- Python 3.6+
- PyTorch
- Matplotlib
- Pandas
- NumPy
- Quartopy (se configurará automáticamente si está disponible en las rutas predeterminadas)

## Estructura del Proyecto

- `compare_agents_external.py`: Script principal para comparar agentes
- `bot/`: Directorio con implementaciones de bots
- `models/`: Directorio con modelos de redes neuronales
- `utils/`: Utilidades del proyecto

## Uso

### Modo Interactivo

Para usar el comparador en modo interactivo, simplemente ejecute:

```bash
python compare_agents_external.py
```

El programa le guiará a través de los siguientes pasos:
1. Selección del directorio de agentes
2. Selección de los agentes a comparar
3. Configuración de parámetros (número de partidas, temperatura, etc.)
4. Ejecución de la comparación
5. Visualización y guardado de resultados

### Modo Línea de Comandos

También puede ejecutar el comparador desde la línea de comandos con argumentos específicos:

```bash
python compare_agents_external.py --agent1 "ruta/al/agente1.pt" --agent2 "ruta/al/agente2.pt" --matches 100
```

#### Argumentos Disponibles

- `--agent1`: Ruta al archivo del primer agente (obligatorio)
- `--agent2`: Ruta al archivo del segundo agente (obligatorio)
- `--agent1-type`: Tipo del primer agente (CNN o CNN_F, por defecto: CNN)
- `--agent2-type`: Tipo del segundo agente (CNN o CNN_F, por defecto: CNN)
- `--agent1-name`: Nombre para identificar al primer agente
- `--agent2-name`: Nombre para identificar al segundo agente
- `--matches`: Número de partidas a jugar por lado (por defecto: 500)
- `--temp`: Temperatura para ambos agentes (por defecto: 0.1)
- `--deterministic`: Usar modo determinista (por defecto: False)
- `--output`: Directorio para guardar resultados
- `--list`: Listar agentes disponibles
- `--agents-dir`: Directorio donde buscar los agentes

### Ejemplos

Listar agentes disponibles:
```bash
python compare_agents_external.py --list
```

Comparar dos agentes con configuración personalizada:
```bash
python compare_agents_external.py --agent1 "C:/Agentes/agente1.pt" --agent2 "C:/Agentes/agente2.pt" --agent1-type CNN --agent2-type CNN_F --matches 200 --temp 0.2 --output "C:/Resultados"
```

## Configuración con .env

El script utiliza un archivo `.env` para configurar rutas y otras opciones. Puede crear este archivo en el directorio raíz del proyecto con el siguiente contenido:

```
# Configuración de rutas para el proyecto
# Ruta principal al proyecto quartopy
QUARTOPY_PATH=C:/ruta/a/quartopy

# Rutas alternativas separadas por comas (sin espacios)
QUARTOPY_FALLBACK_PATHS=C:/ruta/alternativa1,C:/ruta/alternativa2

# Rutas para los agentes y resultados
AGENTS_PATH=C:/ruta/a/agentes
RESULTS_PATH=C:/ruta/a/resultados
```

## Rutas Predeterminadas

Si no se especifica en el archivo `.env`, se utilizarán estas rutas predeterminadas:

- Directorio de agentes: `C:/Users/bravo/Documents/Metodos Numericos Pycharm/Mech Interp/Comparacion entre agentes/Agentes`
- Directorio de resultados: `C:/Users/bravo/Documents/Metodos Numericos Pycharm/Mech Interp/Comparacion entre agentes/Resultados`

## Resultados

El script genera varios archivos de resultados:

1. **Gráfico de comparación**: Un gráfico circular que muestra la distribución de victorias y empates.
2. **CSV de resultados**: Un archivo CSV con estadísticas resumidas de todas las comparaciones realizadas.
3. **JSON detallado**: Un archivo JSON con información detallada de cada comparación.

## Notas Importantes

- El script ejecuta partidas en ambas direcciones (A vs B y B vs A) para equilibrar la ventaja de jugar primero.
- La temperatura controla la aleatoriedad en la toma de decisiones de los agentes (valores más bajos = más determinista).
- El modo determinista fuerza a los agentes a elegir siempre la mejor acción según su modelo.