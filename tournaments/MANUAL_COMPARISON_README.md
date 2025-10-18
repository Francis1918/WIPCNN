# 🤖 Manual Bot Comparison - Guía de Uso

Script mejorado para comparar bots de Quarto con parametrización completa y guardado automático de resultados.

## 🚀 Uso Básico

### Comparación Simple
```bash
python tournaments/manual_bot_comparison.py model1.pt model2.pt
```

### Con Nombres Descriptivos
```bash
python tournaments/manual_bot_comparison.py model1.pt model2.pt --names "Bot Bueno" "Bot Malo"
```

### Más Partidas
```bash
python tournaments/manual_bot_comparison.py model1.pt model2.pt --matches 1000
```

## 📋 Opciones Disponibles

### Modelos y Nombres
| Opción | Descripción |
|--------|-------------|
| `model_a` | Ruta al modelo del Bot A (requerido) |
| `model_b` | Ruta al modelo del Bot B (requerido) |
| `--names NAME_A NAME_B` | Nombres descriptivos para los bots |
| `--bot-a-type {cnn,cnn_f}` | Tipo de bot A (default: cnn) |
| `--bot-b-type {cnn,cnn_f}` | Tipo de bot B (default: cnn) |

### Configuración de Partidas
| Opción | Descripción |
|--------|-------------|
| `--matches N` | Número de partidas por posición (default: 500) |
| `--temp FLOAT` | Temperatura para exploración (default: 0.1) |
| `--deterministic` | Usar modo determinístico |
| `--verbose` | Mostrar detalles de cada partida |
| `--save-matches` | Guardar partidas individuales |
| `--no-mode-2x2` | No usar modo 2x2 |

### Guardado de Resultados
| Opción | Descripción |
|--------|-------------|
| `--save-format {json,csv,both}` | Formato de salida (default: json) |
| `--results-dir DIR` | Directorio para resultados (default: comparison_results) |

## 💡 Ejemplos Avanzados

### 1. Comparación Exhaustiva
```bash
python tournaments/manual_bot_comparison.py \
    CHECKPOINTS/EXP_id03/epoch_0377.pt \
    CHECKPOINTS/EXP_id03/epoch_0009.pt \
    --names "Modelo Avanzado" "Modelo Inicial" \
    --matches 1000 \
    --temp 0.05 \
    --save-format both
```

### 2. Comparación con Bot Francis (CNN_F)
```bash
python tournaments/manual_bot_comparison.py \
    model_standard.pt \
    model_francis.pt \
    --bot-a-type cnn \
    --bot-b-type cnn_f \
    --names "Standard" "Francis" \
    --matches 500
```

### 3. Modo Determinístico
```bash
python tournaments/manual_bot_comparison.py \
    model1.pt model2.pt \
    --deterministic \
    --matches 200
```

### 4. Guardar Partidas para Análisis
```bash
python tournaments/manual_bot_comparison.py \
    model1.pt model2.pt \
    --save-matches \
    --verbose \
    --matches 100
```

## 📊 Formato de Resultados

### JSON (default)
```json
{
  "timestamp": "20251018_155030",
  "bot_a_name": "Bot Bueno",
  "bot_b_name": "Bot Malo",
  "total_matches": 1000,
  "bot_a": {
    "total_wins": 650,
    "wins_as_p1": 320,
    "wins_as_p2": 330,
    "win_rate": 65.0,
    "win_rate_p1": 64.0,
    "win_rate_p2": 66.0
  },
  "bot_b": {
    "total_wins": 300,
    "win_rate": 30.0
  },
  "draws": {
    "total": 50,
    "rate": 5.0
  }
}
```

### CSV
```csv
Métrica,Bot A,Bot B
Nombre,Bot Bueno,Bot Malo
Total partidas,1000,
Victorias totales,650,300
Win rate general (%),65.00,30.00
...
```

## 📈 Interpretación de Resultados

### Estadísticas Clave

1. **Win Rate General**: Porcentaje de victorias sobre el total de partidas
2. **Win Rate P1/P2**: Rendimiento específico por posición
3. **Tasa de Empates**: Indica si los bots son muy similares

### Ejemplo de Salida
```
======================================================================
RESULTADOS DE LA COMPARACIÓN
======================================================================

Bot A: Bot Bueno
Bot B: Bot Malo
Total de partidas: 1000

----------------------------------------------------------------------
Estadística                    Bot A                Bot B               
----------------------------------------------------------------------
Victorias totales              650                  300                 
Win rate general               65.00%               30.00%
Victorias como P1              320                  150                 
Win rate como P1               64.00%               30.00%
Victorias como P2              330                  150                 
Win rate como P2               66.00%               30.00%
----------------------------------------------------------------------
Empates                        50                   
Tasa de empates                5.00%
======================================================================

🏆 Ganador: Bot Bueno
   Margen: 35.00%
```

## 🔧 Mejoras sobre el Script Original

### ✅ Nuevas Características

1. **Parametrización Completa**
   - No requiere editar código
   - Todos los parámetros configurables por CLI

2. **Guardado Automático**
   - Resultados en JSON y/o CSV
   - Timestamp automático
   - Organización en carpetas

3. **Manejo de Errores**
   - Validación de archivos
   - Mensajes claros de error
   - Imports opcionales (CNN_F_bot)

4. **Estadísticas Mejoradas**
   - Win rate por posición
   - Análisis de empates
   - Determinación automática de ganador

5. **Flexibilidad**
   - Soporte para múltiples tipos de bots
   - Configuración de temperatura
   - Modo determinístico opcional

## 📁 Estructura de Archivos

```
tournaments/
├── manual_bot_comparison.py       # Script mejorado
├── MANUAL_COMPARISON_README.md    # Esta guía
└── comparison_results/            # Resultados guardados
    ├── comparison_BotA_vs_BotB_20251018_155030.json
    └── comparison_BotA_vs_BotB_20251018_155030.csv
```

## 🔄 Migración desde Script Original

Si usabas el script original `script para comparar entre agentes`:

### Antes:
```python
# Editar código
bot_A = bot_good
bot_B = bot_Michael
N_MATCHES = 500
# Ejecutar
python "script para comparar entre agentes"
```

### Ahora:
```bash
python tournaments/manual_bot_comparison.py \
    CHECKPOINTS/EXP_id03/epoch_0009.pt \
    CHECKPOINTS/others/20250930_1010-EXP_id03_epoch_0017.pt \
    --names "Bot Good" "Bot Michael" \
    --matches 500 \
    --temp 0.1
```

## 🆘 Solución de Problemas

### Error: "quartopy no está instalado"
```bash
# Verificar que QUARTOPY_PATH esté configurado en .env
echo $QUARTOPY_PATH  # Linux/macOS
echo %QUARTOPY_PATH%  # Windows
```

### Error: "El archivo X no existe"
```bash
# Verificar ruta del modelo
ls CHECKPOINTS/EXP_id03/  # Linux/macOS
dir CHECKPOINTS\EXP_id03\  # Windows
```

### Advertencia: "CNN_F_bot no disponible"
- Normal si solo usas CNN_bot
- Para usar CNN_F_bot, asegúrate de tener `bot/CNN_F_bot.py`

## 📞 Ayuda

Ver todas las opciones:
```bash
python tournaments/manual_bot_comparison.py --help
```

## 🎯 Casos de Uso Comunes

### 1. Evaluar Mejora de Entrenamiento
```bash
# Comparar época inicial vs época final
python tournaments/manual_bot_comparison.py \
    models/checkpoints/epoch_0001.pt \
    models/checkpoints/epoch_0500.pt \
    --names "Inicial" "Final" \
    --matches 1000
```

### 2. Comparar Diferentes Experimentos
```bash
# Comparar dos experimentos diferentes
python tournaments/manual_bot_comparison.py \
    CHECKPOINTS/EXP_id03/best_model.pt \
    CHECKPOINTS/EXP_id05/best_model.pt \
    --names "Experimento 3" "Experimento 5" \
    --save-format both
```

### 3. Análisis Rápido
```bash
# Comparación rápida con pocas partidas
python tournaments/manual_bot_comparison.py \
    model1.pt model2.pt \
    --matches 100 \
    --temp 0.2
```

---

**Creado para facilitar la comparación de bots en el proyecto Hierarchical-SAE** 🤖