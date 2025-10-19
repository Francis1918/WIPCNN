# Mejoras Implementadas en el Entrenamiento RL

## Problema Identificado

Tu agente de época 1000 está perdiendo contra agentes de épocas tempranas (9, 100, 300) con una diferencia del 20% (58.4% vs 39.8%). Esto indica problemas graves en el proceso de entrenamiento.

## Causas Principales

1. **Overfitting**: El agente se sobreajusta a experiencias recientes
2. **Exploración insuficiente**: Temperatura de exploración muy baja (0.5)
3. **Replay buffer pequeño**: Solo 100 épocas de experiencia
4. **Learning rate decay agresivo**: Reduce demasiado rápido el learning rate
5. **Falta de métricas**: No hay forma de detectar problemas durante el entrenamiento

## Mejoras Implementadas en `trainRL_improved.py`

### 1. **Replay Buffer Más Grande** (Línea 68)
```python
# ANTES:
REPLAY_SIZE = 100 * STEPS_PER_EPOCH  # 300,000 experiencias

# AHORA:
REPLAY_SIZE = 200 * STEPS_PER_EPOCH  # 600,000 experiencias
```
**Beneficio**: Mayor diversidad de experiencias, reduce overfitting

### 2. **Mayor Exploración** (Línea 75)
```python
# ANTES:
TEMPERATURE_EXPLORE = 0.5

# AHORA:
TEMPERATURE_EXPLORE = 1.0  # Duplicado
```
**Beneficio**: El agente explora más estrategias diferentes

### 3. **Temperatura Dinámica** (Línea 476)
```python
# NUEVO: Temperatura que decae gradualmente
current_temp = TEMPERATURE_EXPLORE * (1 - e / EPOCHS) + TEMPERATURE_EXPLOIT * (e / EPOCHS)
```
**Beneficio**: Alta exploración al inicio, explotación al final

### 4. **Regularización con Weight Decay** (Línea 117)
```python
# NUEVO:
WEIGHT_DECAY = 1e-5

# En el optimizador (línea 437):
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True, weight_decay=WEIGHT_DECAY)
```
**Beneficio**: Previene overfitting penalizando pesos grandes

### 5. **Learning Rate con Warmup** (Líneas 120-123, 383-410)
```python
# NUEVO: Scheduler personalizado
class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing"""
```
**Beneficio**: 
- Warmup: Estabiliza el entrenamiento inicial
- Cosine decay: Reduce LR suavemente sin llegar a cero

### 6. **Sistema de Métricas Completo** (Líneas 130-234)
```python
class TrainingMetrics:
    """Class to track and visualize training metrics"""
```
**Métricas rastreadas**:
- Loss por época
- Q-values (promedio, máximo, mínimo)
- Gradient norm
- Learning rate
- Win rate
- Reward promedio
- Temperatura de exploración
- Tamaño del replay buffer

### 7. **Visualización con Bokeh** (Líneas 169-234)
```python
def create_bokeh_visualization(self):
    """Create comprehensive Bokeh visualization"""
```
**Genera 6 gráficos interactivos**:
1. Training Loss
2. Q-Values (avg, max, min)
3. Learning Rate
4. Win Rate
5. Gradient Norm
6. Exploration Temperature

## Cómo Usar el Script Mejorado

### Entrenamiento desde cero:
```bash
python trainRL_improved.py
```

### Continuar entrenamiento existente:
El script detecta automáticamente checkpoints previos y continúa desde ahí.

### Visualizar métricas:
Las métricas se guardan cada 10 épocas en:
- `improved_training_metrics.pkl` (datos)
- `improved_training_metrics.html` (visualización Bokeh)

Abre el archivo HTML en tu navegador para ver gráficos interactivos.

## Modificaciones Sugeridas para Ajustar

### Si el entrenamiento es muy lento:
```python
# Línea 68: Reducir replay buffer
REPLAY_SIZE = 150 * STEPS_PER_EPOCH  # En lugar de 200

# Línea 46: Reducir matches por época
MATCHES_PER_EPOCH = 200  # En lugar de 300
```

### Si quieres más exploración:
```python
# Línea 75: Aumentar temperatura
TEMPERATURE_EXPLORE = 1.5  # En lugar de 1.0

# Línea 120: Más épocas de warmup
WARMUP_EPOCHS = 200  # En lugar de 100
```

### Si quieres menos overfitting:
```python
# Línea 117: Aumentar weight decay
WEIGHT_DECAY = 5e-5  # En lugar de 1e-5

# Línea 68: Aumentar replay buffer
REPLAY_SIZE = 300 * STEPS_PER_EPOCH  # En lugar de 200
```

## Aplicar Mejoras a trainRL.py y trainRL_resume.py

### Para trainRL.py:
Reemplaza las siguientes secciones:

1. **Línea 38**: Cambiar BATCH_SIZE y parámetros
2. **Línea 53**: Aumentar REPLAY_SIZE
3. **Línea 64**: Aumentar TEMPERATURE_EXPLORE
4. **Línea 141**: Agregar weight_decay al optimizador
5. **Línea 143**: Reemplazar scheduler con WarmupCosineScheduler
6. **Líneas 172-176**: Agregar temperatura dinámica
7. **Agregar**: Sistema de métricas y visualización Bokeh

### Para trainRL_resume.py:
Las mismas modificaciones que trainRL.py, pero manteniendo la lógica de carga de checkpoints.

## Monitoreo Durante el Entrenamiento

### Señales de buen entrenamiento:
- ✅ Loss disminuye gradualmente
- ✅ Q-values aumentan y se estabilizan
- ✅ Win rate mejora contra rivales antiguos
- ✅ Gradient norm se mantiene estable (< 1.0)

### Señales de problemas:
- ❌ Loss oscila violentamente
- ❌ Q-values explotan o colapsan
- ❌ Win rate disminuye con el tiempo
- ❌ Gradient norm muy alto (> 5.0)

## Comparación de Hiperparámetros

| Parámetro | Valor Original | Valor Mejorado | Razón |
|-----------|---------------|----------------|-------|
| REPLAY_SIZE | 100x | 200x | Más diversidad |
| TEMPERATURE_EXPLORE | 0.5 | 1.0 → 0.1 | Mejor exploración |
| WEIGHT_DECAY | 0 | 1e-5 | Prevenir overfitting |
| LR Schedule | Cosine | Warmup + Cosine | Estabilidad inicial |
| MIN_LR | 0.0 | 1e-6 | Evitar LR = 0 |

## Próximos Pasos

1. **Ejecutar entrenamiento mejorado**:
   ```bash
   python trainRL_improved.py
   ```

2. **Monitorear métricas cada 10 épocas**:
   - Abrir `improved_training_metrics.html`
   - Verificar que loss disminuye
   - Verificar que win rate mejora

3. **Comparar con agentes externos**:
   ```bash
   python compare_agents_external.py --agent1 "models/weights/QuartoCNN1/improved_training_epoch_1000.pt" --agent2 "ruta/agente_externo.pt" --matches 500
   ```

4. **Ajustar hiperparámetros** según resultados observados

## Archivos Generados

- `trainRL_improved.py`: Script de entrenamiento mejorado
- `improved_training_metrics.pkl`: Métricas guardadas
- `improved_training_metrics.html`: Visualización Bokeh
- `improved_training.pkl`: Resultados de contests
- `models/weights/QuartoCNN1/improved_training_epoch_XXXX.pt`: Checkpoints

## Notas Importantes

1. **No uses FlexibleCNN/FlexibleBot** en el entrenamiento - solo para comparación
2. **Guarda métricas frecuentemente** para detectar problemas temprano
3. **Compara regularmente** contra agentes de épocas tempranas
4. **Ajusta hiperparámetros** basándote en las visualizaciones Bokeh

## Soporte y Debugging

Si el entrenamiento no mejora después de 100 épocas:
1. Revisa las visualizaciones Bokeh
2. Verifica que loss está disminuyendo
3. Aumenta TEMPERATURE_EXPLORE a 1.5
4. Aumenta REPLAY_SIZE a 300x
5. Reduce WEIGHT_DECAY a 5e-6