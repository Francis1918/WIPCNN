cer# Mejoras de Entrenamiento RL - Guía Completa

## 📋 Resumen

Se han creado versiones mejoradas de los scripts de entrenamiento que resuelven el problema de rendimiento donde tu agente de época 1000 pierde contra agentes de épocas tempranas (9, 100, 300) con una diferencia del 20%.

## 📁 Archivos Creados

### Scripts de Entrenamiento

1. **[`trainRL_improved.py`](trainRL_improved.py)** - Entrenamiento desde cero con mejoras
   - Inicia un nuevo experimento llamado "improved_training"
   - Incluye todas las mejoras y visualización Bokeh
   - Usa QuartoCNN (no FlexibleCNN)

2. **[`trainRL_resume_improved.py`](trainRL_resume_improved.py)** ⭐ **NUEVO**
   - Continúa entrenamiento existente con mejoras
   - Compatible con tu experimento "ba_increasing_n_last_states"
   - Carga checkpoints y métricas previas automáticamente
   - Incluye visualización Bokeh + Matplotlib (compatibilidad)

### Documentación

3. **[`MEJORAS_ENTRENAMIENTO.md`](MEJORAS_ENTRENAMIENTO.md)** - Guía detallada de mejoras
4. **[`apply_improvements_to_existing.py`](apply_improvements_to_existing.py)** - Guía de modificaciones manuales
5. **[`test_improvements.py`](test_improvements.py)** - Script de verificación

## 🚀 Cómo Usar

### Opción 1: Continuar tu entrenamiento existente (RECOMENDADO)

```bash
python trainRL_resume_improved.py
```

Este script:
- ✅ Detecta automáticamente tu último checkpoint
- ✅ Carga el modelo de época 1000 (o la última disponible)
- ✅ Continúa desde ahí con las mejoras
- ✅ Mantiene compatibilidad con tus checkpoints existentes
- ✅ Genera visualizaciones Bokeh cada 10 épocas

### Opción 2: Iniciar entrenamiento nuevo

```bash
python trainRL_improved.py
```

Este script:
- Inicia desde época 0
- Crea un nuevo experimento "improved_training"
- Útil para comparar con tu entrenamiento anterior

## 🔧 Mejoras Implementadas

### 1. Replay Buffer Más Grande (Línea 57)
```python
# ANTES: 100 * STEPS_PER_EPOCH = 300,000 experiencias
# AHORA: 200 * STEPS_PER_EPOCH = 600,000 experiencias
REPLAY_SIZE = 200 * STEPS_PER_EPOCH
```
**Beneficio**: Mayor diversidad de experiencias, reduce overfitting

### 2. Exploración Mejorada (Línea 65)
```python
# ANTES: TEMPERATURE_EXPLORE = 0.5
# AHORA: TEMPERATURE_EXPLORE = 1.0
```
**Beneficio**: El agente explora más estrategias diferentes

### 3. Temperatura Dinámica (Línea 407)
```python
# Temperatura que decae gradualmente de 1.0 a 0.1
current_temp = TEMPERATURE_EXPLORE * (1 - e / EPOCHS) + TEMPERATURE_EXPLOIT * (e / EPOCHS)
```
**Beneficio**: Alta exploración al inicio, explotación al final

### 4. Weight Decay (Línea 100)
```python
WEIGHT_DECAY = 1e-5
# Aplicado en el optimizador (línea 437)
optimizer = optim.AdamW(..., weight_decay=WEIGHT_DECAY)
```
**Beneficio**: Previene overfitting penalizando pesos grandes

### 5. Learning Rate con Warmup (Líneas 103-104, 284-313)
```python
WARMUP_EPOCHS = 100
MIN_LR = 1e-6

class WarmupCosineScheduler:
    # Warmup: aumenta LR linealmente primeras 100 épocas
    # Cosine decay: reduce LR suavemente sin llegar a 0
```
**Beneficio**: Estabiliza entrenamiento inicial y evita LR = 0

### 6. Sistema de Métricas Completo (Líneas 110-199)
```python
class TrainingMetrics:
    # Rastrea: loss, Q-values, grad_norm, LR, win_rate, etc.
    # Guarda en pickle cada 10 épocas
    # Genera visualización Bokeh interactiva
```

### 7. Visualización Bokeh (Líneas 151-199)
Genera 6 gráficos interactivos:
- Training Loss
- Q-Values (promedio, máximo, mínimo)
- Learning Rate
- Win Rate
- Gradient Norm
- Exploration Temperature

## 📊 Monitoreo del Entrenamiento

### Archivos Generados

Cada 10 épocas se generan:
- `ba_increasing_n_last_states_metrics.pkl` - Datos de métricas
- `ba_increasing_n_last_states_training_metrics.html` - Visualización Bokeh

### Cómo Ver las Métricas

1. Abre el archivo HTML en tu navegador:
   ```bash
   start ba_increasing_n_last_states_training_metrics.html  # Windows
   open ba_increasing_n_last_states_training_metrics.html   # Mac
   xdg-open ba_increasing_n_last_states_training_metrics.html  # Linux
   ```

2. Verás gráficos interactivos donde puedes:
   - Hacer zoom
   - Ver valores exactos al pasar el mouse
   - Comparar diferentes métricas

### Señales de Buen Entrenamiento

✅ **Loss disminuye gradualmente**
- Debe bajar en las primeras 50-100 épocas
- Luego estabilizarse con pequeñas oscilaciones

✅ **Q-values aumentan y se estabilizan**
- Promedio debe aumentar inicialmente
- Luego mantenerse estable sin explotar

✅ **Win rate mejora contra rivales antiguos**
- Debe superar 50% contra épocas tempranas
- Idealmente llegar a 60-70% contra época 9, 100, 300

✅ **Gradient norm se mantiene estable**
- Debe estar entre 0.1 y 1.0
- Picos ocasionales son normales

### Señales de Problemas

❌ **Loss oscila violentamente**
- Solución: Reducir learning rate o aumentar warmup

❌ **Q-values explotan (>10) o colapsan (<-10)**
- Solución: Reducir learning rate, aumentar weight decay

❌ **Win rate disminuye con el tiempo**
- Solución: Aumentar temperatura de exploración

❌ **Gradient norm muy alto (>5.0)**
- Solución: Reducir learning rate

## 🎯 Comparación con Agentes Externos

Después de entrenar 100-200 épocas con las mejoras:

```bash
python compare_agents_external.py \
  --agent1 "models/weights/QuartoCNN1/ba_increasing_n_last_states_epoch_1100.pt" \
  --agent2 "ruta/al/agente_externo_epoca_300.pt" \
  --matches 500
```

Deberías ver mejora en el win rate de tu agente.

## 🔄 Diferencias entre Scripts

| Característica | trainRL_improved.py | trainRL_resume_improved.py |
|----------------|---------------------|----------------------------|
| Inicia desde | Época 0 | Última época guardada |
| Nombre experimento | "improved_training" | "ba_increasing_n_last_states" |
| Carga checkpoints | No | Sí, automáticamente |
| Carga métricas | No | Sí, automáticamente |
| Matplotlib | No | Sí (compatibilidad) |
| Bokeh | Sí | Sí |
| Uso recomendado | Nuevo experimento | Continuar existente |

## ⚙️ Ajustes Personalizados

### Si el entrenamiento es muy lento

Edita [`trainRL_resume_improved.py`](trainRL_resume_improved.py):

```python
# Línea 57: Reducir replay buffer
REPLAY_SIZE = 150 * STEPS_PER_EPOCH  # En lugar de 200

# Línea 52: Reducir matches por época
MATCHES_PER_EPOCH = 200  # En lugar de 300
```

### Si quieres más exploración

```python
# Línea 65: Aumentar temperatura
TEMPERATURE_EXPLORE = 1.5  # En lugar de 1.0

# Línea 103: Más épocas de warmup
WARMUP_EPOCHS = 200  # En lugar de 100
```

### Si quieres menos overfitting

```python
# Línea 100: Aumentar weight decay
WEIGHT_DECAY = 5e-5  # En lugar de 1e-5

# Línea 57: Aumentar replay buffer
REPLAY_SIZE = 300 * STEPS_PER_EPOCH  # En lugar de 200
```

## 📈 Tabla de Hiperparámetros

| Parámetro | Valor Original | Valor Mejorado | Impacto |
|-----------|---------------|----------------|---------|
| REPLAY_SIZE | 100x | 200x | ⬆️ Diversidad |
| TEMPERATURE_EXPLORE | 0.5 | 1.0 → 0.1 | ⬆️ Exploración |
| WEIGHT_DECAY | 0 | 1e-5 | ⬇️ Overfitting |
| LR Schedule | Cosine | Warmup + Cosine | ⬆️ Estabilidad |
| MIN_LR | 0.0 | 1e-6 | ⬆️ Aprendizaje continuo |
| Métricas | Solo win rate | 9 métricas | ⬆️ Diagnóstico |

## 🐛 Solución de Problemas

### Error: "No checkpoint found"
- **Causa**: No hay checkpoints previos
- **Solución**: El script iniciará desde época 0 automáticamente

### Error: "Cannot load metrics"
- **Causa**: Archivo de métricas corrupto o inexistente
- **Solución**: El script creará uno nuevo automáticamente

### Error: "Bokeh module not found"
- **Causa**: Bokeh no está instalado
- **Solución**: `pip install bokeh`

### Advertencia: "Gradient clipping activated"
- **Causa**: Gradientes muy grandes (normal ocasionalmente)
- **Solución**: Si ocurre frecuentemente, reducir learning rate

## 📝 Notas Importantes

1. **No uses FlexibleCNN/FlexibleBot** en el entrenamiento
   - Solo para comparación con agentes externos
   - El entrenamiento usa QuartoCNN

2. **Guarda métricas frecuentemente**
   - Se guardan automáticamente cada 10 épocas
   - Puedes cambiar la frecuencia en línea 583

3. **Compara regularmente**
   - Usa [`compare_agents_external.py`](compare_agents_external.py)
   - Compara contra épocas 9, 100, 300

4. **Paciencia**
   - Las mejoras pueden tardar 50-100 épocas en verse
   - Monitorea las visualizaciones Bokeh

## 🎓 Próximos Pasos

1. **Ejecutar entrenamiento mejorado**:
   ```bash
   python trainRL_resume_improved.py
   ```

2. **Monitorear cada 10 épocas**:
   - Abrir `ba_increasing_n_last_states_training_metrics.html`
   - Verificar que loss disminuye
   - Verificar que win rate mejora

3. **Comparar después de 100 épocas**:
   ```bash
   python compare_agents_external.py \
     --agent1 "models/weights/QuartoCNN1/ba_increasing_n_last_states_epoch_1100.pt" \
     --agent2 "ruta/agente_externo.pt" \
     --matches 500
   ```

4. **Ajustar si es necesario**:
   - Consultar sección "Ajustes Personalizados"
   - Modificar hiperparámetros según resultados

## 📞 Soporte

Si tienes problemas:
1. Revisa las visualizaciones Bokeh
2. Consulta la sección "Solución de Problemas"
3. Verifica los logs del entrenamiento
4. Ajusta hiperparámetros según la guía

## ✅ Checklist de Verificación

Antes de iniciar el entrenamiento:
- [ ] Bokeh instalado (`pip install bokeh`)
- [ ] Checkpoints existentes identificados
- [ ] Espacio en disco suficiente (>10GB)
- [ ] GPU disponible (opcional pero recomendado)

Durante el entrenamiento:
- [ ] Loss disminuye en primeras 50 épocas
- [ ] Q-values se mantienen estables
- [ ] Win rate mejora gradualmente
- [ ] Gradient norm < 5.0

Después de 100 épocas:
- [ ] Comparar con agentes externos
- [ ] Win rate > 50% contra épocas tempranas
- [ ] Ajustar hiperparámetros si es necesario

¡Buena suerte con el entrenamiento mejorado! 🚀