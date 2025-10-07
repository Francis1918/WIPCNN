# 🔧 SOLUCIÓN: Error CUDA después de 2+ horas de ejecución

## 🎯 Problema Identificado

El error `RuntimeError: CUDA error: unspecified launch failure` después de 2+ horas se debe a:

1. **Acumulación de memoria GPU**: Los modelos no se liberaban completamente entre enfrentamientos
2. **Fragmentación de memoria VRAM**: Múltiples cargas/descargas sin sincronización
3. **Falta de limpieza garantizada**: Sin bloque `finally`, los errores impedían la limpieza
4. **Carga directa a GPU**: `torch.load(map_location=device)` puede causar problemas de memoria

## ✅ Soluciones Implementadas

### 1. Carga de Modelos Optimizada
```python
# ❌ ANTES (problemático):
state_dict = torch.load(model_path, map_location=actual_device)

# ✅ AHORA (seguro):
state_dict = torch.load(model_path, map_location='cpu')  # Cargar en CPU
model.load_state_dict(state_dict)
model.to(actual_device)  # Luego mover a GPU
del state_dict  # Liberar inmediatamente
```

### 2. Limpieza Preventiva antes de Cargar
```python
# Antes de cargar cada modelo:
torch.cuda.empty_cache()
torch.cuda.synchronize(actual_device)
```

### 3. Bloque `finally` Garantizado
```python
finally:
    # SIEMPRE se ejecuta, incluso con errores
    if agent1 is not None:
        if hasattr(agent1, 'model'):
            del agent1.model
        del agent1
    
    if agent2 is not None:
        if hasattr(agent2, 'model'):
            del agent2.model
        del agent2
    
    import gc
    gc.collect()
    
    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
```

### 4. Sincronización GPU Agresiva
- Se añadió `torch.cuda.synchronize()` después de cada operación crítica
- Esto asegura que todas las operaciones GPU terminen antes de continuar

## 🚀 Recomendaciones Adicionales

### Para Ejecuciones Muy Largas (6+ horas):

#### 1. Usar CUDA_LAUNCH_BLOCKING para debugging (solo si persiste el error)
```cmd
set CUDA_LAUNCH_BLOCKING=1
python tournament_parallel_CUDA.py --epochs 1 50 100 150
```

**⚠️ IMPORTANTE**: Esto hará el código más lento, úsalo solo para diagnóstico.

#### 2. Reducir Número de Workers Paralelos
Si tienes 16 workers, reduce a 8 o 4:
```cmd
python tournament_parallel_CUDA.py --epochs ... --workers 4
```

**Razón**: Menos procesos = menos presión sobre la GPU

#### 3. Monitorear Uso de GPU durante ejecución
Abre otra terminal y ejecuta:
```cmd
nvidia-smi -l 1
```

Esto te mostrará en tiempo real:
- Uso de memoria GPU
- Temperatura
- Procesos activos

#### 4. Checkpoints Automáticos
El código ahora guarda checkpoints cada 1000 enfrentamientos:
```
tournaments_parallel_CUDA/tournament_XXXXXXXX/results/partial_results_1000.csv
tournaments_parallel_CUDA/tournament_XXXXXXXX/results/partial_results_2000.csv
...
```

Si el torneo falla, puedes recuperar los resultados parciales.

#### 5. Limitar Temperatura de GPU
Si la GPU se sobrecalienta (>80°C), puede causar errores. Opciones:

**a) Limitar potencia de la GPU:**
```cmd
nvidia-smi -pl 200
```
(Ajusta 200 al valor apropiado para tu GPU)

**b) Mejorar ventilación del sistema**

**c) Reducir workers paralelos**

#### 6. Actualizar Drivers NVIDIA
Drivers antiguos pueden tener bugs de gestión de memoria:
```cmd
nvidia-smi
```
Verifica la versión y actualiza si es necesario desde: https://www.nvidia.com/Download/index.aspx

## 📊 Monitoreo de Memoria GPU

### Script de Monitoreo Continuo
Crea un archivo `monitor_gpu.bat`:
```batch
@echo off
:loop
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.free --format=csv,nounits >> gpu_monitor.log
timeout /t 60
goto loop
```

Ejecútalo en paralelo con tu torneo para tener un log completo.

## 🔍 Si el Error Persiste

### 1. Verificar Salud de la GPU
```cmd
nvidia-smi -q -d MEMORY
```

### 2. Reset Completo de GPU
```cmd
nvidia-smi --gpu-reset
```

### 3. Reiniciar el Sistema
A veces el driver NVIDIA necesita un reinicio completo.

### 4. Ejecutar con Menos Épocas
En lugar de todas las épocas, divide en torneos más pequeños:
```python
# En lugar de:
python tournament_parallel_CUDA.py --all

# Hacer:
python tournament_parallel_CUDA.py --epochs 1 50 100 150 200 250 300 350  # Lote 1
python tournament_parallel_CUDA.py --epochs 400 450 500 550 600 650 700 750  # Lote 2
```

## 📈 Mejoras de Rendimiento Implementadas

1. ✅ **Limpieza automática con `finally`**: Garantiza liberación de recursos
2. ✅ **Carga en CPU primero**: Evita fragmentación de VRAM
3. ✅ **Sincronización agresiva**: `torch.cuda.synchronize()` después de operaciones críticas
4. ✅ **Garbage collection forzado**: `gc.collect()` después de liberar modelos
5. ✅ **Checkpoints automáticos**: Guarda progreso cada 1000 enfrentamientos
6. ✅ **Limpieza preventiva**: `empty_cache()` antes de cargar modelos

## 🎮 Ejemplo de Uso Recomendado

Para torneos largos (2+ horas):

```cmd
REM Terminal 1: Ejecutar torneo
python tournament_parallel_CUDA.py --epochs 1 50 100 150 200 --matches 10 --workers 4

REM Terminal 2: Monitorear GPU
nvidia-smi -l 1
```

## 📝 Notas Importantes

- **Los cambios son retrocompatibles**: El código funciona igual que antes, solo más robusto
- **No afecta el rendimiento**: Las optimizaciones son de gestión de memoria, no de cómputo
- **Funciona con multi-GPU**: Las mejoras aplican a configuraciones de múltiples GPUs
- **Checkpoints automáticos**: Nunca perderás más de 1000 enfrentamientos si hay un fallo

## 🆘 Soporte

Si el error persiste después de estas mejoras, proporciona:
1. Salida de `nvidia-smi`
2. Número de épocas y workers usados
3. Tiempo aproximado antes del error
4. Temperatura de GPU cuando ocurrió (si la tienes)

---

**Fecha de actualización**: 2025-10-06
**Versión del código**: tournament_parallel_CUDA.py v2.0 (con fixes de memoria)

