# REPORTE DE ANÁLISIS EXHAUSTIVO DE ARCHIVOS INNECESARIOS
## Proyecto: hierarchical-SAE
**Fecha de análisis:** 2025-10-18  
**Ubicación:** C:\Users\bravo\Documents\Metodos Numericos Pycharm\Mech Interp\hierarchical-SAE

---

## RESUMEN EJECUTIVO

Este reporte identifica **todos los archivos innecesarios, redundantes, temporales y duplicados** del proyecto que pueden ser removidos o archivados sin afectar la funcionalidad principal.

**Total estimado de archivos innecesarios:** ~200+ archivos  
**Espacio estimado a liberar:** Varios GB (principalmente checkpoints de modelos)

---

## CATEGORÍA 1: ARCHIVOS DE DATOS TEMPORALES Y GENERADOS

### 1.1 Archivos Pickle Temporales
**Ubicación:** Raíz del proyecto

| Archivo | Razón | Acción |
|---------|-------|--------|
| `ac_2last_states.pkl` | Datos de entrenamiento temporal | MOVER |
| `ba_increasing_n_last_states.pkl` | Datos de entrenamiento antiguo (mencionado en README pero no existe) | N/A |

**Impacto:** Bajo - Son datos generados que pueden regenerarse

---

### 1.2 Archivos CSV Generados
**Ubicación:** Raíz del proyecto

| Archivo | Razón | Acción |
|---------|-------|--------|
| `board.csv` | Datos del tablero generados automáticamente | MOVER |
| `piece_map.csv` | Mapeo de piezas generado automáticamente | MOVER |

**Impacto:** Bajo - Se regeneran automáticamente al ejecutar el código

---

### 1.3 Imágenes Sin Contexto
**Ubicación:** Raíz del proyecto

| Archivo | Razón | Acción |
|---------|-------|--------|
| `img.png` | Imagen sin propósito claro en documentación | MOVER |

**Impacto:** Mínimo - No referenciada en código

---

## CATEGORÍA 2: SCRIPTS REDUNDANTES Y OBSOLETOS

### 2.1 Scripts de Instalación Duplicados
**Ubicación:** Raíz del proyecto

| Archivo | Razón | Acción Recomendada |
|---------|-------|-------------------|
| `install_cuda_simple.bat` | Redundante con otros scripts de instalación | ELIMINAR (mantener solo setup_dependencies.py) |
| `install_pytorch_cuda_full.py` | Duplicado - hace lo mismo que otros | ELIMINAR |
| `install_pytorch_cuda.py` | Duplicado - hace lo mismo que otros | ELIMINAR |
| `install_pytorch_direct.py` | Duplicado - hace lo mismo que otros | ELIMINAR |

**Recomendación:** Mantener solo `setup_dependencies.py` como script único de instalación

**Impacto:** Ninguno - Son scripts redundantes

---

### 2.2 Scripts de Prueba y Debugging Obsoletos
**Ubicación:** Raíz del proyecto

| Archivo | Razón | Acción |
|---------|-------|--------|
| `a.py` | Script de prueba temporal con ejemplos de TorchRL | ELIMINAR |
| `a.md` | Documentación temporal de estructuras de datos | ELIMINAR |
| `cart_p0ole.py` | Ejemplo de CartPole (no relacionado con Quarto) | ELIMINAR |
| `debugging.py` | Script temporal de debugging | ELIMINAR |
| `test_collector.py` | Pruebas experimentales de colectores | ELIMINAR |
| `try_collector.py` | Experimentos con colectores de TorchRL | ELIMINAR |
| `actions.py` | Definiciones de acciones experimentales | ELIMINAR |

**Impacto:** Ninguno - Son archivos de prueba que no se usan en producción

---

### 2.3 Scripts Vacíos o Sin Uso
**Ubicación:** Raíz del proyecto

| Archivo | Razón | Acción |
|---------|-------|--------|
| `update_requirements.py` | Archivo completamente vacío | ELIMINAR |

**Impacto:** Ninguno - Archivo vacío sin funcionalidad

---

## CATEGORÍA 3: DOCUMENTACIÓN REDUNDANTE Y FRAGMENTADA

### 3.1 Múltiples READMEs de Instalación/Configuración
**Ubicación:** Raíz del proyecto

| Archivo | Razón | Acción Recomendada |
|---------|-------|-------------------|
| `INSTALAR_CUDA_PYTORCH.md` | Documentación específica de instalación CUDA | CONSOLIDAR en README principal |
| `INSTRUCCIONES_PYCHARM.md` | Instrucciones específicas de PyCharm | CONSOLIDAR en README principal |
| `OPTIMIZACIONES_GPU_100.md` | Optimizaciones específicas para GPU | CONSOLIDAR en README principal |
| `README_OPTIMIZACIONES_96GB.md` | Optimizaciones para sistemas con 96GB RAM | CONSOLIDAR en README principal |
| `SOLUCION_CUDA_ERROR_LARGO_PLAZO.md` | Soluciones a errores CUDA | CONSOLIDAR en README principal |
| `TORNEO_MASIVO_README.md` | Documentación de torneos masivos | CONSOLIDAR en README principal |
| `TORNEO_TODOS_CONTRA_TODOS_CUDA.md` | Documentación de torneos CUDA | CONSOLIDAR en README principal |
| `ANALISIS_TOURNAMENT_CUDA.md` | Análisis de torneos CUDA | CONSOLIDAR en README principal |

**Recomendación:** Consolidar toda esta información en:
- `readme.md` (documentación principal)
- `README_DETALLADO.md` (documentación técnica detallada)
- Crear `docs/` folder para documentación adicional si es necesario

**Impacto:** Ninguno - Mejora la organización y mantenibilidad

---

### 3.2 Archivos de Configuración de Usuario
**Ubicación:** Raíz del proyecto

| Archivo | Razón | Acción |
|---------|-------|--------|
| `$PROFILE.txt` | Configuración personal de PowerShell | ELIMINAR (agregar a .gitignore) |
| `Activar entorno Python automática.txt` | Instrucciones personales (archivo no existe) | N/A |

**Impacto:** Ninguno - Configuraciones personales que no deben estar en el repo

---

## CATEGORÍA 4: CONFIGURACIÓN DE IDE

### 4.1 Configuración de PyCharm
**Ubicación:** `.idea/`

| Directorio | Razón | Acción |
|------------|-------|--------|
| `.idea/` | Configuración específica de PyCharm/IntelliJ | ELIMINAR (ya está en .gitignore) |

**Impacto:** Ninguno - Configuración personal del IDE

---

## CATEGORÍA 5: CHECKPOINTS DE MODELOS (CRÍTICO - MAYOR ESPACIO)

### 5.1 Checkpoints Intermedios Excesivos
**Ubicación:** `models/weights/QuartoCNN1/`

**Análisis:**
- Total de archivos .pt: **200+ archivos**
- Rango de épocas: 0 a 1015
- Múltiples checkpoints de época 0 (experimentos fallidos)
- Checkpoints consecutivos cada época

**Archivos Problemáticos:**

#### A) Múltiples Checkpoints de Época 0 (Experimentos Fallidos)
```
20251013_1706-ba_increasing_n_last_states_epoch_0000.pt
20251013_1710-ba_increasing_n_last_states_epoch_0000.pt
20251013_1714-ba_increasing_n_last_states_epoch_0000.pt
20251013_1717-ba_increasing_n_last_states_epoch_0000.pt
20251013_1718-ba_increasing_n_last_states_epoch_0000.pt
20251013_1722-ba_increasing_n_last_states_epoch_0000.pt
20251013_1724-ba_increasing_n_last_states_epoch_0000.pt
20251013_1726-ba_increasing_n_last_states_epoch_0000.pt
20251013_1728-ba_increasing_n_last_states_epoch_0000.pt
20251013_1729-ba_increasing_n_last_states_epoch_0000.pt
20251013_1731-ba_increasing_n_last_states_epoch_0000.pt
20251013_1748-ba_increasing_n_last_states_epoch_0000.pt
20251013_1750-ba_increasing_n_last_states_epoch_0000.pt
20251013_1757-ba_increasing_n_last_states_epoch_0000.pt
20251013_1836-ba_increasing_n_last_states_epoch_0000.pt
```
**Razón:** 15+ intentos de entrenamiento desde época 0 - solo necesitas el último exitoso

#### B) Checkpoints Consecutivos Innecesarios
- Épocas 847-962: **116 checkpoints consecutivos**
- Épocas 964-973: **10 checkpoints consecutivos**
- Épocas 975-1015: **41 checkpoints consecutivos**

**Recomendación de Retención:**
1. **Mantener:** Checkpoints cada 50 o 100 épocas (épocas clave: 0, 50, 100, 150, 200, ..., 1000, 1015)
2. **Mantener:** Último checkpoint (época 1015)
3. **Mantener:** Checkpoints de épocas específicas usadas en torneos
4. **MOVER/ARCHIVAR:** Todos los demás (~180 archivos)

**Espacio estimado a liberar:** 2-5 GB (dependiendo del tamaño de cada checkpoint)

**Impacto:** Bajo - Los checkpoints intermedios raramente se usan después del entrenamiento

---

## CATEGORÍA 6: RESULTADOS DE TORNEOS ANTIGUOS

### 6.1 Directorios de Resultados Temporales
**Ubicación:** Raíz del proyecto

| Directorio | Contenido | Acción |
|------------|-----------|--------|
| `torneomasivo/` | Resultados de torneos masivos antiguos | MOVER |
| `torneomasivo_cli/` | Múltiples torneos CLI con timestamps | MOVER |
| `tournament_parallel/` | Resultados de torneos paralelos antiguos | MOVER |
| `tournament_parallel_massive/` | Resultados de torneos masivos paralelos | MOVER |
| `tournaments_parallel/` | Directorio vacío o duplicado | ELIMINAR |

**Impacto:** Ninguno - Son resultados históricos que pueden archivarse

---

### 6.2 Comparaciones de Agentes Antiguas
**Ubicación:** `analysis/agent_comparisons/`

**Contenido:** 30+ imágenes PNG de comparaciones entre agentes

| Archivos | Razón | Acción |
|----------|-------|--------|
| `comparison_0_vs_*.png` (30+ archivos) | Comparaciones antiguas entre agentes | MOVER |

**Impacto:** Bajo - Son visualizaciones históricas

---

## CATEGORÍA 7: SCRIPTS DUPLICADOS O SIMILARES

### 7.1 Scripts de Torneo Múltiples
**Ubicación:** Raíz del proyecto

| Archivo | Propósito | Estado |
|---------|-----------|--------|
| `tournament.py` | Torneo básico todos contra todos | MANTENER |
| `tournament_parallel.py` | Torneo paralelo optimizado | MANTENER |
| `tournament_parallel_CUDA.py` | Torneo paralelo con CUDA | REVISAR si es diferente |
| `tournament_parallel_massive.py` | Torneo masivo paralelo | REVISAR si es diferente |
| `tournament_bracket.py` | Torneo tipo bracket/eliminación | MANTENER si es diferente |
| `torneomasivo.py` | Torneo masivo | REVISAR si es redundante |

**Recomendación:** Revisar si hay funcionalidad duplicada y consolidar

---

## CATEGORÍA 8: ARCHIVOS DE CACHÉ Y TEMPORALES DEL SISTEMA

### 8.1 Caché de Python
**Ubicación:** Recursivo en todo el proyecto

| Tipo | Ubicación | Acción |
|------|-----------|--------|
| `__pycache__/` | Múltiples directorios | ELIMINAR (regenerable) |
| `*.pyc` | Archivos compilados de Python | ELIMINAR (regenerable) |
| `*.pyo` | Archivos optimizados de Python | ELIMINAR (regenerable) |

**Impacto:** Ninguno - Se regeneran automáticamente

---

### 8.2 Archivos Temporales Comunes
**Ubicación:** Todo el proyecto

| Patrón | Razón | Acción |
|--------|-------|--------|
| `*.tmp` | Archivos temporales | ELIMINAR |
| `*.temp` | Archivos temporales | ELIMINAR |
| `*.bak` | Archivos de respaldo | MOVER |
| `*.swp` | Archivos de swap de editores | ELIMINAR |
| `*~` | Archivos de respaldo de editores | ELIMINAR |

---

## CATEGORÍA 9: DUPLICADOS EN REQUIREMENTS

### 9.1 Dependencias Duplicadas
**Ubicación:** `requirements.txt`

| Problema | Líneas | Acción |
|----------|--------|--------|
| `pandas` aparece 2 veces | Líneas 8 y 10 | **YA CORREGIDO** |

---

## RECOMENDACIONES FINALES

### Prioridad ALTA (Libera más espacio)
1. ✅ **Checkpoints de modelos:** Mantener solo checkpoints clave (cada 50-100 épocas)
   - Archivar ~180 archivos .pt
   - Espacio liberado: 2-5 GB

2. ✅ **Resultados de torneos:** Mover todos los directorios de resultados antiguos
   - `torneomasivo/`, `torneomasivo_cli/`, `tournament_parallel/`, etc.
   - Espacio liberado: 100-500 MB

### Prioridad MEDIA (Mejora organización)
3. ✅ **Documentación:** Consolidar 8 archivos MD en 1-2 archivos principales
4. ✅ **Scripts de instalación:** Eliminar 4 scripts redundantes
5. ✅ **Comparaciones antiguas:** Mover 30+ imágenes PNG de `analysis/agent_comparisons/`

### Prioridad BAJA (Limpieza general)
6. ✅ **Scripts de prueba:** Eliminar 7 archivos de prueba/debugging
7. ✅ **Caché Python:** Eliminar todos los `__pycache__/` y `*.pyc`
8. ✅ **Configuración IDE:** Eliminar `.idea/`
9. ✅ **Archivos temporales:** Eliminar `*.tmp`, `*.bak`, etc.

---

## PLAN DE ACCIÓN SUGERIDO

### Fase 1: Preparación (SIN MODIFICAR PROYECTO)
```bash
# Ejecutar script de análisis para generar este reporte
python cleanup_project.py --dry-run  # (modo simulación)
```

### Fase 2: Limpieza Segura (CON RESPALDO)
```bash
# Mover archivos a carpeta externa con verificación
python cleanup_project.py
```

### Fase 3: Verificación
1. Verificar que el proyecto funciona correctamente
2. Ejecutar tests básicos
3. Confirmar que no se perdió información crítica

### Fase 4: Consolidación (Opcional)
1. Consolidar documentación MD
2. Revisar y eliminar scripts duplicados de torneos
3. Actualizar .gitignore para prevenir futuros archivos innecesarios

---

## ARCHIVOS A MANTENER (ESENCIALES)

### Código Fuente
- ✅ `bot/` - Implementaciones de bots
- ✅ `models/` - Arquitecturas de modelos (excepto checkpoints antiguos)
- ✅ `QuartoRL/` - Funciones de RL
- ✅ `checkpoint_monitor/` - Sistema de monitoreo
- ✅ `utils/` - Utilidades
- ✅ `tests/` - Tests en notebooks

### Scripts Principales
- ✅ `trainRL.py` - Script principal de entrenamiento
- ✅ `trainRL_resume.py` - Reanudar entrenamiento
- ✅ `compare_agents.py` - Comparar agentes
- ✅ `tournament_parallel.py` - Torneo paralelo
- ✅ `check_cuda.py` - Verificar CUDA
- ✅ `setup_dependencies.py` - Instalación de dependencias

### Documentación
- ✅ `readme.md` - README principal
- ✅ `README_DETALLADO.md` - Documentación detallada
- ✅ `LICENSE` - Licencia del proyecto

### Configuración
- ✅ `requirements.txt` - Dependencias (ya corregido)
- ✅ `.gitignore` - Archivos ignorados por Git
- ✅ `.gitmodules` - Submódulos de Git

---

## CONCLUSIÓN

El proyecto tiene una cantidad significativa de archivos innecesarios que pueden ser removidos o archivados de forma segura:

- **~200+ archivos** pueden ser movidos o eliminados
- **2-5 GB** de espacio puede ser liberado
- **Organización mejorada** al consolidar documentación
- **Sin impacto** en la funcionalidad principal del proyecto

**Próximo paso:** Ejecutar `cleanup_project.py` para mover archivos temporales a la carpeta externa de forma segura.

---

**Generado por:** Análisis exhaustivo del proyecto  
**Fecha:** 2025-10-18  
**Versión del reporte:** 1.0