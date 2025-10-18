# 📦 Scripts Legacy

Esta carpeta contiene versiones originales de scripts que han sido mejorados y movidos a otras ubicaciones.

## Archivos

### script_para_comparar_entre_agentes_original.py

**Estado:** Reemplazado por versión mejorada

**Nueva ubicación:** [`tournaments/manual_bot_comparison.py`](../../tournaments/manual_bot_comparison.py)

**Razón del reemplazo:**
- Script original requería editar código para cambiar parámetros
- Rutas hardcodeadas
- Sin guardado automático de resultados
- Sin manejo de errores robusto

**Versión mejorada incluye:**
- ✅ Parametrización completa por línea de comandos
- ✅ Guardado automático en JSON/CSV
- ✅ Manejo robusto de errores
- ✅ Validación de archivos
- ✅ Estadísticas mejoradas
- ✅ Documentación completa

**Documentación:** Ver [`tournaments/MANUAL_COMPARISON_README.md`](../../tournaments/MANUAL_COMPARISON_README.md)

**Uso de la nueva versión:**
```bash
python tournaments/manual_bot_comparison.py model1.pt model2.pt --matches 500
```

## ¿Por qué mantener estos archivos?

Los scripts legacy se mantienen como:
- Referencia histórica
- Backup en caso de necesitar funcionalidad específica
- Documentación de la evolución del proyecto

## Nota

Estos scripts pueden no funcionar con la estructura actual del proyecto. Se recomienda usar las versiones mejoradas en sus ubicaciones correspondientes.

---

**Última actualización:** 2025-10-18