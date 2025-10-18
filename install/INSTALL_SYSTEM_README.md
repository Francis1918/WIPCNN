# 📦 Sistema Universal de Instalación de Dependencias

## 📁 Archivos Creados

Este sistema incluye los siguientes archivos:

1. **[`install_requirements.py`](install_requirements.py)** - Script principal en Python
2. **[`install.bat`](install.bat)** - Script de ayuda para Windows
3. **[`install.sh`](install.sh)** - Script de ayuda para Linux/macOS
4. **[`INSTALL_REQUIREMENTS_README.md`](INSTALL_REQUIREMENTS_README.md)** - Documentación completa
5. **[`QUICK_START.md`](QUICK_START.md)** - Guía rápida de inicio
6. **Este archivo** - Resumen del sistema

## 🚀 Inicio Rápido

### Windows
```cmd
install.bat
```

### Linux/macOS
```bash
chmod +x install.sh
./install.sh
```

### Uso Directo
```bash
python install_requirements.py --install
```

## ✨ Características Principales

- ✅ **Multiplataforma**: Windows, Linux, macOS
- 🔄 **Actualización automática**: pip, setuptools, wheel
- 🐍 **Detección de entornos virtuales**: venv, virtualenv, conda
- 📋 **Gestión completa**: instalar, actualizar, generar requirements.txt
- 🛡️ **Manejo robusto de errores**: Mensajes claros y recuperación
- 🎯 **Flexible**: Múltiples opciones de configuración
- 🌐 **Compatible con encoding**: Funciona correctamente en cualquier sistema

## 📖 Documentación

- **Documentación Completa**: [`INSTALL_REQUIREMENTS_README.md`](INSTALL_REQUIREMENTS_README.md)
- **Guía Rápida**: [`QUICK_START.md`](QUICK_START.md)
- **Ayuda del Script**: `python install_requirements.py --help`

## 🎯 Casos de Uso Comunes

### 1. Proyecto Nuevo
```bash
# Crear entorno virtual e instalar dependencias
python install_requirements.py --create-venv --install
```

### 2. Clonar Repositorio
```bash
git clone <repo>
cd <proyecto>
python install_requirements.py --install
```

### 3. Actualizar Dependencias
```bash
# Actualizar todos los paquetes
python install_requirements.py --upgrade-all
```

### 4. Guardar Dependencias
```bash
# Generar requirements.txt actualizado
python install_requirements.py --freeze
```

## 🔧 Opciones Disponibles

| Opción | Descripción |
|--------|-------------|
| `--install` | Instalar desde requirements.txt |
| `--upgrade-all` | Actualizar todos los paquetes |
| `--freeze` | Generar requirements.txt |
| `--list` | Listar paquetes instalados |
| `--create-venv` | Crear entorno virtual |
| `--verbose` | Modo detallado |

## 🌍 Compatibilidad

### Sistemas Operativos
- ✅ Windows 10/11
- ✅ Linux (Ubuntu, Debian, Fedora, etc.)
- ✅ macOS

### Versiones de Python
- ✅ Python 3.6+
- ✅ Python 3.7+
- ✅ Python 3.8+
- ✅ Python 3.9+
- ✅ Python 3.10+
- ✅ Python 3.11+
- ✅ Python 3.12+
- ✅ Python 3.13+

### Gestores de Entornos
- ✅ venv (estándar)
- ✅ virtualenv
- ✅ conda
- ✅ Sin entorno virtual

## 📊 Ejemplo de Salida

```
[INFO] Sistema operativo: Windows
[INFO] Python: 3.13.9
[INFO] Directorio del proyecto: C:\proyecto
[INFO] Entorno virtual detectado: venv_local

[...] Actualizando pip, setuptools y wheel...
[OK] Herramientas base actualizadas correctamente

[...] Instalando dependencias desde requirements.txt...
[OK] Dependencias instaladas correctamente

[OK] Proceso completado
```

## 🔄 Workflow Recomendado

### Desarrollo Diario
```bash
# 1. Activar entorno
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# 2. Trabajar en el proyecto
# ... hacer cambios ...

# 3. Si instalaste nuevos paquetes
python install_requirements.py --freeze
```

### Actualización Periódica
```bash
# Cada semana/mes
python install_requirements.py --upgrade-all --freeze
git add requirements.txt
git commit -m "Update dependencies"
```

## 🛠️ Integración con Herramientas

### Git Hooks (pre-commit)
```bash
#!/bin/bash
python install_requirements.py --freeze
git add requirements.txt
```

### Docker
```dockerfile
COPY install_requirements.py requirements.txt ./
RUN python install_requirements.py --install
```

### CI/CD
```yaml
# GitHub Actions
- name: Install dependencies
  run: python install_requirements.py --install

# GitLab CI
install:
  script:
    - python install_requirements.py --install
```

## 🎓 Mejores Prácticas

1. **Siempre usa entornos virtuales** para aislar dependencias
2. **Actualiza regularmente** las dependencias para seguridad
3. **Genera requirements.txt** después de instalar nuevos paquetes
4. **Versiona requirements.txt** en tu repositorio
5. **Documenta dependencias especiales** en comentarios

## 🔍 Solución de Problemas

### Problema: Script no ejecuta
**Solución:**
```bash
# Verificar Python
python --version

# Dar permisos (Linux/macOS)
chmod +x install.sh

# Ejecutar directamente
python install_requirements.py --install
```

### Problema: Errores de encoding
**Solución:** El script maneja automáticamente problemas de encoding en Windows

### Problema: No encuentra requirements.txt
**Solución:**
```bash
# Especificar ruta
python install_requirements.py --install --requirements path/to/requirements.txt

# O generar uno nuevo
python install_requirements.py --freeze
```

## 📦 Portabilidad

Este sistema es **completamente portable**. Para usarlo en otro proyecto:

1. Copia los archivos a tu nuevo proyecto:
   - `install_requirements.py`
   - `install.bat` (opcional, para Windows)
   - `install.sh` (opcional, para Linux/macOS)

2. Ejecuta según tu sistema operativo

3. ¡Listo! El script se adapta automáticamente

## 🎯 Ventajas sobre pip install -r

| Característica | pip install -r | Este Sistema |
|----------------|----------------|--------------|
| Actualiza pip automáticamente | ❌ | ✅ |
| Detecta entornos virtuales | ❌ | ✅ |
| Crea entornos virtuales | ❌ | ✅ |
| Actualiza todas las dependencias | ❌ | ✅ |
| Genera requirements.txt | ❌ | ✅ |
| Menú interactivo | ❌ | ✅ |
| Manejo robusto de errores | ⚠️ | ✅ |
| Multiplataforma garantizado | ⚠️ | ✅ |
| Scripts de ayuda incluidos | ❌ | ✅ |

## 📞 Soporte

Para más información:
- Ver ayuda: `python install_requirements.py --help`
- Leer documentación: [`INSTALL_REQUIREMENTS_README.md`](INSTALL_REQUIREMENTS_README.md)
- Guía rápida: [`QUICK_START.md`](QUICK_START.md)

## 📝 Licencia

Este sistema es de uso libre y puede ser adaptado según las necesidades de cualquier proyecto.

---

**Creado para facilitar la gestión de dependencias en proyectos Python** 🐍