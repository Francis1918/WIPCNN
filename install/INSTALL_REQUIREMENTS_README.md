# 📦 Script Universal de Instalación de Dependencias

Script multiplataforma para instalar y actualizar dependencias de Python en cualquier proyecto.

## 🌟 Características

- ✅ **Multiplataforma**: Compatible con Windows, Linux y macOS
- 🔄 **Actualización automática**: Actualiza pip, setuptools y wheel
- 🐍 **Detección de entornos virtuales**: Detecta venv, virtualenv y conda
- 📋 **Gestión de requirements.txt**: Instala, actualiza y genera archivos de dependencias
- 🛡️ **Manejo robusto de errores**: Mensajes claros y recuperación de errores
- 🎯 **Flexible**: Múltiples opciones de configuración

## 🚀 Uso Rápido

### Instalación básica
```bash
python install_requirements.py --install
```

### Crear entorno virtual e instalar
```bash
python install_requirements.py --create-venv --install
```

### Actualizar todas las dependencias
```bash
python install_requirements.py --upgrade-all
```

### Generar requirements.txt
```bash
python install_requirements.py --freeze
```

## 📖 Opciones Disponibles

| Opción | Descripción |
|--------|-------------|
| `-i, --install` | Instalar dependencias desde requirements.txt |
| `-u, --upgrade-all` | Actualizar todos los paquetes instalados |
| `-f, --freeze` | Generar archivo requirements.txt con paquetes actuales |
| `-l, --list` | Listar todos los paquetes instalados |
| `-r, --requirements PATH` | Especificar ruta al archivo requirements.txt |
| `-o, --output FILE` | Nombre del archivo de salida para --freeze |
| `--create-venv` | Crear entorno virtual si no existe |
| `--skip-pip-upgrade` | No actualizar pip, setuptools y wheel |
| `-v, --verbose` | Mostrar salida detallada |

## 💡 Ejemplos de Uso

### 1. Instalación estándar
```bash
# Instala dependencias desde requirements.txt
python install_requirements.py --install
```

### 2. Configuración completa de proyecto nuevo
```bash
# Crea entorno virtual e instala todo
python install_requirements.py --create-venv --install
```

### 3. Actualizar proyecto existente
```bash
# Actualiza pip y todas las dependencias
python install_requirements.py --install --upgrade-all
```

### 4. Generar requirements.txt actualizado
```bash
# Congela las dependencias actuales
python install_requirements.py --freeze
```

### 5. Usar archivo requirements personalizado
```bash
# Instala desde un archivo específico
python install_requirements.py --install --requirements requirements-dev.txt
```

### 6. Modo verbose para debugging
```bash
# Muestra información detallada del proceso
python install_requirements.py --install --verbose
```

### 7. Listar paquetes instalados
```bash
# Muestra todos los paquetes en el entorno
python install_requirements.py --list
```

### 8. Workflow completo
```bash
# Actualiza todo y genera nuevo requirements.txt
python install_requirements.py --upgrade-all --freeze --output requirements-updated.txt
```

## 🔧 Uso con Scripts de Ayuda

### Windows (PowerShell/CMD)
```cmd
install.bat
```

### Linux/macOS
```bash
chmod +x install.sh
./install.sh
```

## 🏗️ Estructura del Proyecto

El script busca automáticamente `requirements.txt` en las siguientes ubicaciones:
- `requirements.txt` (raíz del proyecto)
- `requirements.pip`
- `requirements/base.txt`
- `requirements/production.txt`
- `requirements/development.txt`

## 🐍 Entornos Virtuales

El script detecta automáticamente:
- **venv/virtualenv**: Busca en `venv`, `.venv`, `env`, `.env`
- **conda**: Detecta variable de entorno `CONDA_DEFAULT_ENV`
- **VIRTUAL_ENV**: Detecta variable de entorno `VIRTUAL_ENV`

### Crear entorno virtual manualmente
```bash
# Opción 1: Usando el script
python install_requirements.py --create-venv

# Opción 2: Manualmente
python -m venv venv

# Activar en Windows
venv\Scripts\activate

# Activar en Linux/macOS
source venv/bin/activate
```

## 🔍 Solución de Problemas

### Error: "No se encontró requirements.txt"
- Asegúrate de estar en el directorio correcto del proyecto
- O especifica la ruta: `--requirements path/to/requirements.txt`

### Error: "Permission denied"
- En Linux/macOS, usa: `sudo python install_requirements.py --install`
- O instala en modo usuario: `pip install --user -r requirements.txt`

### Error: "pip no encontrado"
- Verifica que Python esté instalado: `python --version`
- Reinstala pip: `python -m ensurepip --upgrade`

### Problemas con SSL/Certificados
```bash
# Usar mirror alternativo
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

## 📝 Notas Importantes

1. **Entornos virtuales recomendados**: Siempre usa un entorno virtual para evitar conflictos
2. **Actualización de pip**: El script actualiza pip automáticamente (usa `--skip-pip-upgrade` para omitir)
3. **Compatibilidad**: Funciona con Python 3.6+
4. **Permisos**: Puede requerir permisos de administrador en algunos sistemas

## 🤝 Integración con CI/CD

### GitHub Actions
```yaml
- name: Install dependencies
  run: python install_requirements.py --install
```

### GitLab CI
```yaml
install_deps:
  script:
    - python install_requirements.py --install
```

### Docker
```dockerfile
COPY install_requirements.py requirements.txt ./
RUN python install_requirements.py --install
```

## 📊 Salida del Script

El script proporciona feedback visual con emojis:
- ℹ️ Información
- ✅ Éxito
- ⚠️ Advertencia
- ❌ Error
- 🔄 En progreso

## 🔄 Workflow Recomendado

### Para desarrollo
```bash
# 1. Clonar repositorio
git clone <repo>
cd <proyecto>

# 2. Crear entorno virtual e instalar
python install_requirements.py --create-venv --install

# 3. Activar entorno
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate
```

### Para actualizar dependencias
```bash
# 1. Actualizar todos los paquetes
python install_requirements.py --upgrade-all

# 2. Generar nuevo requirements.txt
python install_requirements.py --freeze

# 3. Commit cambios
git add requirements.txt
git commit -m "Update dependencies"
```

## 📄 Licencia

Este script es de uso libre y puede ser adaptado según las necesidades del proyecto.

## 🆘 Ayuda

Para ver todas las opciones disponibles:
```bash
python install_requirements.py --help