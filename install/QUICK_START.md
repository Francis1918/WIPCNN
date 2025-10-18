# 🚀 Guía Rápida - Instalador de Dependencias

## ⚡ Inicio Rápido (3 pasos)

### Windows
```cmd
# 1. Ejecutar el instalador
install.bat

# 2. Seleccionar opción 2 (crear venv e instalar)

# 3. Activar entorno virtual
venv\Scripts\activate
```

### Linux/macOS
```bash
# 1. Dar permisos de ejecución
chmod +x install.sh

# 2. Ejecutar el instalador
./install.sh

# 3. Seleccionar opción 2 (crear venv e instalar)

# 4. Activar entorno virtual
source venv/bin/activate
```

## 📋 Comandos Más Usados

### Instalación Simple
```bash
# Instalar dependencias existentes
python install_requirements.py --install
```

### Proyecto Nuevo
```bash
# Configurar todo desde cero
python install_requirements.py --create-venv --install
```

### Actualizar Proyecto
```bash
# Actualizar todas las dependencias
python install_requirements.py --upgrade-all
```

### Guardar Dependencias
```bash
# Generar requirements.txt actualizado
python install_requirements.py --freeze
```

## 🎯 Casos de Uso Comunes

### 1️⃣ Cloné un repositorio, ¿qué hago?
```bash
cd proyecto-clonado
python install_requirements.py --create-venv --install
```

### 2️⃣ Quiero actualizar mis dependencias
```bash
python install_requirements.py --upgrade-all --freeze
```

### 3️⃣ Instalé nuevos paquetes manualmente
```bash
# Guardar los cambios
python install_requirements.py --freeze
```

### 4️⃣ Tengo múltiples archivos requirements
```bash
# Instalar desde archivo específico
python install_requirements.py --install --requirements requirements-dev.txt
```

### 5️⃣ Quiero ver qué tengo instalado
```bash
python install_requirements.py --list
```

## 🔧 Solución Rápida de Problemas

### ❌ "Python no encontrado"
**Windows:**
```cmd
# Verificar instalación
python --version

# Si no funciona, reinstalar desde:
# https://www.python.org/downloads/
```

**Linux/macOS:**
```bash
# Verificar instalación
python3 --version

# Instalar si es necesario
# Ubuntu/Debian: sudo apt install python3 python3-pip
# macOS: brew install python3
```

### ❌ "No se encontró requirements.txt"
```bash
# Crear uno nuevo con tus paquetes actuales
python install_requirements.py --freeze
```

### ❌ "Permission denied" (Linux/macOS)
```bash
# Opción 1: Dar permisos
chmod +x install.sh

# Opción 2: Ejecutar con bash
bash install.sh

# Opción 3: Usar sudo si es necesario
sudo python3 install_requirements.py --install
```

### ❌ Error de SSL/Certificados
```bash
# Usar repositorio confiable
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

## 📦 Workflow Recomendado

### Para Desarrollo
```bash
# 1. Clonar proyecto
git clone <url-del-repo>
cd <nombre-proyecto>

# 2. Configurar entorno
python install_requirements.py --create-venv --install

# 3. Activar entorno
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate

# 4. ¡Empezar a trabajar!
```

### Para Actualizar Dependencias
```bash
# 1. Actualizar paquetes
python install_requirements.py --upgrade-all

# 2. Probar que todo funciona
python -m pytest  # o tu comando de tests

# 3. Guardar cambios
python install_requirements.py --freeze

# 4. Commit
git add requirements.txt
git commit -m "Update dependencies"
git push
```

## 🎨 Menú Interactivo

Si ejecutas sin argumentos, obtienes un menú:

**Windows:**
```cmd
install.bat
```

**Linux/macOS:**
```bash
./install.sh
```

Opciones del menú:
1. ✅ Instalar dependencias
2. 🆕 Crear venv e instalar
3. 🔄 Actualizar todo
4. 💾 Generar requirements.txt
5. 📋 Listar paquetes
6. 🚀 Instalación completa
7. ⚙️ Modo avanzado
8. 🚪 Salir

## 💡 Tips Pro

### Usar con diferentes Python
```bash
# Python específico
python3.9 install_requirements.py --install

# O con ruta completa
/usr/bin/python3 install_requirements.py --install
```

### Modo verbose para debugging
```bash
python install_requirements.py --install --verbose
```

### Combinar múltiples opciones
```bash
python install_requirements.py --create-venv --install --upgrade-all --verbose
```

### Generar requirements con nombre personalizado
```bash
python install_requirements.py --freeze --output requirements-prod.txt
```

## 🔗 Enlaces Útiles

- 📖 [Documentación completa](INSTALL_REQUIREMENTS_README.md)
- 🐍 [Python.org](https://www.python.org/)
- 📦 [PyPI](https://pypi.org/)
- 🛠️ [pip Documentation](https://pip.pypa.io/)

## ❓ Ayuda

Ver todas las opciones:
```bash
python install_requirements.py --help
```

## 📞 Soporte

Si encuentras problemas:
1. Verifica que Python esté instalado correctamente
2. Asegúrate de estar en el directorio correcto del proyecto
3. Revisa que `requirements.txt` exista
4. Usa `--verbose` para más información
5. Consulta la [documentación completa](INSTALL_REQUIREMENTS_README.md)