@echo off
REM Script de instalación de dependencias para Windows
REM Uso: install.bat [opciones]

echo ========================================
echo   Instalador de Dependencias Python
echo ========================================
echo.

REM Verificar si Python está instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no está instalado o no está en el PATH
    echo Por favor instala Python desde https://www.python.org/
    pause
    exit /b 1
)

echo [INFO] Python detectado correctamente
echo.

REM Si se pasan argumentos, usarlos directamente
if not "%~1"=="" (
    echo [INFO] Ejecutando con argumentos personalizados...
    python install_requirements.py %*
    goto :end
)

REM Menú interactivo
:menu
echo Selecciona una opción:
echo.
echo 1. Instalar dependencias (requirements.txt)
echo 2. Crear entorno virtual e instalar
echo 3. Actualizar todas las dependencias
echo 4. Generar requirements.txt
echo 5. Listar paquetes instalados
echo 6. Instalación completa (crear venv + instalar + actualizar)
echo 7. Modo avanzado (especificar opciones)
echo 8. Salir
echo.

set /p choice="Ingresa tu opción (1-8): "

if "%choice%"=="1" (
    echo.
    echo [INFO] Instalando dependencias...
    python install_requirements.py --install
    goto :end
)

if "%choice%"=="2" (
    echo.
    echo [INFO] Creando entorno virtual e instalando...
    python install_requirements.py --create-venv --install
    echo.
    echo [INFO] Para activar el entorno virtual, ejecuta:
    echo        venv\Scripts\activate
    goto :end
)

if "%choice%"=="3" (
    echo.
    echo [INFO] Actualizando todas las dependencias...
    python install_requirements.py --upgrade-all
    goto :end
)

if "%choice%"=="4" (
    echo.
    echo [INFO] Generando requirements.txt...
    python install_requirements.py --freeze
    goto :end
)

if "%choice%"=="5" (
    echo.
    echo [INFO] Listando paquetes instalados...
    python install_requirements.py --list
    goto :end
)

if "%choice%"=="6" (
    echo.
    echo [INFO] Instalación completa...
    python install_requirements.py --create-venv --install --upgrade-all
    echo.
    echo [INFO] Para activar el entorno virtual, ejecuta:
    echo        venv\Scripts\activate
    goto :end
)

if "%choice%"=="7" (
    echo.
    echo Opciones disponibles:
    echo   --install              Instalar dependencias
    echo   --upgrade-all          Actualizar todo
    echo   --freeze               Generar requirements.txt
    echo   --create-venv          Crear entorno virtual
    echo   --list                 Listar paquetes
    echo   --verbose              Modo detallado
    echo.
    set /p custom_opts="Ingresa las opciones: "
    python install_requirements.py %custom_opts%
    goto :end
)

if "%choice%"=="8" (
    echo.
    echo [INFO] Saliendo...
    exit /b 0
)

echo.
echo [ERROR] Opción inválida. Por favor selecciona 1-8.
echo.
goto :menu

:end
echo.
echo ========================================
echo   Proceso completado
echo ========================================
pause