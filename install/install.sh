#!/bin/bash
# Script de instalación de dependencias para Linux/macOS
# Uso: ./install.sh [opciones]

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para imprimir mensajes
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo "========================================"
    echo "  Instalador de Dependencias Python"
    echo "========================================"
    echo ""
}

# Verificar si Python está instalado
check_python() {
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        print_error "Python no está instalado"
        print_info "Por favor instala Python desde https://www.python.org/"
        exit 1
    fi
    
    # Determinar el comando de Python
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        PYTHON_CMD="python"
    fi
    
    print_success "Python detectado: $($PYTHON_CMD --version)"
    echo ""
}

# Menú interactivo
show_menu() {
    echo "Selecciona una opción:"
    echo ""
    echo "1. Instalar dependencias (requirements.txt)"
    echo "2. Crear entorno virtual e instalar"
    echo "3. Actualizar todas las dependencias"
    echo "4. Generar requirements.txt"
    echo "5. Listar paquetes instalados"
    echo "6. Instalación completa (crear venv + instalar + actualizar)"
    echo "7. Modo avanzado (especificar opciones)"
    echo "8. Salir"
    echo ""
}

# Función para activar entorno virtual
show_activation_instructions() {
    echo ""
    print_info "Para activar el entorno virtual, ejecuta:"
    echo "        source venv/bin/activate"
}

# Función principal
main() {
    print_header
    check_python
    
    # Si se pasan argumentos, usarlos directamente
    if [ $# -gt 0 ]; then
        print_info "Ejecutando con argumentos personalizados..."
        $PYTHON_CMD install_requirements.py "$@"
        exit $?
    fi
    
    # Menú interactivo
    while true; do
        show_menu
        read -p "Ingresa tu opción (1-8): " choice
        echo ""
        
        case $choice in
            1)
                print_info "Instalando dependencias..."
                $PYTHON_CMD install_requirements.py --install
                break
                ;;
            2)
                print_info "Creando entorno virtual e instalando..."
                $PYTHON_CMD install_requirements.py --create-venv --install
                show_activation_instructions
                break
                ;;
            3)
                print_info "Actualizando todas las dependencias..."
                $PYTHON_CMD install_requirements.py --upgrade-all
                break
                ;;
            4)
                print_info "Generando requirements.txt..."
                $PYTHON_CMD install_requirements.py --freeze
                break
                ;;
            5)
                print_info "Listando paquetes instalados..."
                $PYTHON_CMD install_requirements.py --list
                break
                ;;
            6)
                print_info "Instalación completa..."
                $PYTHON_CMD install_requirements.py --create-venv --install --upgrade-all
                show_activation_instructions
                break
                ;;
            7)
                echo "Opciones disponibles:"
                echo "  --install              Instalar dependencias"
                echo "  --upgrade-all          Actualizar todo"
                echo "  --freeze               Generar requirements.txt"
                echo "  --create-venv          Crear entorno virtual"
                echo "  --list                 Listar paquetes"
                echo "  --verbose              Modo detallado"
                echo ""
                read -p "Ingresa las opciones: " custom_opts
                $PYTHON_CMD install_requirements.py $custom_opts
                break
                ;;
            8)
                print_info "Saliendo..."
                exit 0
                ;;
            *)
                print_error "Opción inválida. Por favor selecciona 1-8."
                echo ""
                ;;
        esac
    done
    
    echo ""
    echo "========================================"
    echo "  Proceso completado"
    echo "========================================"
}

# Ejecutar función principal
main "$@"