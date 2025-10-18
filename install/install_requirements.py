#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script universal para instalar y actualizar dependencias de Python.
Compatible con cualquier proyecto y sistema operativo.

Características:
- Detecta automáticamente el sistema operativo
- Maneja entornos virtuales (venv, virtualenv, conda)
- Actualiza pip, setuptools y wheel
- Instala dependencias desde requirements.txt
- Opción para actualizar todas las dependencias
- Manejo robusto de errores
- Compatible con Windows, Linux y macOS
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

# Configurar encoding para Windows
if platform.system() == 'Windows':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


class DependencyInstaller:
    """Clase para manejar la instalación y actualización de dependencias."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.system = platform.system()
        self.python_executable = sys.executable
        self.project_root = Path.cwd()
        
    def log(self, message: str, level: str = "INFO"):
        """Imprime mensajes con formato."""
        # Usar símbolos ASCII en Windows para evitar problemas de encoding
        if self.system == "Windows":
            prefix = {
                "INFO": "[INFO]",
                "SUCCESS": "[OK]",
                "WARNING": "[WARN]",
                "ERROR": "[ERROR]",
                "PROGRESS": "[...]"
            }.get(level, "[*]")
        else:
            prefix = {
                "INFO": "ℹ️",
                "SUCCESS": "✅",
                "WARNING": "⚠️",
                "ERROR": "❌",
                "PROGRESS": "🔄"
            }.get(level, "•")
        
        try:
            print(f"{prefix} {message}")
        except UnicodeEncodeError:
            # Fallback a ASCII si hay problemas
            ascii_prefix = {
                "INFO": "[INFO]",
                "SUCCESS": "[OK]",
                "WARNING": "[WARN]",
                "ERROR": "[ERROR]",
                "PROGRESS": "[...]"
            }.get(level, "[*]")
            print(f"{ascii_prefix} {message}")
    
    def run_command(self, command: List[str], check: bool = True) -> Tuple[bool, str]:
        """
        Ejecuta un comando y retorna el resultado.
        
        Args:
            command: Lista con el comando y sus argumentos
            check: Si debe lanzar excepción en caso de error
            
        Returns:
            Tupla (éxito, salida)
        """
        try:
            if self.verbose:
                self.log(f"Ejecutando: {' '.join(command)}", "PROGRESS")
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=check
            )
            
            if self.verbose and result.stdout:
                print(result.stdout)
            
            return True, result.stdout
        
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            if self.verbose:
                self.log(f"Error: {error_msg}", "ERROR")
            return False, error_msg
        
        except Exception as e:
            self.log(f"Error inesperado: {str(e)}", "ERROR")
            return False, str(e)
    
    def detect_virtual_env(self) -> Optional[str]:
        """Detecta si está en un entorno virtual."""
        # Verifica VIRTUAL_ENV
        if os.environ.get('VIRTUAL_ENV'):
            return 'venv'
        
        # Verifica Conda
        if os.environ.get('CONDA_DEFAULT_ENV'):
            return 'conda'
        
        # Verifica si hay un venv en el directorio actual
        venv_paths = ['venv', '.venv', 'env', '.env']
        for venv_path in venv_paths:
            if (self.project_root / venv_path).exists():
                return 'venv_local'
        
        return None
    
    def create_virtual_env(self, env_name: str = "venv") -> bool:
        """Crea un entorno virtual si no existe."""
        env_path = self.project_root / env_name
        
        if env_path.exists():
            self.log(f"El entorno virtual '{env_name}' ya existe", "INFO")
            return True
        
        self.log(f"Creando entorno virtual '{env_name}'...", "PROGRESS")
        success, _ = self.run_command([self.python_executable, "-m", "venv", str(env_path)])
        
        if success:
            self.log(f"Entorno virtual '{env_name}' creado exitosamente", "SUCCESS")
            return True
        else:
            self.log(f"No se pudo crear el entorno virtual", "ERROR")
            return False
    
    def get_pip_command(self) -> List[str]:
        """Retorna el comando pip apropiado para el sistema."""
        return [self.python_executable, "-m", "pip"]
    
    def upgrade_pip(self) -> bool:
        """Actualiza pip, setuptools y wheel."""
        self.log("Actualizando pip, setuptools y wheel...", "PROGRESS")
        
        pip_cmd = self.get_pip_command()
        success, _ = self.run_command(
            pip_cmd + ["install", "--upgrade", "pip", "setuptools", "wheel"]
        )
        
        if success:
            self.log("Herramientas base actualizadas correctamente", "SUCCESS")
        else:
            self.log("Advertencia: No se pudieron actualizar las herramientas base", "WARNING")
        
        return success
    
    def find_requirements_file(self) -> Optional[Path]:
        """Busca el archivo requirements.txt en el proyecto."""
        possible_names = [
            "requirements.txt",
            "requirements.pip",
            "requirements/base.txt",
            "requirements/production.txt",
            "requirements/development.txt"
        ]
        
        for name in possible_names:
            req_file = self.project_root / name
            if req_file.exists():
                return req_file
        
        return None
    
    def install_requirements(self, requirements_file: Optional[Path] = None) -> bool:
        """Instala las dependencias desde requirements.txt."""
        if requirements_file is None:
            requirements_file = self.find_requirements_file()
        
        if requirements_file is None:
            self.log("No se encontró archivo requirements.txt", "WARNING")
            return False
        
        if not requirements_file.exists():
            self.log(f"El archivo {requirements_file} no existe", "ERROR")
            return False
        
        self.log(f"Instalando dependencias desde {requirements_file.name}...", "PROGRESS")
        
        pip_cmd = self.get_pip_command()
        success, output = self.run_command(
            pip_cmd + ["install", "-r", str(requirements_file)]
        )
        
        if success:
            self.log("Dependencias instaladas correctamente", "SUCCESS")
        else:
            self.log("Error al instalar dependencias", "ERROR")
        
        return success
    
    def upgrade_all_packages(self) -> bool:
        """Actualiza todos los paquetes instalados."""
        self.log("Obteniendo lista de paquetes instalados...", "PROGRESS")
        
        pip_cmd = self.get_pip_command()
        success, output = self.run_command(pip_cmd + ["list", "--outdated", "--format=json"])
        
        if not success:
            self.log("No se pudo obtener la lista de paquetes", "ERROR")
            return False
        
        try:
            import json
            outdated = json.loads(output)
            
            if not outdated:
                self.log("Todos los paquetes están actualizados", "SUCCESS")
                return True
            
            self.log(f"Actualizando {len(outdated)} paquetes...", "PROGRESS")
            
            for package in outdated:
                package_name = package['name']
                self.log(f"Actualizando {package_name}...", "PROGRESS")
                self.run_command(
                    pip_cmd + ["install", "--upgrade", package_name],
                    check=False
                )
            
            self.log("Todos los paquetes actualizados", "SUCCESS")
            return True
        
        except Exception as e:
            self.log(f"Error al actualizar paquetes: {str(e)}", "ERROR")
            return False
    
    def freeze_requirements(self, output_file: str = "requirements.txt") -> bool:
        """Genera un archivo requirements.txt con las dependencias actuales."""
        self.log(f"Generando {output_file}...", "PROGRESS")
        
        pip_cmd = self.get_pip_command()
        success, output = self.run_command(pip_cmd + ["freeze"])
        
        if not success:
            self.log("No se pudo generar requirements.txt", "ERROR")
            return False
        
        try:
            output_path = self.project_root / output_file
            output_path.write_text(output, encoding='utf-8')
            self.log(f"Archivo {output_file} generado correctamente", "SUCCESS")
            return True
        
        except Exception as e:
            self.log(f"Error al escribir {output_file}: {str(e)}", "ERROR")
            return False
    
    def show_installed_packages(self):
        """Muestra los paquetes instalados."""
        self.log("Paquetes instalados:", "INFO")
        pip_cmd = self.get_pip_command()
        self.run_command(pip_cmd + ["list"])
    
    def run(self, args):
        """Ejecuta el proceso completo de instalación/actualización."""
        self.log(f"Sistema operativo: {self.system}", "INFO")
        self.log(f"Python: {sys.version.split()[0]}", "INFO")
        self.log(f"Directorio del proyecto: {self.project_root}", "INFO")
        
        # Detectar entorno virtual
        venv_type = self.detect_virtual_env()
        if venv_type:
            self.log(f"Entorno virtual detectado: {venv_type}", "INFO")
        else:
            self.log("No se detectó entorno virtual", "WARNING")
            if args.create_venv:
                if not self.create_virtual_env():
                    return False
        
        print()  # Línea en blanco
        
        # Actualizar pip
        if not args.skip_pip_upgrade:
            self.upgrade_pip()
            print()
        
        # Instalar requirements
        if args.install:
            req_file = Path(args.requirements) if args.requirements else None
            self.install_requirements(req_file)
            print()
        
        # Actualizar todos los paquetes
        if args.upgrade_all:
            self.upgrade_all_packages()
            print()
        
        # Generar requirements.txt
        if args.freeze:
            self.freeze_requirements(args.output)
            print()
        
        # Mostrar paquetes instalados
        if args.list:
            self.show_installed_packages()
        
        self.log("Proceso completado", "SUCCESS")
        return True


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Script universal para gestionar dependencias de Python",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  %(prog)s --install                    # Instalar desde requirements.txt
  %(prog)s --install --upgrade-all      # Instalar y actualizar todo
  %(prog)s --freeze                     # Generar requirements.txt
  %(prog)s --create-venv --install      # Crear venv e instalar
  %(prog)s --list                       # Listar paquetes instalados
        """
    )
    
    parser.add_argument(
        "-i", "--install",
        action="store_true",
        help="Instalar dependencias desde requirements.txt"
    )
    
    parser.add_argument(
        "-u", "--upgrade-all",
        action="store_true",
        help="Actualizar todos los paquetes instalados"
    )
    
    parser.add_argument(
        "-f", "--freeze",
        action="store_true",
        help="Generar archivo requirements.txt con paquetes actuales"
    )
    
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="Listar todos los paquetes instalados"
    )
    
    parser.add_argument(
        "-r", "--requirements",
        type=str,
        help="Ruta al archivo requirements.txt (por defecto: busca automáticamente)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="requirements.txt",
        help="Nombre del archivo de salida para --freeze (por defecto: requirements.txt)"
    )
    
    parser.add_argument(
        "--create-venv",
        action="store_true",
        help="Crear entorno virtual si no existe"
    )
    
    parser.add_argument(
        "--skip-pip-upgrade",
        action="store_true",
        help="No actualizar pip, setuptools y wheel"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Mostrar salida detallada"
    )
    
    args = parser.parse_args()
    
    # Si no se especifica ninguna acción, instalar por defecto
    if not any([args.install, args.upgrade_all, args.freeze, args.list]):
        args.install = True
    
    installer = DependencyInstaller(verbose=args.verbose)
    
    try:
        success = installer.run(args)
        sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        installer.log("\nProceso interrumpido por el usuario", "WARNING")
        sys.exit(1)
    
    except Exception as e:
        installer.log(f"Error fatal: {str(e)}", "ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main()