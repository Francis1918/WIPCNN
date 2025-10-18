#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
check_cuda.py - Diagnóstico completo de CUDA y PyTorch
Verifica la instalación de CUDA y proporciona comandos de instalación.
"""

import subprocess
import sys
import platform

print("="*80)
print("🔍 DIAGNÓSTICO DE CUDA Y PYTORCH")
print("="*80)

# 1. Verificar sistema operativo
print(f"\n📋 Sistema Operativo: {platform.system()} {platform.release()}")
print(f"   Arquitectura: {platform.machine()}")
print(f"   Python: {sys.version}")

# 2. Verificar PyTorch
print("\n" + "="*80)
print("📦 PYTORCH INSTALADO")
print("="*80)

try:
    import torch
    print(f"✅ PyTorch versión: {torch.__version__}")
    print(f"   CUDA disponible: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"   CUDA versión (PyTorch): {torch.version.cuda}")
        print(f"   Número de GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\n   GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"      Memoria total: {props.total_memory / (1024**3):.2f} GB")
            print(f"      Compute Capability: {props.major}.{props.minor}")
    else:
        print("   ⚠️  CUDA NO está disponible en PyTorch")
        print(f"   PyTorch compilado con CUDA: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'No'}")

except ImportError:
    print("❌ PyTorch NO está instalado")

# 3. Verificar NVIDIA Driver
print("\n" + "="*80)
print("🎮 NVIDIA GPU Y DRIVERS")
print("="*80)

try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("✅ NVIDIA Driver instalado\n")
        # Extraer versión del driver y CUDA
        lines = result.stdout.split('\n')
        for line in lines[:10]:  # Primeras 10 líneas tienen la info importante
            if 'CUDA Version' in line or 'Driver Version' in line or 'NVIDIA' in line:
                print(f"   {line.strip()}")

        # Mostrar GPUs detectadas
        print("\n   GPUs detectadas:")
        for line in lines:
            if 'GeForce' in line or 'RTX' in line or 'GTX' in line or 'Quadro' in line or 'Tesla' in line:
                print(f"   {line.strip()}")

    else:
        print("❌ nvidia-smi no disponible o error al ejecutar")
except FileNotFoundError:
    print("❌ nvidia-smi no encontrado")
    print("   Esto significa que NO tienes drivers NVIDIA instalados")
except Exception as e:
    print(f"⚠️  Error al ejecutar nvidia-smi: {e}")

# 4. Verificar CUDA Toolkit
print("\n" + "="*80)
print("🔧 CUDA TOOLKIT")
print("="*80)

try:
    result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("✅ CUDA Toolkit instalado")
        print(result.stdout)
    else:
        print("⚠️  nvcc no disponible")
except FileNotFoundError:
    print("⚠️  CUDA Toolkit no instalado (nvcc no encontrado)")
    print("   Nota: No es estrictamente necesario para PyTorch")
except Exception as e:
    print(f"⚠️  Error: {e}")

# 5. Recomendaciones
print("\n" + "="*80)
print("💡 RECOMENDACIONES")
print("="*80)

has_nvidia_driver = False
has_torch = False
has_cuda_torch = False

try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    has_nvidia_driver = result.returncode == 0
except:
    pass

try:
    import torch
    has_torch = True
    has_cuda_torch = torch.cuda.is_available()
except:
    pass

if not has_nvidia_driver:
    print("\n❌ NO SE DETECTÓ GPU NVIDIA")
    print("\n   Pasos a seguir:")
    print("   1. Verifica que tengas una GPU NVIDIA")
    print("   2. Descarga e instala los drivers desde:")
    print("      https://www.nvidia.com/Download/index.aspx")
    print("   3. Reinicia tu PC")

elif not has_torch:
    print("\n❌ PYTORCH NO INSTALADO")
    print("\n   Instala PyTorch con CUDA ejecutando:")
    print("\n   En CMD o PowerShell (dentro de tu entorno virtual):")
    print("   " + "-"*60)
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("   " + "-"*60)

elif not has_cuda_torch:
    print("\n❌ PYTORCH INSTALADO PERO SIN SOPORTE CUDA")
    print("\n   Tu PyTorch actual es la versión CPU-only")
    print("   Para reinstalar con soporte CUDA:")
    print("\n   1. Desinstala PyTorch actual:")
    print("   " + "-"*60)
    print("   pip uninstall torch torchvision torchaudio")
    print("   " + "-"*60)
    print("\n   2. Instala PyTorch con CUDA 12.1:")
    print("   " + "-"*60)
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("   " + "-"*60)
    print("\n   O si tienes CUDA 11.8:")
    print("   " + "-"*60)
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("   " + "-"*60)

else:
    print("\n✅ TODO CONFIGURADO CORRECTAMENTE")
    print("   PyTorch con CUDA está funcionando")
    print("   Puedes usar tournament_parallel_CUDA.py sin problemas")

print("\n" + "="*80)
print("📚 RECURSOS ADICIONALES")
print("="*80)
print("\n   • Instalador oficial de PyTorch:")
print("     https://pytorch.org/get-started/locally/")
print("\n   • Drivers NVIDIA:")
print("     https://www.nvidia.com/Download/index.aspx")
print("\n   • Guía de instalación CUDA:")
print("     https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/")

print("\n" + "="*80)
print("Diagnóstico completado")
print("="*80 + "\n")

