from IPython.display import clear_output
import time

# Lista de tareas con su estado inicial
tasks = [
    {"desc": "Saber dónde se guardan los modelos entrenados (agentes) 📂", "done": False},
    {"desc": "Crear una copia de trainRL.py para modificaciones 📄", "done": False},
    {"desc": "Integrar las gráficas para verificar el entrenamiento 📊", "done": False},
    {"desc": "Variar los parámetros de aprendizaje cada 100 épocas ⚙️", "done": False},
    {"desc": "Crear torneo.py para competir entre agentes 🏆", "done": False},
    {"desc": "Crear torneo_paralelo.py para usar todos los núcleos 🚀", "done": False},
    {"desc": "Identificar archivos a desechar después de cada torneo 🗑️", "done": False},
    {"desc": "Evidenciar los resultados obtenidos en este desarrollo ✅", "done": False},
    {"desc": "Iniciar nuevo entrenamiento con los 3 mejores agentes 🥇", "done": False}
]


def show_checklist():
    """Función para mostrar la lista de tareas actualizada."""
    clear_output(wait=True)
    print("📋 Checklist Interactivo 📋")
    print("Ingresa el número de la tarea para cambiar su estado (o 's' para salir).")
    print("-" * 60)
    for i, task in enumerate(tasks):
        status_icon = "✅" if task["done"] else "🔳"
        print(f"{i + 1}. {status_icon} {task['desc']}")
    print("-" * 60)


# Bucle principal para la interacción
while True:
    show_checklist()
    user_input = input("Tu elección: ")

    if user_input.lower() == 's':
        break

    try:
        choice_index = int(user_input) - 1
        if 0 <= choice_index < len(tasks):
            # Cambia el estado de la tarea (de True a False o viceversa)
            tasks[choice_index]["done"] = not tasks[choice_index]["done"]
        else:
            print("Número fuera de rango. Inténtalo de nuevo.")
            time.sleep(1)
    except ValueError:
        print("Entrada no válida. Por favor, ingresa un número o 's'.")
        time.sleep(1)

# Imprimir el resumen final
clear_output(wait=True)
print("--- Resumen Final del Checklist ---")
for task in tasks:
    status_text = "Hecho" if task["done"] else "No Hecho"
    print(f"- {task['desc']}: {status_text}")
