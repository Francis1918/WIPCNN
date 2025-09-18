from torchrl.collectors import SyncDataCollector
from tensordict import TensorDict
import torch
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from torchrl.envs import EnvCreator, GymWrapper

# Ejemplo de entorno personalizado basado en Gymnasium
class CustomEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Definir el espacio de acción: 1 acción discreta con 4 opciones (0, 1, 2, 3)
        self.action_space = spaces.Discrete(4)

        # Definir el espacio de observación: vector de 3 elementos con valores entre -1 y 1
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # Estado inicial
        self.state = np.zeros(3, dtype=np.float32)
        self.steps = 0
        self.max_steps = 100

        # Metadatos requeridos por Gymnasium
        self.metadata = {"render_modes": ["human"]}

    def reset(self, *, seed=None, options=None):
        # Reiniciar el entorno al inicio de un episodio
        super().reset(seed=seed)
        self.state = np.random.uniform(-0.1, 0.1, size=3).astype(np.float32)
        self.steps = 0
        return self.state, {}  # Retornar observación e info (dict vacío)

    def step(self, action):
        # Ejecutar un paso con la acción proporcionada
        assert self.action_space.contains(action), f"Acción {action} no válida"

        # Actualizar el estado según la acción
        if action == 0:
            self.state[0] += 0.1
        elif action == 1:
            self.state[0] -= 0.1
        elif action == 2:
            self.state[1] += 0.1
        else:  # action == 3
            self.state[1] -= 0.1

        # Añadir algo de ruido
        self.state += np.random.uniform(-0.05, 0.05, size=3).astype(np.float32)
        # Limitar los valores entre -1 y 1
        self.state = np.clip(self.state, -1, 1)

        # Calcular recompensa (acercarse al objetivo [0.5, 0.5, 0])
        target = np.array([0.5, 0.5, 0], dtype=np.float32)
        distance = np.linalg.norm(self.state - target)
        reward = -distance  # Recompensa negativa por distancia

        # Incrementar contador de pasos
        self.steps += 1

        # Determinar si el episodio ha terminado
        terminated = False
        truncated = False

        if distance < 0.1:  # Éxito si estamos cerca del objetivo
            terminated = True
            reward += 10  # Bonificación por éxito
        elif self.steps >= self.max_steps:  # Tiempo agotado
            truncated = True

        # Datos personalizados para este paso
        info = {"custom_data": np.array([self.steps, distance, float(terminated)], dtype=np.float32)}

        return self.state, reward, terminated, truncated, info

# Función para crear y devolver el entorno compatible con torchrl
def your_env_fn():
    """Función que crea y devuelve una instancia del entorno personalizado"""
    # Utilizar GymWrapper directamente con la instancia del entorno
    env = CustomEnv()
    return GymWrapper(env)

# Política simple que toma acciones aleatorias
def your_policy(tensordict):
    """Política que selecciona acciones aleatorias"""
    # Esta política simplemente elige acciones aleatorias (0, 1, 2 o 3)
    batch_size = tensordict.batch_size

    if len(batch_size) == 0:
        # Sin batch, solo una muestra
        action = torch.randint(0, 4, (1,))
    else:
        # Con batch, genera una acción para cada elemento
        action = torch.randint(0, 4, batch_size)

    return TensorDict({"action": action}, batch_size=batch_size)

# Ejemplo de uso del colector
if __name__ == "__main__":
    print("Iniciando colector de datos...")

    # Crear el colector con el entorno y la política definidos
    collector = SyncDataCollector(
        create_env_fn=your_env_fn,  # Función que crea el entorno
        policy=your_policy,         # Política para seleccionar acciones
        total_frames=1000,          # Total de frames a recolectar
        frames_per_batch=100,       # Frames por lote
    )

    print("Colector creado. Recolectando datos...")

    # Iterar sobre los lotes de datos recolectados
    for i, batch in enumerate(collector):
        print(f"\nLote {i+1}:")
        print(f"Forma del lote: {batch.batch_size}")
        print(f"Claves disponibles: {batch.keys()}")

        # Mostrar algunas estadísticas del lote
        if "next" in batch and "info" in batch["next"] and "custom_data" in batch["next"]["info"]:
            custom_data = batch["next"]["info"]["custom_data"]
            print(f"Datos personalizados (primeras 5 muestras):\n{custom_data[:5]}")

        # Mostrar promedio de recompensas
        if "reward" in batch:
            rewards = batch["reward"]
            print(f"Recompensa promedio: {rewards.mean().item():.4f}")

        # Mostrar acciones tomadas
        if "action" in batch:
            actions = batch["action"]
            action_counts = torch.bincount(actions.flatten(), minlength=4)
            print(f"Distribución de acciones: {action_counts}")

        # Parar después de 3 lotes para este ejemplo
        if i >= 2:
            break

    print("\nRecolección completada.")
    collector.shutdown()
    print("Colector cerrado correctamente.")
