# -*- coding: utf-8 -*-

"""
FlexibleCNN - A flexible CNN model for the Quarto board game that can adapt to different architectures.
"""

"""
Python 3
18 / 10 / 2025
@author: Kilo Code

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import logging

# Configurar logger (moverlo antes de cualquier uso en funciones de compatibilidad)
logger = logging.getLogger("FlexibleCNN")
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s][%(levelname)s][%(name)s] %(message)s', datefmt='%m-%d %H:%M:%S')
handler.setFormatter(formatter)
# Evitar múltiples handlers si el módulo se importa varias veces
if not logger.handlers:
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Compat loader for torch.load across PyTorch versions
def torch_load_compat(weights_path, map_location=torch.device('cpu')):
    """
    Compatibilidad para cargar checkpoints con distintas versiones de PyTorch.
    Intenta en este orden:
      1) torch.load() por defecto (PyTorch >=2.6 usa weights_only=True por defecto)
      2) Reintentar dentro de torch.serialization.safe_globals([numpy._core.multiarray.scalar])
      3) Como último recurso, reintentar con weights_only=False (menos seguro)
    """
    try:
        return torch.load(weights_path, map_location=map_location)
    except Exception as e:
        logger.warning(f"torch.load failed (default). Error: {e}")
        # Intentar safe_globals (disponible en PyTorch 2.6+)
        try:
            _np = np
            safe_globals_ctx = getattr(torch.serialization, 'safe_globals', None)
            add_safe_globals = getattr(torch.serialization, 'add_safe_globals', None)

            if safe_globals_ctx is not None:
                logger.info("Retrying torch.load inside torch.serialization.safe_globals for numpy._core.multiarray.scalar")
                with torch.serialization.safe_globals([_np._core.multiarray.scalar]):
                    return torch.load(weights_path, map_location=map_location)
            elif add_safe_globals is not None:
                logger.info("Registering numpy._core.multiarray.scalar globally via add_safe_globals and retrying")
                torch.serialization.add_safe_globals([_np._core.multiarray.scalar])
                return torch.load(weights_path, map_location=map_location)
        except Exception as e2:
            logger.warning(f"safe_globals attempt failed: {e2}")

        # Último recurso: intentar weights_only=False (solo si confías en el checkpoint)
        try:
            logger.warning("Retrying torch.load with weights_only=False (less safe). Only do this if checkpoint is trusted.")
            return torch.load(weights_path, map_location=map_location, weights_only=False)
        except TypeError:
            # Este parámetro no existe en versiones antiguas de torch
            logger.error("torch.load does not support weights_only parameter in this torch version; re-raising original exception.")
            raise e
        except Exception as e3:
            logger.error(f"Final attempt to load model failed: {e3}")
            raise


class FlexibleNN(nn.Module):
    """
    Base class for flexible neural networks that can adapt to different architectures.
    """
    @property
    def name(self) -> str:
        return "FlexibleNN"
    
    @classmethod
    def from_file(cls, weights_path: str, strict: bool = False):
        """
        Load the model from a file with flexible architecture adaptation.
        
        Args:
            weights_path: Path to the weights file.
            strict: Whether to strictly enforce that the keys in state_dict match.
                   Default: False for flexibility.
        
        Returns:
            Model instance with loaded weights.
        """
        model = cls()
        
        try:
            # Intentar cargar el estado del modelo
-            state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
+            state_dict = torch_load_compat(weights_path, map_location=torch.device('cpu'))

            # Verificar si hay diferencias en las claves del estado
            model_keys = set(model.state_dict().keys())
            loaded_keys = set(state_dict.keys())
            
            if model_keys != loaded_keys:
                logger.warning("Model architecture differs from checkpoint")
                logger.info(f"Missing keys: {model_keys - loaded_keys}")
                logger.info(f"Unexpected keys: {loaded_keys - model_keys}")
                
                # Intentar cargar con strict=False
                model.load_state_dict(state_dict, strict=False)
                logger.info("Model loaded with partial weights (strict=False)")
            else:
                # Si las claves coinciden, cargar normalmente
                model.load_state_dict(state_dict, strict=strict)
                logger.info("Model loaded successfully with exact architecture match")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
        return model
    
    def export_model(self, checkpoint_suffix: str, checkpoint_folder: str = "weights/") -> str:
        """
        Export the model to a file with the datetime and suffix in the filename.
        
        Args:
            checkpoint_suffix: Suffix for the checkpoint file name.
            checkpoint_folder: Folder to save the model weights.
        
        Returns:
            The full path to the saved model file.
        """
        from datetime import datetime
        import os
        
        # Crear el nombre del checkpoint
        checkpoint_name = f"{datetime.now().strftime('%Y%m%d_%H%M')}-{checkpoint_suffix}.pt"
        
        # Asegurar que el directorio existe
        os.makedirs(os.path.join(checkpoint_folder, self.name), exist_ok=True)
        
        # Ruta completa del archivo
        file_path = os.path.join(checkpoint_folder, self.name, checkpoint_name)
        
        # Guardar el modelo
        torch.save(self.state_dict(), file_path)
        
        return file_path

class FlexibleQuartoCNN(FlexibleNN):
    """
    A flexible CNN model for the Quarto board game that can adapt to different architectures.
    """
    @property
    def name(self) -> str:
        return "FlexibleQuartoCNN"
    
    def __init__(self, fc_inpiece_size=16, k1_size=16, k2_size=32, n_neurons=128, fc2_piece_input=None):
        """
        Initialize the flexible CNN model with configurable parameters.
        
        Args:
            fc_inpiece_size: Size of the piece input layer (must be multiple of 16)
            k1_size: Size of the first convolutional layer
            k2_size: Size of the second convolutional layer
            n_neurons: Size of the fully connected layer
            fc2_piece_input: Optional override for fc2_piece input size (for compatibility)
        """
        super().__init__()
        
        # Validar parámetros
        assert fc_inpiece_size % 16 == 0, "fc_inpiece_size must be a multiple of 16"
        
        # Capa de entrada para características de pieza
        self.fc_in_piece = nn.Linear(16, fc_inpiece_size)
        
        # Capas convolucionales
        self.conv1 = nn.Conv2d(16 + fc_inpiece_size // 16, k1_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(k1_size, k2_size, kernel_size=3, padding=1)
        
        # Capa completamente conectada
        self.fc1 = nn.Linear(k2_size * 4 * 4, n_neurons)
        
        # Cabezas de salida
        self.fc2_board = nn.Linear(n_neurons, 4 * 4)
        
        # Para fc2_piece, usar el tamaño de entrada especificado o el predeterminado
        if fc2_piece_input is not None:
            # Usar el tamaño de entrada especificado (para compatibilidad con modelos existentes)
            self.fc2_piece = nn.Linear(fc2_piece_input, 4 * 4)
            logger.info(f"Using custom fc2_piece input size: {fc2_piece_input}")
        else:
            # Usar el mismo tamaño que fc1 (comportamiento predeterminado)
            self.fc2_piece = nn.Linear(n_neurons, 4 * 4)
            
        # Guardar el tamaño de entrada esperado para fc2_piece
        self.fc2_piece_input_size = self.fc2_piece.in_features
        
        # Dropout para regularización
        self.dropout = nn.Dropout(0.5)
        
        # Guardar dimensiones para referencia
        self.dimensions = {
            'fc_inpiece_size': fc_inpiece_size,
            'k1_size': k1_size,
            'k2_size': k2_size,
            'n_neurons': n_neurons
        }
        
        logger.info(f"Initialized FlexibleQuartoCNN with dimensions: {self.dimensions}")
    
    def forward(self, x_board: torch.Tensor, x_piece: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            x_board: Input tensor of the board with placed pieces (batch_size, 16, 4, 4).
            x_piece: Input tensor of selected piece to place (batch_size, 16).
            
        Returns:
            qav_board: Action value for the board position (batch_size, 16).
            qav_piece: Action value for the selected piece (batch_size, 16).
        """
        # Procesar la pieza
        piece_feat = F.relu(self.fc_in_piece(x_piece))
        piece_map = piece_feat.view(-1, piece_feat.size(1) // 16, 4, 4)
        
        # Concatenar con el tablero
        x = torch.cat([x_board, piece_map], dim=1)
        
        # Pasar por las capas convolucionales
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Aplanar y pasar por la capa completamente conectada
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Calcular salidas
        logits_board = self.fc2_board(x)
        qav_board = F.tanh(logits_board)
        
        # Para fc2_piece, manejar el caso especial donde la entrada podría tener un tamaño diferente
        if x.size(1) != self.fc2_piece_input_size:
            # Si las dimensiones no coinciden, adaptar la entrada silenciosamente
            if self.fc2_piece_input_size > x.size(1):
                # Expandir
                x_resized = torch.zeros((x.size(0), self.fc2_piece_input_size), device=x.device)
                x_resized[:, :x.size(1)] = x
            else:
                # Reducir
                x_resized = x[:, :self.fc2_piece_input_size]
            
            logits_piece = self.fc2_piece(x_resized)
        else:
            # Caso normal
            logits_piece = self.fc2_piece(x)
            
        qav_piece = F.tanh(logits_piece)
        
        return qav_board, qav_piece
    
    def predict(self, x_board: torch.Tensor, x_piece: torch.Tensor, 
                TEMPERATURE: float = 1.0, DETERMINISTIC: bool = True):
        """
        Predicts the preferred order of board positions and pieces.
        
        Args:
            x_board: Input tensor of shape (batch_size, 16, 4, 4).
            x_piece: Input tensor of shape (batch_size, 16).
            TEMPERATURE: Sampling temperature (>0). Lower values make predictions more deterministic.
            DETERMINISTIC: If True, use argmax instead of sampling.
            
        Returns:
            board_indices: Predicted board position indices.
            piece_indices: Predicted piece indices.
        """
        # Validar entradas
        assert x_board.shape[1:] == (16, 4, 4), "Input tensor must have shape (batch_size, 16, 4, 4)"
        assert x_piece.shape[1] == 16, "Input tensor must have shape (batch_size, 16)"
        assert x_board.shape[0] == x_piece.shape[0], "Input tensors must have the same batch size"
        
        # Modo evaluación
        self.eval()
        
        with torch.no_grad():
            # Obtener valores de acción
            qav_board, qav_piece = self.forward(x_board, x_piece)
            
            if DETERMINISTIC:
                # Ordenar por valor de acción (mayor a menor)
                board_indices = torch.argsort(qav_board, descending=True, dim=1)
                piece_indices = torch.argsort(qav_piece, descending=True, dim=1)
            else:
                # Usar softmax con temperatura para muestreo estocástico
                board_probs = F.softmax(qav_board / TEMPERATURE, dim=1)
                piece_probs = F.softmax(qav_piece / TEMPERATURE, dim=1)
                
                # Muestrear sin reemplazo
                board_indices = torch.multinomial(board_probs, board_probs.shape[1], replacement=False)
                piece_indices = torch.multinomial(piece_probs, piece_probs.shape[1], replacement=False)
            
            return board_indices, piece_indices
    
    @classmethod
    def auto_detect_architecture(cls, weights_path: str):
        """
        Automatically detect the architecture from a weights file.
        
        Args:
            weights_path: Path to the weights file.
            
        Returns:
            A dictionary with the detected architecture parameters.
        """
        try:
            # Cargar el estado del modelo
-            state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
+            state_dict = torch_load_compat(weights_path, map_location=torch.device('cpu'))

            # Detectar dimensiones
            architecture = {}
            
            # Detectar fc_inpiece_size
            if 'fc_in_piece.weight' in state_dict:
                fc_inpiece_size = state_dict['fc_in_piece.weight'].shape[0]
                architecture['fc_inpiece_size'] = fc_inpiece_size
            
            # Detectar k1_size
            if 'conv1.weight' in state_dict:
                k1_size = state_dict['conv1.weight'].shape[0]
                architecture['k1_size'] = k1_size
            
            # Detectar k2_size
            if 'conv2.weight' in state_dict:
                k2_size = state_dict['conv2.weight'].shape[0]
                architecture['k2_size'] = k2_size
            
            # Detectar n_neurons
            if 'fc1.weight' in state_dict:
                n_neurons = state_dict['fc1.weight'].shape[0]
                architecture['n_neurons'] = n_neurons
            
            # Detectar fc2_piece_input (importante para el error específico)
            if 'fc2_piece.weight' in state_dict:
                fc2_piece_shape = state_dict['fc2_piece.weight'].shape
                if len(fc2_piece_shape) > 1:
                    fc2_piece_input = fc2_piece_shape[1]
                    architecture['fc2_piece_input'] = fc2_piece_input
                    logger.info(f"Detected fc2_piece input size: {fc2_piece_input}")
                    
                    # Si también detectamos n_neurons, verificar si coinciden
                    if 'n_neurons' in architecture and architecture['n_neurons'] != fc2_piece_input:
                        logger.info(f"Note: fc2_piece input size ({fc2_piece_input}) differs from n_neurons ({architecture['n_neurons']})")
            
            logger.info(f"Detected architecture: {architecture}")
            return architecture
            
        except Exception as e:
            logger.error(f"Error detecting architecture: {e}")
            return {}
    
    @classmethod
    def from_file(cls, weights_path: str, strict: bool = False):
        """
        Load the model from a file with automatic architecture detection.
        
        Args:
            weights_path: Path to the weights file.
            strict: Whether to strictly enforce that the keys in state_dict match.
                   Default: False for flexibility.
        
        Returns:
            Model instance with loaded weights.
        """
        # Detectar arquitectura
        architecture = cls.auto_detect_architecture(weights_path)
        
        # Crear modelo con la arquitectura detectada o usar valores predeterminados
        model = cls(
            fc_inpiece_size=architecture.get('fc_inpiece_size', 16),
            k1_size=architecture.get('k1_size', 16),
            k2_size=architecture.get('k2_size', 32),
            n_neurons=architecture.get('n_neurons', 128),
            fc2_piece_input=architecture.get('fc2_piece_input', None)
        )
        
        try:
            # Cargar el estado del modelo
-            state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
+            state_dict = torch_load_compat(weights_path, map_location=torch.device('cpu'))

            # Verificar si hay diferencias en las claves del estado
            model_keys = set(model.state_dict().keys())
            loaded_keys = set(state_dict.keys())
            
            if model_keys != loaded_keys:
                logger.info("Model architecture differs from checkpoint")
                logger.info(f"Missing keys: {model_keys - loaded_keys}")
                logger.info(f"Unexpected keys: {loaded_keys - model_keys}")
            
            # Intentar cargar con strict=False para permitir diferencias en la arquitectura
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Model loaded from {weights_path} with flexible architecture")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
        return model