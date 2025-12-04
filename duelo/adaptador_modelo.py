"""
Adaptador para cargar modelos CNN de diferentes arquitecturas.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple


class QuartoCNNExtended(nn.Module):
    """
    Arquitectura CNN extendida para Quarto con BatchNorm y capas adicionales.
    Soporta: fc_in_piece, conv1, bn1, conv2, bn2, fc1, bn_fc1, fc1b, bn_fc1b,
             fc1c, bn_fc1c, fc1d, bn_fc1d, fc2_board, fc2_piece
    """

    def __init__(self, state_dict: dict):
        super().__init__()
        self._crear_capas_desde_state_dict(state_dict)

    def _crear_capas_desde_state_dict(self, state_dict: dict):
        """Crea todas las capas basándose en el state_dict."""

        # fc_in_piece
        if 'fc_in_piece.weight' in state_dict:
            w = state_dict['fc_in_piece.weight']
            self.fc_in_piece = nn.Linear(w.shape[1], w.shape[0])
        else:
            self.fc_in_piece = nn.Linear(16, 16)

        # conv1
        if 'conv1.weight' in state_dict:
            w = state_dict['conv1.weight']
            self.conv1 = nn.Conv2d(w.shape[1], w.shape[0], kernel_size=w.shape[2], padding=w.shape[2]//2)
            conv1_out = w.shape[0]
        else:
            self.conv1 = nn.Conv2d(17, 16, kernel_size=3, padding=1)
            conv1_out = 16

        # bn1 (BatchNorm después de conv1)
        if 'bn1.weight' in state_dict:
            self.bn1 = nn.BatchNorm2d(conv1_out)
            self.has_bn1 = True
        else:
            self.has_bn1 = False

        # conv2
        if 'conv2.weight' in state_dict:
            w = state_dict['conv2.weight']
            self.conv2 = nn.Conv2d(w.shape[1], w.shape[0], kernel_size=w.shape[2], padding=w.shape[2]//2)
            conv2_out = w.shape[0]
        else:
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            conv2_out = 32

        # bn2 (BatchNorm después de conv2)
        if 'bn2.weight' in state_dict:
            self.bn2 = nn.BatchNorm2d(conv2_out)
            self.has_bn2 = True
        else:
            self.has_bn2 = False

        # fc1
        if 'fc1.weight' in state_dict:
            w = state_dict['fc1.weight']
            self.fc1 = nn.Linear(w.shape[1], w.shape[0])
            fc1_out = w.shape[0]
        else:
            self.fc1 = nn.Linear(conv2_out * 16, 256)
            fc1_out = 256

        # bn_fc1
        if 'bn_fc1.weight' in state_dict:
            self.bn_fc1 = nn.BatchNorm1d(fc1_out)
            self.has_bn_fc1 = True
        else:
            self.has_bn_fc1 = False

        # fc1b (capa adicional)
        if 'fc1b.weight' in state_dict:
            w = state_dict['fc1b.weight']
            self.fc1b = nn.Linear(w.shape[1], w.shape[0])
            fc1b_out = w.shape[0]
            self.has_fc1b = True
        else:
            self.has_fc1b = False
            fc1b_out = fc1_out

        # bn_fc1b
        if 'bn_fc1b.weight' in state_dict:
            self.bn_fc1b = nn.BatchNorm1d(fc1b_out)
            self.has_bn_fc1b = True
        else:
            self.has_bn_fc1b = False

        # fc1c
        if 'fc1c.weight' in state_dict:
            w = state_dict['fc1c.weight']
            self.fc1c = nn.Linear(w.shape[1], w.shape[0])
            fc1c_out = w.shape[0]
            self.has_fc1c = True
        else:
            self.has_fc1c = False
            fc1c_out = fc1b_out

        # bn_fc1c
        if 'bn_fc1c.weight' in state_dict:
            self.bn_fc1c = nn.BatchNorm1d(fc1c_out)
            self.has_bn_fc1c = True
        else:
            self.has_bn_fc1c = False

        # fc1d
        if 'fc1d.weight' in state_dict:
            w = state_dict['fc1d.weight']
            self.fc1d = nn.Linear(w.shape[1], w.shape[0])
            fc1d_out = w.shape[0]
            self.has_fc1d = True
        else:
            self.has_fc1d = False
            fc1d_out = fc1c_out

        # bn_fc1d
        if 'bn_fc1d.weight' in state_dict:
            self.bn_fc1d = nn.BatchNorm1d(fc1d_out)
            self.has_bn_fc1d = True
        else:
            self.has_bn_fc1d = False

        # fc2_board
        if 'fc2_board.weight' in state_dict:
            w = state_dict['fc2_board.weight']
            self.fc2_board = nn.Linear(w.shape[1], w.shape[0])
        else:
            self.fc2_board = nn.Linear(fc1d_out, 16)

        # fc2_piece
        if 'fc2_piece.weight' in state_dict:
            w = state_dict['fc2_piece.weight']
            self.fc2_piece = nn.Linear(w.shape[1], w.shape[0])
        else:
            self.fc2_piece = nn.Linear(fc1d_out, 16)

        # Dropout
        self.dropout = nn.Dropout(0.2)

    def forward(self, x_board, x_piece=None):
        """Forward pass con arquitectura extendida."""
        device = next(self.parameters()).device
        batch_size = 1

        # Preparar tablero como 16 canales one-hot
        if isinstance(x_board, torch.Tensor) and x_board.dim() == 4 and x_board.shape[1] == 16:
            x_board = x_board.to(device)
            batch_size = x_board.shape[0]
        else:
            if isinstance(x_board, torch.Tensor):
                tablero_flat = x_board.view(-1).tolist()
            elif isinstance(x_board, list):
                if isinstance(x_board[0], list):
                    tablero_flat = [item for row in x_board for item in row]
                else:
                    tablero_flat = x_board
            else:
                tablero_flat = list(x_board)

            x_board_tensor = torch.zeros(1, 16, 4, 4, device=device)
            for pos in range(16):
                pieza_en_pos = int(tablero_flat[pos])
                if 0 <= pieza_en_pos < 16:
                    row = pos // 4
                    col = pos % 4
                    x_board_tensor[0, pieza_en_pos, row, col] = 1.0
            x_board = x_board_tensor

        # Preparar pieza como one-hot
        if x_piece is None:
            x_piece_oh = torch.zeros(batch_size, 16, device=device)
        elif isinstance(x_piece, int):
            x_piece_oh = torch.zeros(batch_size, 16, device=device)
            if 0 <= x_piece < 16:
                x_piece_oh[:, x_piece] = 1.0
        elif isinstance(x_piece, torch.Tensor):
            if x_piece.dim() == 0:
                x_piece_oh = torch.zeros(batch_size, 16, device=device)
                idx = int(x_piece.item())
                if 0 <= idx < 16:
                    x_piece_oh[:, idx] = 1.0
            elif x_piece.shape[-1] == 16:
                x_piece_oh = x_piece.float().to(device)
                if x_piece_oh.dim() == 1:
                    x_piece_oh = x_piece_oh.unsqueeze(0)
            else:
                x_piece_oh = torch.zeros(batch_size, 16, device=device)
                idx = int(x_piece.view(-1)[0].item())
                if 0 <= idx < 16:
                    x_piece_oh[:, idx] = 1.0
        else:
            x_piece_oh = torch.zeros(batch_size, 16, device=device)

        # Procesar pieza
        piece_feat = F.relu(self.fc_in_piece(x_piece_oh))
        piece_map = piece_feat.view(batch_size, 1, 4, 4)

        # Concatenar tablero + piece_map
        x = torch.cat([x_board, piece_map], dim=1)

        # Conv1 + BN1
        x = self.conv1(x)
        if self.has_bn1:
            x = self.bn1(x)
        x = F.relu(x)

        # Conv2 + BN2
        x = self.conv2(x)
        if self.has_bn2:
            x = self.bn2(x)
        x = F.relu(x)

        # Flatten
        x = x.flatten(start_dim=1)

        # FC1 + BN_FC1
        x = self.fc1(x)
        if self.has_bn_fc1:
            x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # FC1B + BN_FC1B
        if self.has_fc1b:
            x = self.fc1b(x)
            if self.has_bn_fc1b:
                x = self.bn_fc1b(x)
            x = F.relu(x)
            x = self.dropout(x)

        # FC1C + BN_FC1C
        if self.has_fc1c:
            x = self.fc1c(x)
            if self.has_bn_fc1c:
                x = self.bn_fc1c(x)
            x = F.relu(x)
            x = self.dropout(x)

        # FC1D + BN_FC1D
        if self.has_fc1d:
            x = self.fc1d(x)
            if self.has_bn_fc1d:
                x = self.bn_fc1d(x)
            x = F.relu(x)

        x = self.dropout(x)

        # Salidas
        logits_board = self.fc2_board(x)
        logits_piece = self.fc2_piece(x)

        return logits_board, logits_piece


class QuartoCNN(nn.Module):
    """
    Arquitectura CNN específica para Quarto con salidas duales (tablero y pieza).
    Soporta las capas: fc_in_piece, conv1, conv2, fc1, fc2_board, fc2_piece

    La arquitectura real espera 17 canales de entrada:
    - 16 canales binarios (uno por cada pieza posible en cada casilla)
    - 1 canal adicional (posiblemente para pieza actual o máscara)
    """

    def __init__(self, state_dict: dict):
        super().__init__()

        # Inferir dimensiones desde state_dict
        self._crear_capas_desde_state_dict(state_dict)
        self._detectar_formato_entrada(state_dict)

    def _detectar_formato_entrada(self, state_dict: dict):
        """Detecta el formato de entrada esperado basado en conv1."""
        if 'conv1.weight' in state_dict:
            w = state_dict['conv1.weight']
            self.in_channels = w.shape[1]  # Número de canales de entrada
        else:
            self.in_channels = 17  # Default basado en el error

    def _crear_capas_desde_state_dict(self, state_dict: dict):
        """Crea las capas con las dimensiones correctas basándose en el state_dict."""

        # fc_in_piece: procesa la pieza actual
        if 'fc_in_piece.weight' in state_dict:
            w = state_dict['fc_in_piece.weight']
            self.fc_in_piece = nn.Linear(w.shape[1], w.shape[0])
            self.piece_features = w.shape[0]
        else:
            self.fc_in_piece = nn.Linear(16, 16)
            self.piece_features = 16

        # conv1: primera capa convolucional
        if 'conv1.weight' in state_dict:
            w = state_dict['conv1.weight']
            self.conv1 = nn.Conv2d(w.shape[1], w.shape[0], kernel_size=w.shape[2], padding=w.shape[2]//2)
            self.conv1_out = w.shape[0]
        else:
            self.conv1 = nn.Conv2d(17, 16, kernel_size=3, padding=1)
            self.conv1_out = 16

        # conv2: segunda capa convolucional
        if 'conv2.weight' in state_dict:
            w = state_dict['conv2.weight']
            self.conv2 = nn.Conv2d(w.shape[1], w.shape[0], kernel_size=w.shape[2], padding=w.shape[2]//2)
            self.conv2_out = w.shape[0]
        else:
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.conv2_out = 32

        # fc1: capa fully connected intermedia
        if 'fc1.weight' in state_dict:
            w = state_dict['fc1.weight']
            self.fc1 = nn.Linear(w.shape[1], w.shape[0])
            self.fc1_out = w.shape[0]
        else:
            self.fc1 = nn.Linear(32 * 4 * 4 + 16, 64)
            self.fc1_out = 64

        # fc2_board: salida para posición del tablero
        if 'fc2_board.weight' in state_dict:
            w = state_dict['fc2_board.weight']
            self.fc2_board = nn.Linear(w.shape[1], w.shape[0])
            self.fc2_board_in = w.shape[1]
        else:
            self.fc2_board = nn.Linear(64, 16)
            self.fc2_board_in = 64

        # fc2_piece: salida para selección de pieza
        if 'fc2_piece.weight' in state_dict:
            w = state_dict['fc2_piece.weight']
            self.fc2_piece = nn.Linear(w.shape[1], w.shape[0])
            self.fc2_piece_in = w.shape[1]
        else:
            self.fc2_piece = nn.Linear(64, 16)
            self.fc2_piece_in = 64

        # Detectar si fc2_piece necesita features adicionales de pieza
        # Si fc2_piece_in > fc2_board_in, significa que recibe pieza + fc1_out
        self.piece_head_needs_extra = (self.fc2_piece_in > self.fc2_board_in)

    def _preparar_entrada_17_canales(self, tablero, pieza_actual, device):
        """
        Prepara la entrada con 17 canales.
        - Canales 0-15: one-hot encoding de cada pieza en el tablero
        - Canal 16: máscara de casillas ocupadas o pieza actual
        """
        batch_size = 1

        # Convertir tablero a lista plana
        if isinstance(tablero, torch.Tensor):
            tablero_flat = tablero.view(-1).tolist()
        elif isinstance(tablero, list):
            if isinstance(tablero[0], list):
                tablero_flat = [item for row in tablero for item in row]
            else:
                tablero_flat = tablero
        else:
            tablero_flat = list(tablero)

        # Crear tensor de 17 canales
        x = torch.zeros(batch_size, 17, 4, 4, device=device)

        # Canales 0-15: one-hot encoding de piezas
        for pos in range(16):
            pieza = tablero_flat[pos]
            if pieza >= 0 and pieza < 16:
                row = pos // 4
                col = pos % 4
                x[0, int(pieza), row, col] = 1.0

        # Canal 16: máscara de casillas ocupadas
        for pos in range(16):
            row = pos // 4
            col = pos % 4
            if tablero_flat[pos] >= 0:
                x[0, 16, row, col] = 1.0

        return x

    def forward(self, x, pieza=None):
        """
        Forward pass siguiendo la arquitectura original:
        1. piece_feat = fc_in_piece(x_piece) -> [batch, 16]
        2. piece_map = piece_feat.view(-1, 1, 4, 4) -> [batch, 1, 4, 4]
        3. x = cat([x_board, piece_map]) -> [batch, 17, 4, 4]
        4. conv1, conv2, flatten, fc1, fc2_board, fc2_piece

        Args:
            x: Estado del tablero - lista/tensor del tablero
            pieza: Pieza actual (índice 0-15 o None)

        Returns:
            Tuple (logits_tablero, logits_pieza) cada uno de forma [batch, 16]
        """
        device = next(self.parameters()).device
        batch_size = 1

        # Preparar tablero como 16 canales one-hot
        if isinstance(x, torch.Tensor) and x.dim() == 4 and x.shape[1] == 16:
            x_board = x.to(device)
            batch_size = x_board.shape[0]
        else:
            # Convertir tablero a tensor de 16 canales
            if isinstance(x, torch.Tensor):
                tablero_flat = x.view(-1).tolist()
            elif isinstance(x, list):
                if isinstance(x[0], list):
                    tablero_flat = [item for row in x for item in row]
                else:
                    tablero_flat = x
            else:
                tablero_flat = list(x)

            x_board = torch.zeros(1, 16, 4, 4, device=device)
            for pos in range(16):
                pieza_en_pos = int(tablero_flat[pos])
                if 0 <= pieza_en_pos < 16:
                    row = pos // 4
                    col = pos % 4
                    x_board[0, pieza_en_pos, row, col] = 1.0

        # Preparar pieza como one-hot
        if pieza is None:
            x_piece = torch.zeros(batch_size, 16, device=device)
        elif isinstance(pieza, int):
            x_piece = torch.zeros(batch_size, 16, device=device)
            if 0 <= pieza < 16:
                x_piece[:, pieza] = 1.0
        elif isinstance(pieza, torch.Tensor):
            if pieza.dim() == 0:
                x_piece = torch.zeros(batch_size, 16, device=device)
                idx = int(pieza.item())
                if 0 <= idx < 16:
                    x_piece[:, idx] = 1.0
            elif pieza.shape[-1] == 16:
                x_piece = pieza.float().to(device)
                if x_piece.dim() == 1:
                    x_piece = x_piece.unsqueeze(0)
            else:
                x_piece = torch.zeros(batch_size, 16, device=device)
                idx = int(pieza.view(-1)[0].item())
                if 0 <= idx < 16:
                    x_piece[:, idx] = 1.0
        else:
            x_piece = torch.zeros(batch_size, 16, device=device)

        # 1. Procesar pieza con fc_in_piece
        piece_feat = F.relu(self.fc_in_piece(x_piece))  # [batch, 16]

        # 2. Reshape a mapa 4x4
        piece_map = piece_feat.view(batch_size, 1, 4, 4)  # [batch, 1, 4, 4]

        # 3. Concatenar tablero (16 canales) + piece_map (1 canal) = 17 canales
        x = torch.cat([x_board, piece_map], dim=1)  # [batch, 17, 4, 4]

        # 4. Convoluciones
        x = F.relu(self.conv1(x))  # [batch, 16, 4, 4]
        x = F.relu(self.conv2(x))  # [batch, 32, 4, 4]

        # 5. Flatten
        x = x.flatten(start_dim=1)  # [batch, 512]

        # 6. Fully connected
        x = F.relu(self.fc1(x))  # [batch, 128]

        # 7. Salidas duales
        logits_board = self.fc2_board(x)  # [batch, 16]

        # Para fc2_piece, algunas arquitecturas concatenan pieza original
        if self.piece_head_needs_extra:
            # Concatenar x con x_piece para fc2_piece
            x_for_piece = torch.cat([x, x_piece], dim=1)  # [batch, 128+16=144]
            logits_piece = self.fc2_piece(x_for_piece)
        else:
            logits_piece = self.fc2_piece(x)  # [batch, 16]

        return logits_board, logits_piece


class RedGenerica(nn.Module):
    """Wrapper genérico para cualquier red CNN cargada."""

    def __init__(self, modelo_base: nn.Module):
        super().__init__()
        self.modelo_base = modelo_base
        self._detectar_arquitectura()

    def _detectar_arquitectura(self):
        """Detecta información de la arquitectura."""
        self.info = {
            'capas': [],
            'parametros_totales': 0,
            'tipo': type(self.modelo_base).__name__
        }

        for nombre, modulo in self.modelo_base.named_modules():
            if isinstance(modulo, (nn.Conv2d, nn.Linear)):
                self.info['capas'].append({
                    'nombre': nombre,
                    'tipo': type(modulo).__name__,
                    'params': sum(p.numel() for p in modulo.parameters())
                })

        self.info['parametros_totales'] = sum(
            p.numel() for p in self.modelo_base.parameters()
        )

    def forward(self, x, pieza=None):
        # Si el modelo base soporta pieza como argumento
        if isinstance(self.modelo_base, (QuartoCNN, QuartoCNNExtended)):
            return self.modelo_base(x, pieza)
        else:
            return self.modelo_base(x)


class AdaptadorModelo:
    """Adapta diferentes arquitecturas CNN para el juego de Quarto."""

    TAMANIO_TABLERO = 4
    CASILLAS = 16
    PIEZAS = 16

    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

    def cargar_modelo(self, ruta: Path) -> Tuple[nn.Module, dict]:
        """
        Carga un modelo desde archivo .pt o .pth.
        Intenta múltiples estrategias de carga.
        """
        ruta = Path(ruta)

        if not ruta.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {ruta}")

        # Intentar cargar el checkpoint
        checkpoint = torch.load(ruta, map_location=self.device, weights_only=False)

        modelo = None
        metadata = {'ruta': str(ruta), 'nombre': ruta.stem}

        # Estrategia 1: Es un modelo completo
        if isinstance(checkpoint, nn.Module):
            modelo = checkpoint
            metadata['tipo_carga'] = 'modelo_completo'

        # Estrategia 2: Es un diccionario con el modelo
        elif isinstance(checkpoint, dict):
            metadata['claves_disponibles'] = list(checkpoint.keys())

            # Buscar el modelo en diferentes claves comunes
            claves_metadata = ['config', 'arquitectura', 'architecture', 'hparams', 'hyperparameters']

            # Extraer metadata si existe
            for clave in claves_metadata:
                if clave in checkpoint:
                    metadata[clave] = checkpoint[clave]

            # Intentar reconstruir el modelo
            modelo = self._reconstruir_desde_checkpoint(checkpoint, metadata)

        if modelo is None:
            raise ValueError(f"No se pudo cargar el modelo desde {ruta}")

        modelo = modelo.to(self.device)
        modelo.eval()

        # Wrap en RedGenerica para normalizar
        modelo_adaptado = RedGenerica(modelo)
        metadata['info_arquitectura'] = modelo_adaptado.info

        return modelo_adaptado, metadata

    def _reconstruir_desde_checkpoint(self, checkpoint: dict, metadata: dict) -> Optional[nn.Module]:
        """Intenta reconstruir el modelo desde un checkpoint."""

        # Obtener state_dict
        state_dict = None
        for clave in ['state_dict', 'model_state_dict', 'model']:
            if clave in checkpoint and isinstance(checkpoint[clave], dict):
                state_dict = checkpoint[clave]
                break

        if state_dict is None and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            # El checkpoint ES el state_dict
            state_dict = checkpoint

        if state_dict is None:
            return None

        # Detectar tipo de arquitectura por las claves del state_dict
        claves = set(state_dict.keys())

        # Claves básicas de QuartoCNN
        quarto_cnn_keys = {'fc_in_piece.weight', 'conv1.weight', 'conv2.weight',
                          'fc1.weight', 'fc2_board.weight', 'fc2_piece.weight'}

        # Claves de arquitectura extendida (con BatchNorm)
        extended_keys = {'bn1.weight', 'bn2.weight', 'bn_fc1.weight', 'fc1b.weight',
                        'fc1c.weight', 'fc1d.weight'}

        # Verificar si tiene claves extendidas (BatchNorm, capas adicionales)
        has_extended = bool(claves & extended_keys)

        if quarto_cnn_keys.issubset(claves):
            if has_extended:
                # Usar arquitectura extendida con BatchNorm
                try:
                    modelo = QuartoCNNExtended(state_dict)
                    modelo.load_state_dict(state_dict)
                    metadata['tipo_carga'] = 'QuartoCNNExtended'
                    return modelo
                except Exception as e:
                    print(f"Error cargando como QuartoCNNExtended: {e}")

            # Intentar arquitectura básica
            try:
                modelo = QuartoCNN(state_dict)
                modelo.load_state_dict(state_dict)
                metadata['tipo_carga'] = 'QuartoCNN'
                return modelo
            except Exception as e:
                print(f"Error cargando como QuartoCNN básico: {e}")

        # Si tiene arquitectura definida, intentar crearla
        if 'arquitectura' in checkpoint or 'architecture' in checkpoint:
            arch_info = checkpoint.get('arquitectura') or checkpoint.get('architecture')
            modelo = self._crear_desde_arquitectura(arch_info)

            if modelo is not None:
                modelo.load_state_dict(state_dict)
                metadata['tipo_carga'] = 'reconstruido_arquitectura'
                return modelo

        # Intentar inferir arquitectura genérica
        modelo = self._inferir_arquitectura(state_dict)
        if modelo is not None:
            try:
                modelo.load_state_dict(state_dict)
                metadata['tipo_carga'] = 'inferido_state_dict'
                return modelo
            except Exception:
                pass

        # Último intento: crear QuartoCNNExtended con strict=False
        if has_extended:
            try:
                modelo = QuartoCNNExtended(state_dict)
                modelo.load_state_dict(state_dict, strict=False)
                metadata['tipo_carga'] = 'QuartoCNNExtended_parcial'
                return modelo
            except Exception:
                pass

        # Último intento: crear QuartoCNN con strict=False
        try:
            modelo = QuartoCNN(state_dict)
            modelo.load_state_dict(state_dict, strict=False)
            metadata['tipo_carga'] = 'QuartoCNN_parcial'
            return modelo
        except Exception:
            pass

        return None

    def _crear_desde_arquitectura(self, arch_info: dict) -> Optional[nn.Module]:
        """Crea un modelo desde información de arquitectura."""
        if arch_info is None:
            return None

        # Arquitectura genérica CNN para Quarto
        capas = []

        in_channels = arch_info.get('in_channels', 1)

        for i, capa_info in enumerate(arch_info.get('capas', [])):
            if capa_info['tipo'] == 'conv':
                capas.append(nn.Conv2d(
                    in_channels if i == 0 else capa_info.get('in_ch', 64),
                    capa_info.get('out_ch', 64),
                    capa_info.get('kernel', 3),
                    padding=capa_info.get('padding', 1)
                ))
                capas.append(nn.ReLU())
            elif capa_info['tipo'] == 'linear':
                capas.append(nn.Flatten())
                capas.append(nn.Linear(capa_info['in_features'], capa_info['out_features']))

        if capas:
            return nn.Sequential(*capas)
        return None

    def _inferir_arquitectura(self, state_dict: dict) -> Optional[nn.Module]:
        """Infiere la arquitectura desde el state_dict para modelos Sequential."""
        claves = list(state_dict.keys())

        # Verificar si es un modelo Sequential (claves como "0.weight", "2.weight", etc.)
        sequential_pattern = any(clave.split('.')[0].isdigit() for clave in claves if 'weight' in clave)

        if not sequential_pattern:
            return None

        # Detectar patrón de capas
        capas_info = []

        for clave in claves:
            if 'weight' in clave:
                tensor = state_dict[clave]
                idx = clave.split('.')[0]
                if idx.isdigit():
                    if len(tensor.shape) == 4:  # Conv2d
                        capas_info.append({
                            'idx': int(idx),
                            'tipo': 'conv',
                            'out_ch': tensor.shape[0],
                            'in_ch': tensor.shape[1],
                            'kernel': tensor.shape[2]
                        })
                    elif len(tensor.shape) == 2:  # Linear
                        capas_info.append({
                            'idx': int(idx),
                            'tipo': 'linear',
                            'out_features': tensor.shape[0],
                            'in_features': tensor.shape[1]
                        })

        if not capas_info:
            return None

        # Ordenar por índice
        capas_info.sort(key=lambda x: x['idx'])

        # Construir modelo
        modulos = nn.ModuleDict()
        max_idx = max(c['idx'] for c in capas_info)

        for i in range(max_idx + 1):
            capa = next((c for c in capas_info if c['idx'] == i), None)
            if capa:
                if capa['tipo'] == 'conv':
                    modulos[str(i)] = nn.Conv2d(
                        capa['in_ch'], capa['out_ch'], capa['kernel'],
                        padding=capa['kernel'] // 2
                    )
                elif capa['tipo'] == 'linear':
                    modulos[str(i)] = nn.Linear(capa['in_features'], capa['out_features'])
            else:
                # Asumir ReLU para índices faltantes entre capas
                modulos[str(i)] = nn.ReLU()

        # Convertir a Sequential
        layers = [modulos[str(i)] for i in range(max_idx + 1)]
        return nn.Sequential(*layers)


    def preparar_entrada(self, estado_tablero, piezas_disponibles) -> torch.Tensor:
        """Prepara la entrada para el modelo."""
        # Convertir estado del juego a tensor
        if isinstance(estado_tablero, torch.Tensor):
            x = estado_tablero.clone()
        else:
            x = torch.tensor(estado_tablero, dtype=torch.float32)

        # Asegurar dimensiones correctas [batch, channels, height, width]
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(0)

        return x.to(self.device)