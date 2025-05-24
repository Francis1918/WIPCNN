# Architecture
To excel at Quarto, a neural network (NN) must handle **two interdependent tasks**: **placing a piece** on the board and **selecting a piece** for the opponent, while evaluating complex spatial and combinatorial relationships. Below is a specialized **hybrid architecture** inspired by AlphaZero and Transformers, optimized for Quarto’s unique challenges:

---

### **1. Input Representation**
Encode the game state as a **multi-channel tensor** capturing:
- **Board State**: `4x4x4` tensor (one-hot encoding of 4 binary attributes for each cell: height, color, shape, texture).
- **Selected Piece**: `4x1` vector (current piece to place, one-hot for each attribute).
- **Available Pieces**: `16x1` binary mask (1 = piece is still in play).
- **Phase Indicator**: `2x1` one-hot (placement or selection phase).

**Example**:  
`Input = concatenate([board_tensor, selected_piece, available_pieces, phase_indicator])`

---

### **2. Architecture Design**
#### **A. Spatial Feature Extractor (CNN Branch)**  
Processes the board state using **3D convolutions** to detect lines, patterns, and threats:  
- **Conv3D** (filters=128, kernel_size=(2,2,4), stride=1, ReLU) → Extracts local spatial patterns.  
- **Residual Blocks** (2x Conv3D + BatchNorm + Skip Connections) → Captures deeper board dynamics.  
- **Global Attention Pooling** → Focuses on critical regions (e.g., near-complete lines).  

#### **B. Piece Interaction Network (Transformer Branch)**  
Models relationships between available pieces and game state:  
- **Embedding Layer**: Projects available pieces and selected piece into 64D vectors.  
- **Multi-Head Self-Attention** (4 heads) → Identifies strategic piece selections (e.g., denying opponent useful pieces).  
- **Cross-Attention** → Links piece embeddings to board features (e.g., "Which piece could complete a line?").  

#### **C. Fusion & Decision Heads**  
- **Concatenate** CNN and Transformer outputs → `[spatial_features, piece_features]`.  
- **Dense Layers** (256 units, Swish activation) → Fuse multimodal information.  
- **Dual Policy Heads**:  
  - **Placement Head**: `16` outputs (softmax over empty cells).  
  - **Selection Head**: `15` outputs (softmax over available pieces).  
- **Value Head**: `1` output (tanh, predicts win probability [-1, 1]).  

---

### **3. Training Strategy**  
#### **A. Reinforcement Learning (AlphaZero-Style)**  
- **Self-Play**: Train against itself using Monte Carlo Tree Search (MCTS) for exploration.  
- **Reward**: +1 for win, -1 for loss, 0 for draw.  
- **Loss**: Weighted sum of:  
  - Policy loss (cross-entropy with MCTS visit counts).  
  - Value loss (MSE between predicted and actual outcome).  

#### **B. Data Augmentation**  
- Apply **rotations/flips** to the board to exploit symmetry (reduces overfitting).  
- Randomize piece attribute order (prevants bias toward specific attributes).  

#### **C. Curriculum Learning**  
- Start with simplified Quarto (e.g., 3x3 board, 2 attributes), then scale up.  

---

### **4. Key Innovations**  
- **3D Convolutions**: Directly model attribute interactions (e.g., tall + red pieces).  
- **Cross-Attention**: Explicitly link piece selection to board threats (e.g., "If I give the opponent a tall piece, where could they win?").  
- **Dual Policy Heads**: Separates placement/selection tasks while sharing underlying logic.  

---

### **5. Performance Enhancers**  
- **Mask Invalid Actions**: Set logits of illegal moves (e.g., occupied cells) to `-inf`.  
- **Symmetric MCTS Rollouts**: Evaluate mirrored/rotated states during tree search.  
- **Meta-Learning**: Fine-tune on human expert games to address strategy gaps.  

---

### **Implementation Tools**  
- **Framework**: PyTorch or JAX (for GPU acceleration).  
- **MCTS Library**: Custom implementation with prioritization by NN value estimates.  

This architecture balances **spatial reasoning** (CNN), **combinatorial logic** (Transformer), and **long-term planning** (RL+MCTS), making it robust against both heuristic-based and NN opponents in Quarto.
# Quarto Code
Here\'s a PyTorch implementation of the Quarto-specific architecture. This code focuses on the neural network structure (training loops and MCTS are excluded for brevity):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock3D(nn.Module):
    """3D Residual Block with BatchNorm"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(in_channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class SpatialFeatureExtractor(nn.Module):
    """3D CNN for processing board state"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 128, kernel_size=(2,2,4))
        self.res_blocks = nn.Sequential(
            ResidualBlock3D(128),
            ResidualBlock3D(128)
        )
        self.attn_pool = nn.Linear(128, 1)  # Attention pooling

    def forward(self, x):
        # Input: (B, 1, 4, 4, 4)
        x = F.relu(self.conv1(x))  # (B, 128, 3, 3, 1)
        x = self.res_blocks(x)
        
        # Attention pooling
        batch, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C)
        attn_weights = F.softmax(self.attn_pool(x), dim=1)
        return (x * attn_weights).sum(dim=(1,2,3))  # (B, 128)

class PieceInteractionNetwork(nn.Module):
    """Transformer-based piece relationship model"""
    def __init__(self):
        super().__init__()
        self.piece_embed = nn.Linear(4, 64)  # Embed individual pieces
        self.selected_embed = nn.Linear(4, 64)
        
        # Transformer components
        self.self_attn = nn.MultiheadAttention(64, 4)
        self.cross_attn = nn.MultiheadAttention(64, 4)
        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(64)

    def forward(self, selected_piece, available_pieces, spatial_feat):
        # Embed pieces
        available_emb = self.piece_embed(available_pieces)  # (B, 16, 64)
        selected_emb = self.selected_embed(selected_piece).unsqueeze(1)  # (B, 1, 64)
        
        # Self-attention over pieces
        tokens = torch.cat([selected_emb, available_emb], dim=1).permute(1,0,2)
        attn_out, _ = self.self_attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)
        
        # Cross-attention with spatial features
        spatial_feat = spatial_feat.unsqueeze(0)  # (1, B, 128)
        cross_out, _ = self.cross_attn(tokens, spatial_feat, spatial_feat)
        tokens = self.norm2(tokens + cross_out)
        
        return tokens.mean(dim=0)  # (B, 64)

class QuartoNet(nn.Module):
    """Complete Quarto Architecture"""
    def __init__(self):
        super().__init__()
        self.spatial_net = SpatialFeatureExtractor()
        self.piece_net = PieceInteractionNetwork()
        
        # Fusion and decision layers
        self.fc = nn.Sequential(
            nn.Linear(128+64+2, 256),  # +2 for phase indicator
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU()
        )
        
        # Output heads
        self.placement_head = nn.Linear(256, 16)
        self.selection_head = nn.Linear(256, 16)
        self.value_head = nn.Linear(256, 1)

    def forward(self, board, selected_piece, available_pieces, phase):
        # Input processing
        spatial_feat = self.spatial_net(board.unsqueeze(1))  # (B, 128)
        piece_feat = self.piece_net(selected_piece, available_pieces, spatial_feat)
        
        # Concatenate features + phase
        combined = torch.cat([spatial_feat, piece_feat, phase], dim=1)
        hidden = self.fc(combined)
        
        # Outputs
        place_logits = self.placement_head(hidden)
        select_logits = self.selection_head(hidden)
        value = torch.tanh(self.value_head(hidden))
        
        return place_logits, select_logits, value

# Example usage
if __name__ == "__main__":
    # Mock input data
    batch_size = 4
    board = torch.randn(batch_size, 4, 4, 4)  # Board state
    selected_piece = torch.randn(batch_size, 4)  # Current piece to place
    available_pieces = torch.randn(batch_size, 16)  # Available pieces
    phase = torch.tensor([[1,0],[0,1],[1,0],[0,1]])  # Phase indicator
    
    model = QuartoNet()
    p_logits, s_logits, value = model(board, selected_piece, available_pieces, phase)
    
    print("Placement logits shape:", p_logits.shape)
    print("Selection logits shape:", s_logits.shape)
    print("Value shape:", value.shape)
```

Key implementation details:
1. **3D Convolutions**: Process the 4x4x4 board through specialized 3D residual blocks
2. **Dual Attention**: Combines self-attention for piece relationships and cross-attention between pieces and board state
3. **Phase Handling**: Explicit phase indicator input helps network distinguish between placement/selection modes
4. **Output Masking**: While not shown here, you\'d mask invalid moves using `torch.where` with game rules before applying softmax

To use this effectively:
1. Add action masking based on game rules before applying softmax
2. Implement MCTS with this network as the policy/value estimator
3. Use symmetric data augmentation during training (rotations/flips)
4. Combine with reinforcement learning pipeline (e.g., AlphaZero-style training)

Would you like me to elaborate on any specific component or show how to integrate this with a training loop?