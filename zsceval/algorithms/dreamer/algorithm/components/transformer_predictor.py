import torch
import torch.nn as nn
import torch.nn.functional as F  # Added for TransfomerPredictor


class ToMTransformerSliding(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        window_size: int,
        dim_feedforward: int = None,  # Allow overriding
        t_max: int = 1000,  # Max sequence length for pos encoding
        dropout: float = 0.1  # Standard dropout
    ):
        """
        input_dim:    原始特征维度 N（比如 world-model 给出的 h_t 和 z_t 拼接后的长度）
        d_model:      Transformer 内部隐藏维度
        nhead:        注意力头数
        num_layers:   TransformerEncoder 堆叠层数
        window_size:  我们要“滑动窗口”每个子序列的长度 t
        dim_feedforward: FFN hidden dim
        t_max:        用于位置编码的最大序列长度
        dropout:      Dropout rate
        """
        super().__init__()
        assert window_size <= t_max, "window_size 必须 <= t_max"
        self.window_size = window_size
        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        # 1. 把 input_dim 映射到 d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        # 2. 位置编码
        self.pos_embedding = nn.Embedding(t_max, d_model)
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # PyTorch default, expects [seq_len, batch, dim]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.layer_norm_out = nn.LayerNorm(
            d_model)  # Optional: final layer norm

    def forward(
        self,
        X_win: torch.Tensor,
        pad_mask_win: torch.Tensor,  # True for padded positions
        reset_mask_win: torch.Tensor  # True for new episode start positions within window
    ) -> torch.Tensor:
        """
        对单个长度为 t (window_size) 的子窗口做一次 Transformer。
        Args:
            X_win:          Input features for the window [B, t, input_dim].
            pad_mask_win:   Padding mask [B, t]. True indicates a position is padding.
            reset_mask_win: Reset mask [B, t]. True indicates a position is a new episode start.
                            Used to construct a batch-uniform attention mask.
        Returns:
            torch.Tensor: Output features from the Transformer [B, t, d_model].
        """
        B, t, _ = X_win.shape
        assert t == self.window_size, f"Input sequence length {t} does not match window_size {self.window_size}"
        device = X_win.device

        # A) Project X to d_model and add positional embedding
        x_projected = self.input_proj(X_win)  # [B, t, d_model]

        # Create positional indices [0, 1, ..., t-1]
        pos_idx = torch.arange(t, device=device).unsqueeze(
            0).expand(B, -1)  # [B, t]
        x_with_pos = x_projected + \
            self.pos_embedding(pos_idx)  # [B, t, d_model]

        # B) Construct attention mask [t, t] based on reset_mask_win
        # This mask prevents attention across detected episode boundaries within the window *for the whole batch*.
        # True means "query i cannot attend key j".
        attn_mask = torch.zeros(t, t, dtype=torch.bool, device=device)

        # If any item in the batch has a reset at window position j0,
        # then create a segmentation point at j0 for the entire batch's attention mask.
        any_reset_at_window_pos = reset_mask_win.any(dim=0)  # Shape: [t]

        reset_indices = any_reset_at_window_pos.nonzero(
            as_tuple=False).squeeze(-1)
        for j0 in reset_indices:
            # Tokens before j0 cannot attend to tokens at or after j0
            attn_mask[:j0, j0:] = True
            # Tokens at or after j0 cannot attend to tokens before j0
            attn_mask[j0:, :j0] = True

        # C) Pass to TransformerEncoder
        # PyTorch TransformerEncoderLayer expects [seq_len, batch, features] by default
        x_tr_input = x_with_pos.transpose(0, 1)  # [t, B, d_model]

        # src_key_padding_mask: [B, t] (True for padded positions)
        # attn_mask: [t, t] (True for positions not allowed to attend)
        transformer_output = self.transformer_encoder(
            x_tr_input,
            mask=attn_mask,  # For self-attention, applied to all heads
            src_key_padding_mask=pad_mask_win
        )

        transformer_output = transformer_output.transpose(
            0, 1)  # -> [B, t, d_model]
        transformer_output = self.layer_norm_out(
            transformer_output)  # Optional final norm

        return transformer_output


# Corrected typo from TransfomerPredictor
class TransformerPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_atten_head: int,
        num_transformer_layers: int,  # Renamed for clarity
        num_prediction_heads: int,  # Renamed for clarity
        num_classes: int,
        window_size: int,  # Make window_size a parameter
        dim_feedforward_transformer: int = None,
        t_max_transformer: int = 1000,
        dropout_transformer: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.window_size = window_size  # Store window_size

        self.transformer = ToMTransformerSliding(
            input_dim=input_dim,  # ToMTransformerSliding takes raw input_dim
            d_model=d_model,
            nhead=n_atten_head,
            num_layers=num_transformer_layers,
            window_size=window_size,
            dim_feedforward=dim_feedforward_transformer,
            t_max=t_max_transformer,
            dropout=dropout_transformer
        )

        # Prediction heads operate on the output of the Transformer for a specific time step (e.g., the last one)
        self.prediction_heads = nn.ModuleList(
            [nn.Linear(d_model, num_classes)
             for _ in range(num_prediction_heads)]
        )

    def forward(self,
                feat_win: torch.Tensor,      # [B, window_size, input_dim]
                # [B, window_size], True for padding
                pad_mask_win: torch.Tensor,
                # [B, window_size], True for reset
                reset_mask_win: torch.Tensor
                ):
        """
        Args:
            feat_win: Features for the current window [B, window_size, input_dim].
            pad_mask_win: Padding mask for the window [B, window_size].
            reset_mask_win: Reset mask for the window [B, window_size].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - samples (torch.Tensor): Sampled actions [B, num_prediction_heads, num_classes] (one-hot).
                - logits (torch.Tensor): Raw logits [B, num_prediction_heads, num_classes].
        """
        # Get transformer output for the entire window
        # transformer_output shape: [B, window_size, d_model]
        transformer_output_sequence = self.transformer(
            feat_win, pad_mask_win, reset_mask_win)

        # Use the features from the LAST time step of the window for prediction
        # This assumes the last item in the window corresponds to the "current" time step
        # for which a prediction is needed.
        # Shape: [B, d_model]
        current_step_features = transformer_output_sequence[:, -1, :]

        logits_list = [head(current_step_features)
                       for head in self.prediction_heads]
        # Shape: [B, num_prediction_heads, num_classes]
        logits = torch.stack(logits_list, dim=1)

        # Create a distribution for sampling
        try:
            dist = torch.distributions.OneHotCategorical(logits=logits)
            samples = dist.sample()
        except ValueError as e:
            # Handle potential numerical issues if logits are problematic
            print(f"Error creating OneHotCategorical distribution: {e}")
            print(f"Logits shape: {logits.shape}, min: {logits.min()}, max: {logits.max()}, "
                  f"has_nan: {torch.isnan(logits).any()}, has_inf: {torch.isinf(logits).any()}")
            # Provide a fallback or re-raise; for now, a placeholder
            samples = torch.zeros_like(logits)
            # Depending on the severity, you might want to raise e

        return samples, logits
