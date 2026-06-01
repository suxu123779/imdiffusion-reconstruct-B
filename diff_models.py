import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = TransformerEncoderLayerWithAttn(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return TransformerEncoderWithAttn(encoder_layer, num_layers=layers)


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    raise RuntimeError(f"unsupported activation: {activation}")


class TransformerEncoderLayerWithAttn(nn.Module):
    """TransformerEncoderLayer-compatible block that can optionally return attention.

    The submodule names intentionally match torch.nn.TransformerEncoderLayer so
    existing checkpoints keep the same state_dict keys.
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(self, src, return_attn=False):
        if return_attn:
            try:
                src2, attn = self.self_attn(
                    src,
                    src,
                    src,
                    need_weights=True,
                    average_attn_weights=False,
                )
            except TypeError:
                src2, attn = self.self_attn(src, src, src, need_weights=True)
                if attn.dim() == 3:
                    attn = attn.unsqueeze(1)
        else:
            src2 = self.self_attn(src, src, src, need_weights=False)[0]
            attn = None
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if return_attn:
            return src, attn
        return src


class TransformerEncoderWithAttn(nn.Module):
    """Small nn.TransformerEncoder-compatible wrapper with optional attn return."""

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, src, return_attn=False):
        output = src
        attn_layers = []
        for layer in self.layers:
            if return_attn:
                output, attn = layer(output, return_attn=True)
                attn_layers.append(attn)
            else:
                output = layer(output)
        if return_attn:
            return output, attn_layers
        return output


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        self.strategy_embedding = nn.Embedding(
            2, config['diffusion_embedding_dim']
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step, strategy_type, return_hidden=False, return_feature_attn=False):
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)
        # print("strategy type is")
        # print(strategy_type)

        strategy_emb = self.strategy_embedding(strategy_type)
        # print("strategy emb is")
        # print(strategy_emb.shape)
        skip = []
        feature_attn_layers = []
        for layer in self.residual_layers:
            if return_feature_attn:
                x, skip_connection, feature_attn = layer(
                    x,
                    cond_info,
                    diffusion_emb,
                    strategy_emb,
                    return_feature_attn=True,
                )
                feature_attn_layers.append(feature_attn)
            else:
                x, skip_connection = layer(x, cond_info, diffusion_emb,strategy_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        hidden = x
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        if return_feature_attn:
            feature_attn = torch.cat(feature_attn_layers, dim=0)
            feature_attn = feature_attn.permute(1, 2, 0, 3, 4, 5).contiguous()
            B_attn, L_attn, layer_count, heads, K_attn, _ = feature_attn.shape
            feature_attn = feature_attn.reshape(B_attn, L_attn, layer_count * heads, K_attn, K_attn)
        if return_hidden and return_feature_attn:
            return x, hidden, feature_attn
        if return_hidden:
            return x, hidden
        if return_feature_attn:
            return x, feature_attn
        return x


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.strategy_projection = nn.Linear(diffusion_embedding_dim, channels)

        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape, return_attn=False):
        B, channel, K, L = base_shape
        if K == 1:
            if return_attn:
                return y, None
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        if return_attn:
            y, attn_layers = self.feature_layer(y.permute(2, 0, 1), return_attn=True)
        else:
            y = self.feature_layer(y.permute(2, 0, 1))
            attn_layers = None
        y = y.permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        if return_attn:
            attn_by_layer = []
            for attn in attn_layers:
                if attn is None:
                    continue
                if attn.dim() == 3:
                    attn = attn.unsqueeze(1)
                attn_by_layer.append(attn.reshape(B, L, attn.shape[1], K, K))
            if not attn_by_layer:
                raise RuntimeError("feature attention was requested but not returned")
            return y, torch.stack(attn_by_layer, dim=0)
        return y

    def forward(self, x, cond_info, diffusion_emb, strategy_emb, return_feature_attn=False):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        strategy_emb = self.strategy_projection(strategy_emb).unsqueeze(-1)

        # print("strategy emb is")
        # print(strategy_emb)
        # print(strategy_emb.shape)
        y = x + diffusion_emb + strategy_emb

        y = self.forward_time(y, base_shape)
        if return_feature_attn:
            y, feature_attn = self.forward_feature(y, base_shape, return_attn=True)
        else:
            y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
            feature_attn = None
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        if return_feature_attn:
            return (x + residual) / math.sqrt(2.0), skip, feature_attn
        return (x + residual) / math.sqrt(2.0), skip
