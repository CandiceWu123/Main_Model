import torch
from torch import nn, Tensor
import torch.nn.functional as F
import copy
from typing import List, Optional
from .position_encoding import PositionEmbeddingSine
import fvcore.nn.weight_init as weight_init                

class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = nn.ReLU()
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# cross-attention
class CrossAttentionLayer(nn.Module):
    def __init__(
            self,
            d_model,
            nhead,
            dropout=0.0
    ):
        super(CrossAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        """
        self.pe_layer = PositionEmbeddingSine(d_model // 2, normalize=True)
        self.query_proj = nn.Linear(d_model, 1)
        self.heads = nhead
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm5 = nn.BatchNorm1d(d_model)
        """
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, query, key, value,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        query2 = self.multihead_attn(query=self.with_pos_embed(query, query_pos),
                                     key=self.with_pos_embed(key, pos),
                                     value=value, attn_mask=memory_mask,
                                     key_padding_mask=memory_key_padding_mask)[0]
        query = query + self.dropout(query2)
        query = self.norm(query)

        return query


# prenorm = False
class MultiScaleMaskedTransformerDecoder(nn.Module):
    def __init__(
            self,
            in_channels,
            *,
            hidden_dim: int,
            num_queries: int,
            nheads: int,
            dim_feedforward: int,
            dec_layers: int,
            mask_dim: int,
            enforce_input_project: bool,
            add_mask: bool
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        self.add_mask = add_mask
        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
        
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.key_level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.value_level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for i in range(self.num_feature_levels):
            if in_channels[i] != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels[i], hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    # support和query如何交互
    def input_process(self, s_features, q_features, s_masks, q_masks, nshot):
        # from low-resolution to high-resolution
        if nshot == 1:
            x = []
            y = []
            masks = []
            for i in range(self.num_feature_levels):
                x.append(s_features[i] * s_masks[i])
                y.append(q_features[i])
                masks.append(~q_masks[i].bool())
            if self.add_mask:
                return x, y, masks
            else:
                x = []
                y = []
                for i in range(self.num_feature_levels):
                    x.append(s_features[i] * s_masks[i])
                    y.append(q_features[i] * q_masks[i])
                return x, y, masks
        else:
            y = [q_features[l_i] * q_masks[l_i] for l_i in range(self.num_feature_levels)]
            masks = []
            x_level = []
            for s_k in range(nshot):
                x_shot = [s_features[s_k][l_i] * s_masks[s_k][l_i] for l_i in range(self.num_feature_levels)]
                x_level.append(x_shot) # [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]] (bsz, ch, h, w)
            # x = [torch.mean(torch.stack(x_level[:][l_i], dim=1), dim=1) for l_i in range(self.num_feature_levels)]
            x = []
            for l_i in range(self.num_feature_levels):
                temp = 0
                for s_k in range(nshot):
                    temp = temp + x_level[s_k][l_i]
                x.append(temp/nshot)
            return x, y, masks

    def forward(self, s_features, q_features, s_masks, q_masks, mask_features, nshot):
        x, y, masks = self.input_process(s_features, q_features, s_masks, q_masks, nshot)
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        assert len(y) == self.num_feature_levels

        key = []  # x_s_features
        value = []  # y_q_features
        pos = []  # key
        size_list = []
        attn_mask = []

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            key.append(self.input_proj[i](x[i]).flatten(2) + self.key_level_embed.weight[i][None, :, None])
            value.append(y[i].flatten(2) + self.value_level_embed.weight[i][None, :, None])
            if self.add_mask:
                attn_mask.append(masks[i].flatten(2).unsqueeze(1).repeat(1, self.num_heads, self.num_queries, 1).flatten(0, 1))

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            key[-1] = key[-1].permute(2, 0, 1)
            value[-1] = value[-1].permute(2, 0, 1)

        _, bs, _ = key[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            # level_index = self.num_feature_levels - i - 1
            if self.add_mask:
                cur_attn_mask = attn_mask[level_index]
                cur_attn_mask[torch.where(cur_attn_mask.sum(-1) == cur_attn_mask.shape[-1])] = False
            # print(key[level_index].shape[0])
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, key[level_index], value[level_index],
                memory_mask=cur_attn_mask if self.add_mask else None,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

        # QNC -> NQC
        output = self.decoder_norm(output).transpose(0, 1)
        mask_embed = self.mask_embed(output)
        # 1/4 (H, W)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)  # (b, num_queries, h, w)

        return output, outputs_mask


if __name__ == "__main__":
    mask = torch.randn(1, 1, 384, 384)
    q_masks = [torch.zeros(1, 1, 12, 12)] + [torch.zeros(1, 1, 24, 24)] + [torch.zeros(1, 1, 48, 48)]
    feature1 = torch.randn(1, 256, 48, 48)
    feature2 = torch.randn(1, 512, 24, 24)
    feature3 = torch.randn(1, 1024, 12, 12)
    features = [feature3] + [feature2] + [feature1]
    s_features = features
    q_features = features
    feat_channels = [256, 512, 1024]
    s_masks = q_masks
    predictor = MultiScaleMaskedTransformerDecoder([1024, 512, 256], hidden_dim=256,
                                                   num_queries=100, nheads=8, dim_feedforward=2048, dec_layers=3,
                                                   mask_dim=256, enforce_input_project=False, add_mask=False)
    mask_features = torch.randn(1, 256, 96, 96)
    output, output_masks = predictor(s_features, q_features, s_masks, q_masks, mask_features)
    print(output.shape)
    print(output_masks.shape)


