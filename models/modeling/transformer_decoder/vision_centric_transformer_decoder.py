# Modified from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d
from detectron2.utils.registry import Registry
from .position_encoding import PositionEmbeddingSine
from .group_attention import MixerMlp, AssignAttention, Mlp


TRANSFORMER_DECODER_REGISTRY = Registry("TRANSFORMER_MODULE")
TRANSFORMER_DECODER_REGISTRY.__doc__ = """
Registry for transformer module in MaskFormer.
"""


def build_transformer_decoder(cfg, in_channels, mask_classification=True):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME # "MultiScaleMaskedTransformerDecoder"
    return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, in_channels, mask_classification)


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self, tgt, tgt_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None, query_pos: Optional[Tensor] = None
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self, tgt, tgt_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None, query_pos: Optional[Tensor] = None
    ):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self, tgt, tgt_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None, query_pos: Optional[Tensor] = None
    ):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.multihead_attn( # bug
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before: # False
            return self.forward_pre(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos) # bug


class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion):
        super(BottleNeck, self).__init__()
        # Define the layers
        self.conv1 = nn.Conv2d(in_channels, out_channels // expansion, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // expansion)
        self.conv2 = nn.Conv2d(out_channels // expansion, out_channels // expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels // expansion)
        self.conv3 = nn.Conv2d(out_channels // expansion, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, mask_features): # [B, 256, 56, 56], [B, C, H, W]
        out = self.relu(self.bn1(self.conv1(mask_features))) # [B, Nv//4, H, W]
        out = self.relu(self.bn2(self.conv2(out))) # [B, Nv//4, H, W]
        out = self.bn3(self.conv3(out)) # [B, Nv, H, W]
        out += self.shortcut(mask_features) # [B, Nv, H, W]
        out = self.relu(out) # [B, Nv, H, W]
        return out

class ProtoVCQ(nn.Module):
    def __init__(self, in_channels, out_channels, num_pixel, num_query=100, num_classes=71):
        super().__init__()
        self.prototype = nn.Embedding(num_classes, out_channels) # [71, 256]
        self.pre_proj = nn.Sequential(
            nn.Conv2d(in_channels, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, 1))
        self.query_mixer = MLP(num_pixel, 256, num_query, 3) 
        self.pre_attn = CrossAttentionLayer(
                    d_model=256,
                    nhead=4,
                    dropout=0.0,
                    normalize_before=False
                )
        self.assign_attn = AssignAttention(out_channels, 1, qkv_bias=True, gumbel=True)

    def forward(self, x):
        bt, c, h, w = x.shape
        x = self.pre_proj(x)
        x = x.flatten(2) # [bt, c, hw]
        prototypes = self.prototype.weight.unsqueeze(1).repeat(1, bt, 1) # [71, bt, c]
        query = self.query_mixer(x).permute(2, 0, 1) # [bt, c, 100] -> [100, bt, c]
        query = self.pre_attn(query, prototypes)
        query = query.transpose(0, 1) # [bt, 100, c]
        query = query + self.assign_attn(query, x.transpose(1, 2))[0] # [bt, 100, c]
        query = query.transpose(0, 1) # [100, bt, c]
        return query, prototypes


class VCQ(nn.Module):
    def __init__(self, in_channels, out_channels, num_pixel, num_query=100):
        super().__init__()

        self.pre_proj = nn.Sequential(
            nn.Conv2d(in_channels, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, 1))
        self.query_mixer = MLP(num_pixel, 256, num_query, 3) 
        self.assign_attn = AssignAttention(out_channels, 1, qkv_bias=True, gumbel=True)


    def forward(self, x):
        # bt, c, h, w = x.shape
        x = self.pre_proj(x)
        x = x.flatten(2) # [bt, c, hw]
        query = self.query_mixer(x).permute(2, 0, 1) # [bt, c, 100] -> [100, bt, c]
        query = query.transpose(0, 1) # [bt, 100, c]
        query = query + self.assign_attn(query, x.transpose(1, 2))[0] # [bt, 100, c]
        query = query.transpose(0, 1) # [100, bt, c]
        return query

@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskedVCTDecoder(nn.Module):
    _version = 2

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        num_frames: int,  # * add for avs
        audio_out_dim: int,  # * add for query init
        visual_out_dim: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        dataset_name: str,
        use_cosine_loss: bool,
        is_s4_data: bool,
        is_ms3_data: bool
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            num_frames: number of frames
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)  
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        # query type
        self.audio_out_dim = audio_out_dim
        self.visual_out_dim = visual_out_dim
        self.num_classes = num_classes

        # data type
        self.is_s4_data = is_s4_data
        self.is_ms3_data = is_ms3_data

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )


            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm, 
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        query_feat_dim = hidden_dim
        self.query_feat = nn.Embedding(num_queries, query_feat_dim)  

        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)  
        
        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.all_num_feature_levels = 4 # add audio
        self.level_embed = nn.Embedding(self.all_num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.all_num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        # others
        self.dataset_name = dataset_name
        self.use_cosine_loss = use_cosine_loss   


        if is_s4_data:
            self.visual_query_block = ProtoVCQ(in_channels=hidden_dim, out_channels=hidden_dim, num_pixel=96*96, num_query=self.num_queries, num_classes=23)
        elif is_ms3_data:
            self.visual_query_block = VCQ(in_channels=hidden_dim, out_channels=hidden_dim, num_pixel=96*96)
        else:
            self.visual_query_block = ProtoVCQ(in_channels=hidden_dim, out_channels=hidden_dim, num_pixel=96*96, num_query=self.num_queries, num_classes=self.num_classes)

        if not is_ms3_data:
            scale = hidden_dim ** -0.5
            self.audio_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
            nn.init.normal_(self.audio_proj.weight, std=scale)
            self.proto_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
            nn.init.normal_(self.proto_proj.weight, std=scale)

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        ret["num_frames"] = cfg.MODEL.FUSE_CONFIG.NUM_FRAMES

        ret["audio_out_dim"] = cfg.MODEL.FUSE_CONFIG.AUDIO_OUT_DIM
        ret["visual_out_dim"] = cfg.MODEL.FUSE_CONFIG.VISUAL_OUT_DIM

        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1 # 注意此处有-1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["dataset_name"] = cfg.DATASETS.TRAIN[0][:5]
        if cfg.MODEL.MASK_FORMER.COSINE_WEIGHT > 0:
            ret["use_cosine_loss"] = True
        else:
            ret["use_cosine_loss"] = False
        is_s4_data = True if cfg.INPUT.DATASET_MAPPER_NAME == "avss4_semantic" else False
        is_ms3_data = True if cfg.INPUT.DATASET_MAPPER_NAME == "avsms3_semantic" else False
        ret["is_s4_data"] = is_s4_data
        ret["is_ms3_data"] = is_ms3_data
        return ret

    def forward(self, x, audio_features, mask_features, mask=None):
        # x is a list of multi-scale feature [B, 256, 7, 7], [B, 256, 14, 14], [B, 256, 28, 28]
        # audio_features: [B, 256, 6, 4]
        # mask_features: [B, 256, 56, 56]
     
        bt, c_m, h_m, w_m = mask_features.shape # B, 256, 56, 56
    
        assert len(x) == self.num_feature_levels # 3
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        size_list.append(audio_features.shape[-2:]) # [6, 4]
        pos.append(self.pe_layer(audio_features, None).flatten(2)) # [B, 256, 6*4]
        src.append(self.input_proj[0](audio_features).flatten(2) + self.level_embed.weight[0][None, :, None]) # [B, 256, 6*4]
        pos[-1] = pos[-1].permute(2, 0, 1) # [6*4, B, 256]
        src[-1] = src[-1].permute(2, 0, 1) # [6*4, B, 256]

        for i in range(self.num_feature_levels): # 3           
            size_list.append(x[i].shape[-2:])  # * [7,7] [14,14] [28,28]
            pos.append(self.pe_layer(x[i], None).flatten(2)) # [B, 256, 7*7] [B, 256, 14*14] [B, 256, 28*28]
            src.append(self.input_proj[i + 1](x[i]).flatten(2) + self.level_embed.weight[i + 1][None, :, None]) # [B, 256, 7*7] [B, 256, 14*14] [B, 256, 28*28]
            # flatten BxCxHW to HWxBxC
            pos[-1] = pos[-1].permute(2, 0, 1) # [7*7, B, 256], [14*14, B, 256] [28*28, B, 256]
            src[-1] = src[-1].permute(2, 0, 1) # [7*7, B, 256], [14*14, B, 256] [28*28, B, 256]

        # * NOTE: For avs, we change it with time sequence.
        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bt, 1) # [100, B, 256]
        query_embed = query_embed.reshape(self.num_queries, -1, query_embed.shape[-1]) # [100, B, 256]
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bt, 1) # [100, B, 256]
        output = output.reshape(self.num_queries, -1, output.shape[-1])  # * output is equal to query_feat # [100, B, 256]

        if self.is_ms3_data:
            visual_querys = self.visual_query_block(mask_features) # [Nv, B, C], [cls, B, C]
        else:
            visual_querys, prototypes = self.visual_query_block(mask_features) # [Nv, B, C], [cls, B, C]
            prototypes = prototypes.transpose(0, 1) # [B, cls, C]

        output = output + visual_querys # [Nv, B, 256], [cls, B, 256]
        
        predictions_class = []
        predictions_mask = []
        middles_attn_mask = [] 
        # prediction heads on learnable query features, attn_mask: [B*8, Nv, attn_mask_target_size], [280, 100, 6*4]
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0]) # [6, 4]
        predictions_class.append(outputs_class) # [B, Nv, 72]
        predictions_mask.append(outputs_mask) # [B, Nv, 56, 56]
        if self.use_cosine_loss: # ada loss
            middles_attn_mask.append(outputs_mask.reshape(bt, self.num_queries, -1)) # [B, Nv, 56*56]
        for i in range(self.num_layers): # 9
            level_index = i % self.all_num_feature_levels # i % 4
            if level_index == 0: # key and value are audio_features
                attn_mask = torch.zeros_like(attn_mask, dtype=torch.bool) # [B*8, 100, H*W]
            else:
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False # ？[B*8, 100, H*W]
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, # [Nv, B, C]
                src[level_index], # [H*W, B, C]
                memory_mask=attn_mask, # [B*8, Nv, H*W]
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], # [H*W, B, C]
                query_pos=query_embed, # [Nv, B, C]
            ) # [Nv, B, C]
            output = self.transformer_self_attention_layers[i](output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_embed) # [100, B, C]

            # FFN
            output = self.transformer_ffn_layers[i](output) # [100, B, C]

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.all_num_feature_levels]
            )
            predictions_class.append(outputs_class) # list([B, 100, 72], ...)
            predictions_mask.append(outputs_mask) # list([B, 100, 56, 56], ...)
            if self.use_cosine_loss:
                if i == self.num_layers - 1:  #! without the last layer
                    continue
                else:
                    middles_attn_mask.append(outputs_mask.reshape(bt, self.num_queries, -1))  
        assert len(predictions_class) == self.num_layers + 1

        out = {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "aux_outputs": self._set_aux_loss(predictions_class if self.mask_classification else None, predictions_mask),
            "middles_attn_mask": middles_attn_mask,  # for cosine loss
        }

        if not self.is_ms3_data:
            predict_audio_embed = self.audio_proj(audio_features.mean(2).mean(2))
            predict_proto_embed = self.proto_proj(prototypes)
            predict_audio_embed = predict_audio_embed / predict_audio_embed.norm(dim=-1, keepdim=True)
            predict_proto_embed = predict_proto_embed / predict_proto_embed.norm(dim=-1, keepdim=True)
            out['feats'] = {'audio_feat':predict_audio_embed, 'proto_feat':predict_proto_embed}
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)

        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [{"pred_logits": a, "pred_masks": b} for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
