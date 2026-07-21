import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from torch.distributions import Categorical, Normal

import math
from maenvs4vrp.utils.ops import gather_by_index


from torch.nn.functional import scaled_dot_product_attention
#import logging

from einops import rearrange
from torch import Tensor

from collections.abc import Callable
from typing import Union, Tuple, Callable, Optional, List, Any

import abc

from dataclasses import dataclass, fields

'''
PARCO-style multi-agent policy.

Adapted from:
  - PARCO:  https://github.com/ai4co/parco  (arxiv 2409.03811)

Key PARCO ingredients:
  1. Per-agent independent action sampling  (Sec. 3.2)
  2. Highest-probability conflict handler   (Sec. 3.3)
  3. Per-agent log-probability for REINFORCE (Sec. 3.4)
'''


# ---------------------------------------------------------------------------
# PARCO embeddings and attention modules
# ---------------------------------------------------------------------------

# https://github.com/ai4co/parco/blob/main/parco/models/nn/positional_encoder.py
class PositionalEncoder(torch.nn.Module):
    """ "
    Positional encoder for transformer models.
    This module is used to add positional encodings to the input of the model:
    x = x + pe[:, :x.shape[1]]
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoder, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x, add=False):
        return x + self.pe[:, : x.shape[1]] if add else self.pe[:, : x.shape[1]]


# https://github.com/ai4co/parco/blob/main/parco/models/nn/transformer.py
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """

    def __init__(self, dim: int, eps: float = 1e-6, **unused_kwargs):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Note: done because in previous RMS norm implementation, the dim parameter was not being loaded
        weight_key = prefix + "weight"
        if weight_key in state_dict:
            weight = state_dict[weight_key]
            if not hasattr(self, "dim"):
                self.dim = weight.size(0)
                self.weight = nn.Parameter(torch.ones(self.dim, device=weight.device))
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class Normalization(nn.Module):
    def __init__(self, embed_dim, normalization="batch"):
        super(Normalization, self).__init__()
        if normalization != "layer":
            normalizer_class = {
                "batch": nn.BatchNorm1d,
                "instance": nn.InstanceNorm1d,
                "rms": RMSNorm,
            }.get(normalization, None)
            self.normalizer = (
                normalizer_class(embed_dim, affine=True)
                if normalizer_class is not None
                else None
            )
        else:
            self.normalizer = "layer"

    def forward(self, x):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(x.view(-1, x.size(-1))).view(*x.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(x.permute(0, 2, 1)).permute(0, 2, 1)
        elif self.normalizer == "layer":
            return (x - x.mean((1, 2)).view(-1, 1, 1)) / torch.sqrt(
                x.var((1, 2)).view(-1, 1, 1) + 1e-05
            )
        elif isinstance(self.normalizer, RMSNorm):
            return self.normalizer(x)
        else:
            assert self.normalizer is None, "Unknown normalizer type {}".format(
                self.normalizer
            )
            return x


# https://github.com/ai4co/rl4co/blob/main/rl4co/models/nn/attention.py
class MultiHeadAttention(nn.Module):
    """PyTorch native implementation of Flash Multi-Head Attention with automatic mixed precision support.
    Uses PyTorch's native `scaled_dot_product_attention` implementation, available from 2.0

    Note:
        If `scaled_dot_product_attention` is not available, use custom implementation of `scaled_dot_product_attention` without Flash Attention.

    Args:
        embed_dim: total dimension of the model
        num_heads: number of heads
        bias: whether to use bias
        attention_dropout: dropout rate for attention weights
        causal: whether to apply causal mask to attention scores
        device: torch device
        dtype: torch dtype
        sdpa_fn: scaled dot product attention function (SDPA) implementation
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        attention_dropout: float = 0.0,
        causal: bool = False,
        device: str = None,
        dtype: torch.dtype = None,
        sdpa_fn: Callable | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.attention_dropout = attention_dropout
        self.sdpa_fn = sdpa_fn if sdpa_fn is not None else scaled_dot_product_attention

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, (
            "Only support head_dim <= 128 and divisible by 8"
        )

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(self, x, attn_mask=None):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        attn_mask: bool tensor of shape (batch, seqlen)
        """
        # Project query, key, value
        q, k, v = rearrange(
            self.Wqkv(x), "b s (three h d) -> three b h s d", three=3, h=self.num_heads
        ).unbind(dim=0)

        if attn_mask is not None:
            attn_mask = (
                attn_mask.unsqueeze(1)
                if attn_mask.ndim == 3
                else attn_mask.unsqueeze(1).unsqueeze(2)
            )

        # Scaled dot product attention
        out = self.sdpa_fn(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout,
        )
        return self.out_proj(rearrange(out, "b h s d -> b s (h d)"))

# MLP
# https://github.com/ai4co/rl4co/blob/main/rl4co/models/nn/mlp.py

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_neurons: list[int] = [64, 32],
        dropout_probs: None | list[float] = None,
        hidden_act: str = "ReLU",
        out_act: str = "Identity",
        input_norm: str = "None",
        output_norm: str = "None",
    ):
        super().__init__()

        assert input_norm in ["Batch", "Layer", "None"]
        assert output_norm in ["Batch", "Layer", "None"]

        if dropout_probs is None:
            dropout_probs = [0.0] * len(num_neurons)
        elif len(dropout_probs) != len(num_neurons):
            log.info(
                "dropout_probs List length should match the num_neurons List length for MLP, dropouts set to False instead"
            )
            dropout_probs = [0.0] * len(num_neurons)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_neurons = num_neurons
        self.hidden_act = getattr(nn, hidden_act)()
        self.out_act = getattr(nn, out_act)()
        self.dropouts = []
        for i in range(len(dropout_probs)):
            self.dropouts.append(nn.Dropout(p=dropout_probs[i]))

        input_dims = [input_dim] + num_neurons
        output_dims = num_neurons + [output_dim]

        self.lins = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(input_dims, output_dims)):
            self.lins.append(nn.Linear(in_dim, out_dim))

        self.input_norm = self._get_norm_layer(input_norm, input_dim)
        self.output_norm = self._get_norm_layer(output_norm, output_dim)

    def forward(self, xs):
        xs = self.input_norm(xs)
        for i, lin in enumerate(self.lins[:-1]):
            xs = lin(xs)
            xs = self.hidden_act(xs)
            xs = self.dropouts[i](xs)
        xs = self.lins[-1](xs)
        xs = self.out_act(xs)
        xs = self.output_norm(xs)
        return xs

    @staticmethod
    def _get_norm_layer(norm_method, dim):
        if norm_method == "Batch":
            in_norm = nn.BatchNorm1d(dim)
        elif norm_method == "Layer":
            in_norm = nn.LayerNorm(dim)
        elif norm_method == "None":
            in_norm = nn.Identity()  # kinda placeholder
        else:
            raise RuntimeError(f"Not implemented normalization layer type {norm_method}")
        return in_norm

    def _get_act(self, is_last):
        return self.out_act if is_last else self.hidden_act

# ---------------------------------------------------------------------------
# https://github.com/ai4co/parco/blob/main/parco/models/nn/transformer.py#L153
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        feedforward_hidden: Optional[int] = None,  # if None, use 4 * embed_dim
        normalization: Optional[str] = "instance",
        norm_after: bool = False,  # if True, perform same as Kool et al.
        bias: bool = True,
        sdpa_fn: Optional[Callable] = None,
    ):
        super(TransformerBlock, self).__init__()
        feedforward_hidden = (
            4 * embed_dim if feedforward_hidden is None else feedforward_hidden
        )
        num_neurons = [feedforward_hidden] if feedforward_hidden > 0 else []
        ffn = MLP(
            input_dim=embed_dim,
            output_dim=embed_dim,
            num_neurons=num_neurons,
            hidden_act="ReLU",
        )

        self.norm_attn = (
            Normalization(embed_dim, normalization)
            if normalization is not None
            else lambda x: x
        )
        self.attention = MultiHeadAttention(
            embed_dim, num_heads, bias=bias, sdpa_fn=sdpa_fn
        )
        self.norm_ffn = (
            Normalization(embed_dim, normalization)
            if normalization is not None
            else lambda x: x
        )
        self.ffn = ffn
        self.norm_after = norm_after

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if not self.norm_after:
            # normal transformer structure
            h = x + self.attention(self.norm_attn(x), mask)
            h = h + self.ffn(self.norm_ffn(h))
        else:
            # from Kool et al. (2019)
            h = self.norm_attn(x + self.attention(x, mask))
            h = self.norm_ffn(h + self.ffn(h))
        return h


# https://github.com/ai4co/parco/blob/main/parco/tasks/ffsp_old/FFSP_PARCO/FFSPModel_SUB.py

class FeedForward(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()

        self.W1 = nn.Linear(embed_dim, embed_dim)
        self.W2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))

class TransformerFFN(nn.Module):
    def __init__(self, embed_dim=128, normalization="batch") -> None:
        super().__init__()

        self.ffn = FeedForward(embed_dim=embed_dim)
        self.norm1 = Normalization(embed_dim=embed_dim, normalization=normalization)
        self.norm2 = Normalization(embed_dim=embed_dim, normalization=normalization)

    def forward(self, x, x_old):

        x = self.norm1(x_old + x)
        x = self.norm2(x + self.ffn(x))

        return x

# ---------------------------------------------------------------------------
# PARCO Embedding block (adapted to MAENVS4VRP)
# ---------------------------------------------------------------------------

# https://github.com/ai4co/parco/blob/main/parco/models/env_embeddings/hcvrp.py

class LearnedPositionalEncoder(torch.nn.Module):
    def __init__(self, d_model, max_len=8):
        super().__init__()

        self.pe = torch.nn.Embedding(max_len, d_model)

    def forward(self, x, add=True):
        seq_len = x.size(1)

        positions = torch.arange(
            seq_len,
            device=x.device
        )

        pe = self.pe(positions).unsqueeze(0)

        return x + pe if add else pe
    
class InitEmbedding(nn.Module):
    """
    Initial embedding for nodes and agents in the environment. 
    Combines learned positional encodings with linear projections of the input features.

    """

    def __init__(
        self,
        node_feat_dim: int = 3,
        agent_feat_dim: int = 4,
        embed_dim: int = 128,
        linear_bias: bool = False,
     ):  
        super(InitEmbedding, self).__init__()

        self.pe = LearnedPositionalEncoder(embed_dim)
        self.init_embed_agents = nn.Linear(agent_feat_dim, embed_dim, linear_bias)
        self.init_embed_nodes = nn.Linear(node_feat_dim, embed_dim, linear_bias)


    def forward(self, nodes_obs, agents_obs ):

        """
        Observations:

        """
        
        agents_embedding = self.init_embed_agents(agents_obs)
        agents_embedding = self.pe(agents_embedding, add=True)  # [B, m, hdim]  
        nodes_embedding = self.init_embed_nodes(nodes_obs)
        
        return torch.cat(
            [agents_embedding, nodes_embedding], -2
        )  # [B, m+N, hdim]

    
#-------------------------------------------------------------------------------------------------------
# Context and Dynamic Embedding implementations
#-------------------------------------------------------------------------------------------------------
class ContextEmbedding(nn.Module):
    """Base class for environment context embeddings. The context embedding is used to modify the
    query embedding of the problem node of the current partial solution.
    Consists of a linear layer that projects the node features to the embedding space."""

    def __init__(self, embed_dim, num_heads,
                 agents_obs_dim,
                 global_obs_dim,
                 linear_bias=False,
                 use_communication=True,
                 num_communication_layers=1,
                 use_final_norm=False,
                 **communication_kwargs):   # note: see TransformerBlock
        
        super(ContextEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Feature projection
        self.proj_agent_feats = nn.Linear(agents_obs_dim, embed_dim, bias=linear_bias)
        self.proj_global_feats = nn.Linear(global_obs_dim, embed_dim, bias=linear_bias)
        self.project_context = nn.Linear(embed_dim * 4, embed_dim, bias=linear_bias)

        if use_communication:
            self.communication_layers = nn.ModuleList(
                [
                    CommunicationLayer(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                    )
                    for _ in range(num_communication_layers)
                ]
            )
        else:
            self.communication_layers = None

        self.norm = (
            Normalization(embed_dim, communication_kwargs.get("normalization", "rms"))
            if use_final_norm
            else None
        )

    def _cur_node_embedding(self, embeddings, cur_node_idx):
        cur_node_embedding = gather_by_index(embeddings, cur_node_idx)
        return cur_node_embedding

    def forward(self, agents_obs, global_obs, current_node_idx, node_embeddings, agents_embeddings, active_agents_mask=None):

        batch = agents_obs.shape[0]
        num_agents = agents_obs.shape[1]
        cur_node_embedding = self._cur_node_embedding(node_embeddings, current_node_idx)

        agent_state_embed = self.proj_agent_feats(agents_obs)  # [B, M, hdim]

        if global_obs is None:
            global_embed = torch.zeros(
                batch,
                self.embed_dim,
                device=agents_obs.device
            )
        else:
            global_embed = self.proj_global_feats(global_obs)[..., None, :].repeat(1, num_agents, 1)  # [B, M, hdim]

        context_embed = torch.cat(
            [cur_node_embedding, agent_state_embed, global_embed, agents_embeddings], dim=-1
        )

        # [B, M, hdim, 3] -> [B, M, hdim]
        context_embed = self.project_context(context_embed)
        if self.communication_layers is not None:
            h_comm = context_embed
            for layer in self.communication_layers:
                h_comm = layer(h_comm, active_agents_mask=active_agents_mask)
        else:
            h_comm = context_embed
        if self.norm is not None:
            h_comm = self.norm(h_comm)
        return h_comm



class DynamicEmbedding(nn.Module):
    """

    Dynamic embedding

    """

    def __init__(self, nodes_dyn_dim, 
                 embed_dim, 
                 linear_bias=False):
        super(DynamicEmbedding, self).__init__()
        self.projection = nn.Linear(nodes_dyn_dim, 3 * embed_dim, bias=linear_bias)

    def forward(self, dyn_node_obs):
        glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic = self.projection(
            dyn_node_obs
        ).chunk(3, dim=-1)
        return glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic

# ---------------------------------------------------------------------------
# PARCO Comuncation Layer block
# ---------------------------------------------------------------------------

# https://github.com/ai4co/parco/blob/main/parco/models/env_embeddings/communication.py

class CommunicationLayer(nn.Module):

    def __init__(self, embed_dim=128, num_heads=8, normalization="batch"):
        super().__init__()
        self.mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.feed_forward = TransformerFFN(embed_dim=embed_dim, normalization=normalization)

    def forward(self, x, active_agents_mask=None):

        x = self.feed_forward(self.mha(x, attn_mask=active_agents_mask), x)
        return x

# ---------------------------------------------------------------------------
# PARCO Encoder block
# ---------------------------------------------------------------------------

# https://github.com/ai4co/parco/blob/main/parco/models/encoder.py

class PARCOEncoder(nn.Module):
    def __init__(
        self,
        num_heads: int = 8,
        embed_dim: int = 128,
        num_layers: int = 3,
        normalization: str = "instance",
        use_final_norm: bool = False,
        norm_after: bool = False,
        use_pos_token: bool = False,
        trainable_pos_token: bool = True,
        **transformer_kwargs,
    ):
        super(PARCOEncoder, self).__init__()

        self.layers = nn.Sequential(
            *(
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    normalization=normalization,
                    norm_after=norm_after,
                    **transformer_kwargs,
                )
                for _ in range(num_layers)
            )
        )

        if use_pos_token and trainable_pos_token:
            self.pos_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        elif use_pos_token:
            self.pos_token = torch.zeros(1, 1, embed_dim)
        else:
            self.pos_token = None
        self.use_pos_token = use_pos_token
        self.norm = Normalization(embed_dim, normalization) if use_final_norm else None

    def forward(
        self, init_embend, mask: Union[Tensor, None] = None
    ) -> Tuple[Tensor, Tensor]:
        # Transfer to embedding space

        if self.use_pos_token:
            # Add a POS (pause-of-sequence) action to the embeddings
            # [B, N, H] -> [B, N+1, H]
            pos_token = self.pos_token.expand(init_embend.size(0), 1, -1).to(init_embend.device)
            init_embend = torch.cat([init_embend, pos_token], dim=1)

        # Process embedding
        h = init_embend
        for layer in self.layers:
            h = layer(h, mask)

        # https://github.com/meta-llama/llama/blob/8fac8befd776bc03242fe7bc2236cdb41b6c609c/llama/model.py#L493
        if self.norm is not None:
            h = self.norm(h)

        # Return latent representation and initial embedding
        # [B, N, H]
        return h, init_embend    

# ---------------------------------------------------------------------------
# PARCO Pointer block
# ---------------------------------------------------------------------------

class PointerAttention(nn.Module):

    def __init__(self, use_tanh=True, C=10, embed_dim=128, num_heads=8, linear_bias=False):
        super(PointerAttention, self).__init__()
        self.use_tanh = use_tanh
        self.C = C
        self.num_heads = num_heads
        self.project_out = nn.Linear(
            embed_dim, embed_dim, bias=linear_bias
        )

    def _make_heads(self, query, key, value):
        query = rearrange(query, "... g (h s) -> ... h g s", h=self.num_heads) #[B, m, E] -> [B, h, m, dh]
        key = rearrange(key, "... g (h s) -> ... h g s", h=self.num_heads) #[B, N, E] -> [B, h, N, dh]
        value = rearrange(value, "... g (h s) -> ... h g s", h=self.num_heads) #[B, N, E] -> [B, h, N, dh]

        return query, key, value

    def _unmake_heads(self, glimpse):
        glimpse = rearrange(glimpse, "... h n g -> ... n (h g)", h=self.num_heads) #[B, h, N, dh] -> [B, N, E]

        return glimpse

    def forward(self, glimpse_q, glimpse_k, glimpse_v, logit_k, mask=None):

        glimpse_q, glimpse_k, glimpse_v = self._make_heads(
            query=glimpse_q,
            key=glimpse_k,
            value=glimpse_v
        )

        if mask.ndim == 3:
            attn_mask = mask.unsqueeze(1)
        else:
            attn_mask = mask.unsqueeze(1).unsqueeze(2)

        glimpse = scaled_dot_product_attention(query=glimpse_q, key=glimpse_k, value=glimpse_v, attn_mask=attn_mask)

        glimpse = self._unmake_heads(glimpse=glimpse)

        glimpse = self.project_out(glimpse)

        u = torch.matmul(glimpse, logit_k.transpose(-2, -1)) / math.sqrt(glimpse.size(-1))

        if self.use_tanh:
            logits = torch.tanh(u) * self.C
        else:
            logits = u

        if mask is not None:
            logits = logits.masked_fill(mask.expand_as(logits) == False, float('-inf'))

        return logits

# ---------------------------------------------------------------------------
# PARCO Decoder block
# ---------------------------------------------------------------------------

# https://github.com/ai4co/parco/blob/main/parco/models/decoder.py    

class PARCODecoder(nn.Module):
    def __init__(
        self,
        nodes_dyn_obs_dim,
        agents_obs_dim,
        global_obs_dim,
        embed_dim: int = 128,
        num_heads: int = 8,
        linear_bias: bool = False,
        use_graph_context: bool = False,
        use_pos_token: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.use_pos_token = use_pos_token
        assert embed_dim % num_heads == 0
        self.context_embedding = ContextEmbedding(embed_dim, num_heads=num_heads, agents_obs_dim=agents_obs_dim, global_obs_dim=global_obs_dim, linear_bias=linear_bias)
        self.dynamic_embedding = DynamicEmbedding(nodes_dyn_dim=nodes_dyn_obs_dim, embed_dim=embed_dim)


        if use_graph_context:
            raise ValueError("PARCO does not use graph context")


    def _compute_q(self, agents_obs, global_obs, cur_nodes_idx, node_embeddings, agents_embeddings, graph_context=None, active_agents_mask=None):

        context_embedding = self.context_embedding(agents_obs, 
                                                   global_obs, 
                                                   cur_nodes_idx, 
                                                   node_embeddings, 
                                                   agents_embeddings,
                                                   active_agents_mask=active_agents_mask)
        
        if graph_context is not None:
            glimpse_q = context_embedding + graph_context
        else:
            glimpse_q = context_embedding
        # add seq_len dim if not present
        glimpse_q = glimpse_q.unsqueeze(1) if glimpse_q.ndim == 2 else glimpse_q

        return glimpse_q

    def _compute_kvl(self, nodes_dyn_obs, cached):
        glimpse_k, glimpse_v, logit_k = cached
        # Compute dynamic embeddings and add to static embeddings
        if nodes_dyn_obs is not None:
            glimpse_k_dyn, glimpse_v_dyn, logit_k_dyn = self.dynamic_embedding(nodes_dyn_obs)
            glimpse_k = glimpse_k + glimpse_k_dyn
            glimpse_v = glimpse_v + glimpse_v_dyn
            logit_k = logit_k + logit_k_dyn


        return glimpse_k, glimpse_v, logit_k
    
    
    def forward(
        self,
        agents_obs,
        global_obs,
        cur_nodes_idx,
        nodes_dyn_obs,
        node_embeddings,
        agents_embeddings,
        cached,
        active_agents_mask=None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute the logits of the next actions given the current state

        Args:
            cache: Precomputed embeddings
            td: TensorDict with the current environment state
            num_starts: Number of starts for the multi-start decoding
        """
        glimpse_q = self._compute_q(agents_obs, global_obs, 
                                    cur_nodes_idx, node_embeddings, agents_embeddings, 
                                    active_agents_mask=active_agents_mask)
        glimpse_k, glimpse_v, logit_k = self._compute_kvl(nodes_dyn_obs, cached)

        return glimpse_q, glimpse_k, glimpse_v, logit_k 
        

# ---------------------------------------------------------------------------
# PARCO Policy block
# ---------------------------------------------------------------------------

# https://github.com/ai4co/parco/blob/main/parco/models/policy.py


class PolicyNet(nn.Module):
    def __init__(self, nodes_stat_obs_dim,
                       nodes_dyn_obs_dim,
                       agents_obs_dim,
                       global_obs_dim,
                       embed_dim,
                       linear_bias=True,
                       depot_idx: int = 0):
        super(PolicyNet, self).__init__()

        self.init_embed = InitEmbedding(nodes_stat_obs_dim, agents_obs_dim, embed_dim)

        self.encoder = PARCOEncoder(embed_dim=embed_dim)
        self.decoder = PARCODecoder(nodes_dyn_obs_dim=nodes_dyn_obs_dim,
                                    agents_obs_dim=agents_obs_dim,
                                    global_obs_dim=global_obs_dim,
                                    embed_dim=embed_dim)
        
        self.pointer = PointerAttention(
            use_tanh=True,
            C=10,
            embed_dim=128,
            num_heads=8,
            linear_bias=False
        )

        self.project_node_embeddings = nn.Linear(
            embed_dim, 3 * embed_dim, bias=linear_bias
        )

        self._initialize_parameters()
        self.cache = None

    def _initialize_parameters(self):
        for name, param in self.named_parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)

    def _precompute_cache(self, embeddings: Tuple[Tensor, Tensor]):

        (
            glimpse_key_fixed,
            glimpse_val_fixed,
            logit_key,
        ) = self.project_node_embeddings(
            embeddings
        ).chunk(3, dim=-1)

        self.cached_embed = (
            glimpse_key_fixed,
            glimpse_val_fixed,
            logit_key)
        

    def make_cache_(self, nodes_obs, agents_obs):

        n_nodes, n_agents = nodes_obs.shape[1], agents_obs.shape[1]

        init_embeds = self.init_embed(nodes_obs, agents_obs)
        embend, init_embend = self.encoder(init_embeds)

        self.nodes_encoded = embend[:, n_agents:, :]
        self.agents_encoded = embend[:, :n_agents, :]
        self._precompute_cache(self.nodes_encoded)



    def _prev_node_embedding(self, cur_node_idx):
        prev_node_embedding = gather_by_index(self.nodes_encoded, cur_node_idx)
        return prev_node_embedding


    def forward(self, nodes_dyn_obs=None, all_agents_obs=None, global_obs=None, cur_nodes_idx=None, agents_action_mask=None, active_agents_mask=None):

        glimpse_q, glimpse_k, glimpse_v, logit_k = self.decoder(all_agents_obs, 
                                                                global_obs, 
                                                                cur_nodes_idx, 
                                                                nodes_dyn_obs, 
                                                                self.nodes_encoded,
                                                                self.agents_encoded,
                                                                self.cached_embed,
                                                                active_agents_mask=None)

        logits = self.pointer(glimpse_q=glimpse_q, glimpse_k=glimpse_k, glimpse_v=glimpse_v, logit_k=logit_k, mask=agents_action_mask)

        return logits


    def get_action(self, nodes_dyn_obs=None, all_agents_obs=None, global_obs=None, cur_node_idx=None, mask=None, deterministic=False):
        action_logits = self.forward(nodes_dyn_obs, all_agents_obs, global_obs, cur_node_idx, mask)
        probs = torch.distributions.Categorical(logits=action_logits)
        if deterministic:
            return probs.mode
        return probs.sample()


    def get_action_and_logs(self, nodes_dyn_obs=None, all_agents_obs=None, global_obs=None, cur_node_idx=None, mask=None, action=None, deterministic=False):
        
        action_logits = self.forward(nodes_dyn_obs, all_agents_obs, global_obs, cur_node_idx, mask)

        probs = torch.distributions.Categorical(logits=action_logits)

        if action is None:
            if deterministic:
                action = probs.mode
            else:
                action = probs.sample()

        return action, probs.log_prob(action), probs.entropy()
    
    def get_centralized_action_and_logs(
        self, 
        nodes_dyn_obs=None, 
        all_agents_obs=None, 
        global_obs=None,
        cur_node_idx=None, 
        mask=None, 
        action=None, 
        deterministic=False,
        active_agents_mask=None,
    ):
        action_logits = self.forward(nodes_dyn_obs, all_agents_obs, global_obs, cur_node_idx, mask, active_agents_mask)
        
        if action is None:
            action, log_probs, entropy = self.sample_joint_actions_from_L(
                logits=action_logits,
                deterministic=deterministic
            )
        return action, log_probs, entropy

    
    def solve_conflict_actions(self, next_actions, probs, cur_node_idx, solve_method="highest_prob"):

        probs = probs.probs

        if solve_method == "highest_prob":
            action_probs = probs.gather(-1, next_actions.unsqueeze(-1)).squeeze(-1)

            _, order = torch.sort(
                action_probs,
                dim=1,
                descending=True,
                stable=True,
            )

            sorted_actions = next_actions.gather(1, order)

            new_actions = next_actions.clone()

            B, A = sorted_actions.shape

            for b in range(B):
                used = set()

                for k in range(A):
                    agent = order[b, k]
                    action = sorted_actions[b, k].item()

                    # depot nunca gera conflito
                    if action == 0:
                        continue

                    if action in used:
                        new_actions[b, agent] = cur_node_idx[b, agent]
                    else:
                        used.add(action)

        return new_actions
    
    def sample_joint_actions_from_L(self, logits, deterministic):
        """
        Joint action sampling from a score matrix L, following Algorithm 1
        (arXiv:2510.12273): sequentially samples (agent, node) pairs from the
        joint distribution over feasible pairs E, masking the picked agent's row
        and the picked node's column after each draw so no agent or node can be
        reused in a later step -- i.e. conflict-free by construction.

        Args:
            logits:       [B, M, N] score matrix L (B batches, M agents, N nodes)
            deterministic: if True, take the argmax at each step instead of sampling.

        Returns:
            agent_actions:   LongTensor  [B, M] — node index assigned to each agent,
                            indexed by agent position (compatible with env step_all).
            agent_log_probs: FloatTensor [B, M] — log-prob of each agent's action
                            under the masked/renormalized joint distribution,
                            also indexed by agent position.
        """
        B, M, N = logits.shape
        device = logits.device

        L = logits.clone()

        # Buffers in *sampling* order (k = 0..M-1)
        sampled_agents = torch.zeros(B, M, dtype=torch.long,  device=device)
        sampled_nodes  = torch.zeros(B, M, dtype=torch.long,  device=device)
        step_log_probs = torch.zeros(B, M, dtype=torch.float, device=device)
        step_entropies = torch.zeros(B, M, dtype=torch.float, device=device)

        agent_used = torch.zeros(B, M, dtype=torch.bool, device=device)
        node_used  = torch.zeros(B, N, dtype=torch.bool, device=device)

        for k in range(M):
            # Snapshot the masks so autograd holds a stable reference for backward.
            # In-place updates to agent_used / node_used below would otherwise corrupt
            # the mask tensors saved by masked_fill's backward pass.
            agent_used_k = agent_used.clone()
            node_used_k  = node_used.clone()

            # mask already-assigned agents (rows) and nodes (columns)
            step_logits = L.masked_fill(agent_used_k.unsqueeze(-1), float('-inf'))
            step_logits = step_logits.masked_fill(node_used_k.unsqueeze(1), float('-inf'))

            flat_logits = step_logits.reshape(B, M * N)

            # Depot (node 0) is never added to node_used, so every unassigned agent
            # always has at least the depot available — all-inf rows cannot occur.
            dist = torch.distributions.Categorical(logits=flat_logits)

            flat_idx = flat_logits.argmax(dim=-1) if deterministic else dist.sample()
            step_log_probs[:, k] = dist.log_prob(flat_idx)
            step_entropies[:, k] = dist.entropy()

            m_k = flat_idx // N   # which agent was picked at step k
            v_k = flat_idx %  N   # which node was assigned to that agent

            sampled_agents[:, k] = m_k
            sampled_nodes[:, k]  = v_k

            agent_used.scatter_(1, m_k.unsqueeze(1), True)
            node_used.scatter_(1,  v_k.unsqueeze(1), True)
            node_used[:, 0] = False   # depot is always open; multiple agents may return simultaneously

        # Re-index by agent: agent_actions[b, m] = node chosen for agent m
        #                    agent_log_probs[b, m] = log-prob of agent m's action
        #                    agent_entropies[b, m] = entropy of the step where agent m was sampled
        agent_actions   = torch.zeros(B, M, dtype=torch.long,  device=device)
        agent_log_probs = torch.zeros(B, M, dtype=torch.float, device=device)
        agent_entropies = torch.zeros(B, M, dtype=torch.float, device=device)
        agent_actions.scatter_(1,   sampled_agents, sampled_nodes)
        agent_log_probs.scatter_(1, sampled_agents, step_log_probs)
        agent_entropies.scatter_(1, sampled_agents, step_entropies)

        return agent_actions, agent_log_probs, agent_entropies