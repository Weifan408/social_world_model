import torch
import torch.nn as nn
import torch.nn.functional as F


from zsceval.algorithms.utils.cnn_encoder import CNNEncoder
from zsceval.algorithms.utils.cnn_decoder import CNNDecoder
from zsceval.algorithms.utils.mix import MIXBase
from zsceval.algorithms.utils.mlp import MLP, MLPBase, MLPDecoder
from zsceval.algorithms.utils.util import check


class Encoder(nn.Module):
    def __init__(self, args, obs_shape, hidden=None, device=torch.device("cpu")):
        super(Encoder, self).__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)

        if "Dict" in obs_shape.__class__.__name__:
            self._mixed_obs = True
            self.base = MIXBase(
                args,
                obs_shape,
                cnn_layers_params=args.cnn_layers_params,
            )
        else:
            self._mixed_obs = False
            # MARK: MLPBase will not be used
            self.base = (
                CNNEncoder(
                    args,
                    obs_shape,
                    cnn_layers_params=args.cnn_layers_params,
                )
                if len(obs_shape) == 3
                else MLPBase(
                    args,
                    obs_shape,
                    use_attn_internal=args.use_attn_internal,
                    use_cat_self=True,
                )
            )

        if hidden is not None:
            self.mlp = nn.Linear(
                self.base.output_size,
                hidden
            )
        else:
            self.mlp = None

        self.cnn_output_dim = self.base.cnn_output_dim
        self.output_size = self.base.output_size if hidden is None else hidden

    def forward(self, obs):
        B, N = obs.shape[:2]
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
                # [B, T, H, W, C] -> [B*T, H, W, C]
                obs[key] = obs[key].view(-1, *obs[key].shape[2:])
        else:
            obs = check(obs).to(**self.tpdv)
            obs = obs.view(-1, *obs.shape[2:])

        embed = self.base(obs)
        if self.mlp is not None:
            embed = F.relu(self.mlp(embed))
        embed = embed.view(B, N, -1)
        return embed


class Decoder(nn.Module):
    def __init__(self, args, obs_shape, cnn_output_dim, hidden=None):
        super(Decoder, self).__init__()

        self.pre_decoder = nn.Linear(
            args.wm_feat_size,
            args.wm_model_hidden
        )

        if "Dict" in obs_shape.__class__.__name__:
            raise NotImplementedError(
                "Decoder for Dict observation is not implemented")
        else:
            self._mixed_obs = False
            # MARK: MLPBase will not be used
            self.decoder = (
                CNNDecoder(
                    args=args,
                    obs_shape=obs_shape,
                    cnn_output_dim=cnn_output_dim,
                    hidden=hidden,
                    cnn_layers_params=args.cnn_layers_params
                )
                if len(obs_shape) == 3
                else MLPDecoder(
                    args,
                    obs_shape,
                    use_attn_internal=args.use_attn_internal,
                    use_cat_self=True,
                )
            )

    def forward(self, feat):
        B, N = feat.shape[:2]
        feat = feat.view(-1, *feat.shape[2:])
        i_feat = F.relu(self.pre_decoder(feat))
        x = self.decoder(i_feat)
        x = x.view(B, N, *x.shape[1:])
        i_feat = i_feat.view(B, N, *i_feat.shape[1:])
        return x, i_feat
