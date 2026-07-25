"""Graph encoders."""
import torch
import torch.nn as nn
import manifolds
import layers.hyp_layers as hyp_layers

class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        return output


class HyboNet(Encoder):
    """
    HyboNet.
    """

    def __init__(self, c, args):
        super(HyboNet, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        self.curvatures.append(self.c)
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                hyp_layers.LorentzGraphConvolution(
                    self.manifold, in_dim, out_dim, args.bias, args.dropout, args.use_att, args.local_agg,
                    nonlin=act if i != 0 else None
                )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        return super(HyboNet, self).encode(x, adj)
