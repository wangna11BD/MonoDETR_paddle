import paddle
from .transformer import TransformerEncoder, TransformerEncoderLayer

class DepthPredictor(paddle.nn.Layer):

    def __init__(self, model_cfg):
        """
        Initialize depth predictor and depth encoder
        Args:
            model_cfg [EasyDict]: Depth classification network config
        """
        super().__init__()
        depth_num_bins = int(model_cfg['num_depth_bins'])
        depth_min = float(model_cfg['depth_min'])
        depth_max = float(model_cfg['depth_max'])
        self.depth_max = depth_max
        bin_size = 2 * (depth_max - depth_min) / (depth_num_bins * (1 +
            depth_num_bins))
        bin_indice = paddle.linspace(start=0, stop=depth_num_bins - 1, num=
            depth_num_bins)
        bin_value = (bin_indice + 0.5).pow(y=2
            ) * bin_size / 2 - bin_size / 8 + depth_min
        bin_value = paddle.concat(x=[bin_value, paddle.to_tensor([depth_max])],
            axis=0)
        initializer = paddle.nn.initializer.Assign(bin_value)
        self.depth_bin_values = self.create_parameter(bin_value.shape, default_initializer=initializer)
        self.depth_bin_values.stop_gradient = True
        d_model = model_cfg['hidden_dim']
        self.downsample = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels
            =d_model, out_channels=d_model, kernel_size=(3, 3), stride=(2, 
            2), padding=1), paddle.nn.GroupNorm(32, d_model))
        self.proj = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=
            d_model, out_channels=d_model, kernel_size=(1, 1)), paddle.nn.
            GroupNorm(32, d_model))
        self.upsample = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=
            d_model, out_channels=d_model, kernel_size=(1, 1)), paddle.nn.
            GroupNorm(32, d_model))
        self.depth_head = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels
            =d_model, out_channels=d_model, kernel_size=(3, 3), padding=1),
            paddle.nn.GroupNorm(32, num_channels=d_model), paddle.nn.ReLU(),
            paddle.nn.Conv2D(in_channels=d_model, out_channels=d_model,
            kernel_size=(3, 3), padding=1), paddle.nn.GroupNorm(32,
            num_channels=d_model), paddle.nn.ReLU())
        self.depth_classifier = paddle.nn.Conv2D(in_channels=d_model,
            out_channels=depth_num_bins + 1, kernel_size=(1, 1))
        depth_encoder_layer = TransformerEncoderLayer(d_model, nhead=8,
            dim_feedforward=256, dropout=0.1)
        self.depth_encoder = TransformerEncoder(depth_encoder_layer, 1)
        self.depth_pos_embed = paddle.nn.Embedding(int(self.depth_max) + 1, 256)

    def forward(self, feature, mask, pos):
        assert len(feature) == 4
        src_16 = self.proj(feature[1])
        src_32 = self.upsample(paddle.nn.functional.interpolate(x=feature[2
            ], size=src_16.shape[-2:]))
        src_8 = self.downsample(feature[0])
        src = (src_8 + src_16 + src_32) / 3
        src = self.depth_head(src)
        depth_logits = self.depth_classifier(src)
        depth_probs = paddle.nn.functional.softmax(x=depth_logits, axis=1)
        weighted_depth = (depth_probs * self.depth_bin_values.reshape([1, -1,
            1, 1])).sum(axis=1)
        B, C, H, W = src.shape
        src = src.flatten(start_axis=2).transpose(perm=[2, 0, 1])
        mask = mask.cast('int32').flatten(start_axis=1)
        pos = pos.flatten(start_axis=2).transpose(perm=[2, 0, 1])
        depth_embed = self.depth_encoder(src, mask, pos)
        depth_embed = depth_embed.transpose(perm=[1, 2, 0]).reshape([B, C, H, W])
        depth_pos_embed_ip = self.interpolate_depth_embed(weighted_depth)
        depth_embed = depth_embed + depth_pos_embed_ip
        return depth_logits, depth_embed, weighted_depth

    def interpolate_depth_embed(self, depth):
        depth = depth.clip(min=0, max=self.depth_max)
        pos = self.interpolate_1d(depth, self.depth_pos_embed)
        pos = pos.transpose(perm=[0, 3, 1, 2])
        return pos

    def interpolate_1d(self, coord, embed):
        floor_coord = coord.floor()
        delta = (coord - floor_coord).unsqueeze(axis=-1)
        floor_coord = floor_coord.astype(dtype='int64')
        ceil_coord = (floor_coord + 1).clip(max=embed._num_embeddings - 1)
        return embed(floor_coord) * (1 - delta) + embed(ceil_coord) * delta
