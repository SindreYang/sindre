
import torch
import torch.nn as nn
from sindre.ai.attention_utils.models.decoder import CrossAttentionDecoder
from sindre.ai.attention_utils.models.transformer import Transformer
from sindre.ai.embedder import FourierEmbedder
from sindre.ai.implicit_utils.utils import Latents2Meshes


class ShapeVAE(nn.Module):
    def __init__(
            self,
            *,
            num_latents: int,
            embed_dim: int,
            width: int,
            heads: int,
            num_decoder_layers: int,
            num_freqs: int = 8,
            include_pi: bool = True,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            label_type: str = "binary",
            drop_path_rate: float = 0.0,
            scale_factor: float = 1.0,
    ):
        super().__init__()
        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)

        self.post_kl = nn.Linear(embed_dim, width)

        self.transformer = Transformer(
            n_ctx=num_latents,
            width=width,
            layers=num_decoder_layers,
            heads=heads,
            qkv_bias=qkv_bias,
            norm_method= "LayerNorm",
            atten_method = "SDPA",
            drop_path_rate=drop_path_rate
        )

        self.geo_decoder = CrossAttentionDecoder(
            fourier_embedder=self.fourier_embedder,
            out_channels=1,
            width=width,
            heads=heads,
            qkv_bias=qkv_bias,
            norm_method= "LayerNorm",
            atten_method = "SDPA",
        )

        self.scale_factor = scale_factor
        self.latent_shape = (num_latents, embed_dim)

    def forward(self, latents):
        latents = self.post_kl(latents)
        latents = self.transformer(latents)
        return latents



if __name__ =="__main__":
    num_latents = 16
    embed_dim = 32
    width = 64
    heads = 4
    num_decoder_layers = 2
    # 初始化模型
    model = ShapeVAE(
        num_latents=num_latents,
        embed_dim=embed_dim,
        width=width,
        heads=heads,
        num_decoder_layers=num_decoder_layers,
    )
    # 推理
    batch_size = 2
    latents = torch.randn(batch_size, num_latents, embed_dim)
    output = model.forward(latents)
    print(f"input.shape:{latents.shape} \noutput.shape: {output.shape}")

    #重建为网格
    Latents2Meshes=Latents2Meshes()
    output_meshes= Latents2Meshes.run(latents=output,fun_callback=model.geo_decoder)
    print(output_meshes)



