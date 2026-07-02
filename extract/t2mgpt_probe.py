"""
Day-one T2M-GPT latent probe. Runs INSIDE the T2M-GPT repo root.

Path: prompt -> CLIP ViT-B/32 feature (512,) -> trans.sample() -> VQ token
indices -> codebook dequantize -> (T', 512) continuous code vectors ->
mean-pool -> latent (512,).

The codebook vectors are the extraction site from the eval plan; unlike
MDM's pooled x0, this latent DECODES (vqvae.forward_decoder), which is what
makes T2M-GPT the testbed for the causal style-transfer arithmetic.

Usage (repo root, GPU):
  python t2mgpt_probe.py --text_prompt "a person is walking"
"""
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
import clip

import models.vqvae as vqvae
import models.t2m_trans as trans

# hyperparameters from the released checkpoints (README eval command)
HP = dict(nb_code=512, code_dim=512, output_emb_width=512, down_t=2,
          stride_t=2, width=512, depth=3, dilation_growth_rate=3,
          embed_dim_gpt=1024, clip_dim=512, block_size=51, num_layers=9,
          n_head_gpt=16, drop_out_rate=0.1, ff_rate=4)


def main():
    p = ArgumentParser()
    p.add_argument("--text_prompt", default="a person is walking")
    p.add_argument("--vq_ckpt", default="pretrained/VQVAE/net_last.pth")
    p.add_argument("--trans_ckpt",
                   default="pretrained/VQTransformer_corruption05/net_best_fid.pth")
    p.add_argument("--out", default="probe_t2mgpt")
    cli = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    args = Namespace(**HP, quantizer="ema_reset", mu=0.99, dataname="t2m")
    net = vqvae.HumanVQVAE(args, args.nb_code, args.code_dim,
                           args.output_emb_width, args.down_t, args.stride_t,
                           args.width, args.depth, args.dilation_growth_rate)
    net.load_state_dict(torch.load(cli.vq_ckpt, map_location="cpu")["net"],
                        strict=True)
    net.eval().to(dev)

    tr = trans.Text2Motion_Transformer(
        num_vq=args.nb_code, embed_dim=args.embed_dim_gpt,
        clip_dim=args.clip_dim, block_size=args.block_size,
        num_layers=args.num_layers, n_head=args.n_head_gpt,
        drop_out_rate=args.drop_out_rate, fc_rate=args.ff_rate)
    tr.load_state_dict(torch.load(cli.trans_ckpt, map_location="cpu")["trans"],
                       strict=True)
    tr.eval().to(dev)

    clip_model, _ = clip.load("ViT-B/32", device=dev, jit=False)
    clip_model.eval()

    with torch.no_grad():
        tokens = clip.tokenize([cli.text_prompt], truncate=True).to(dev)
        feat = clip_model.encode_text(tokens).float()          # (1, 512)
        print(f"[1] CLIP feature: {tuple(feat.shape)}")

        idx = tr.sample(feat, if_categorial=False)             # (1, T')
        print(f"[2] VQ token indices: {tuple(idx.shape)}")

        codes = net.vqvae.quantizer.dequantize(idx)            # (1, T', 512)
        print(f"[3] codebook vectors: {tuple(codes.shape)}")

        latent = codes[0].mean(dim=0).cpu().numpy()            # (512,)
        print(f"[4] pooled latent: {latent.shape}")

        # decode check — this is the path the transfer test will use
        x = codes.permute(0, 2, 1)                             # (1, 512, T')
        motion = net.vqvae.decoder(x)
        print(f"[5] decoded motion: {tuple(motion.shape)}")

    np.save(f"{cli.out}_latent.npy", latent)
    np.save(f"{cli.out}_tokens.npy", idx[0].cpu().numpy())
    print("T2M-GPT probe complete — extraction AND decode paths verified.")


if __name__ == "__main__":
    main()
