import argparse
import copy
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
"""
Reference code:
https://github.com/justinpinkney/stable-diffusion/blob/main/notebooks/imagic.ipynb
"""

parser = argparse.ArgumentParser()

# directories
parser.add_argument('--config', type=str, default='configs/256_text.yaml')
parser.add_argument('--ckpt', type=str, default='pretrained/256_text.ckpt')
parser.add_argument('--save_folder', type=str, default='outputs/text_edit')
parser.add_argument(
    '--input_image_path',
    type=str,
    default='test_data/test_mask_edit/256_input_image/27044.jpg')
parser.add_argument(
    '--text_prompt',
    type=str,
    default='He is a teen. The face is covered with short pointed beard.')

# hyperparameters
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--stage1_lr', type=float, default=0.001)
parser.add_argument('--stage1_num_iter', type=int, default=500)
parser.add_argument('--stage2_lr', type=float, default=1e-6)
parser.add_argument('--stage2_num_iter', type=int, default=1000)
parser.add_argument(
    '--alpha_list',
    type=str,
    default='-1, -0.5, 0, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5')
parser.add_argument('--set_random_seed', type=bool, default=False)
parser.add_argument('--save_checkpoint', type=bool, default=True)

args = parser.parse_args()


def load_model_from_config(config, ckpt, device="cpu", verbose=False):
    """Loads a model from config and a ckpt
    if config is a path will use omegaconf to load
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    pl_sd = torch.load(ckpt, map_location="cpu")
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model


@torch.no_grad()
def sample_model(model,
                 sampler,
                 c,
                 h,
                 w,
                 ddim_steps,
                 scale,
                 ddim_eta,
                 start_code=None,
                 n_samples=1):
    """Sample the model"""
    uc = None
    if scale != 1.0:
        uc = model.get_learned_conditioning(n_samples * [""])

    # print(f'model.model.parameters(): {model.model.parameters()}')
    # for name, param in model.model.named_parameters():
    #     if param.requires_grad:
    #         print (name, param.data)
    #         break

    # print(f'unconditional_guidance_scale: {scale}') # 1.0
    # print(f'unconditional_conditioning: {uc}') # None
    with model.ema_scope("Plotting"):

        shape = [3, 64, 64]  # [4, h // 8, w // 8]
        samples_ddim, _ = sampler.sample(
            S=ddim_steps,
            conditioning=c,
            batch_size=n_samples,
            shape=shape,
            verbose=False,
            start_code=start_code,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc,
            eta=ddim_eta,
        )
        return samples_ddim


def load_img(path, target_size=256):
    """Load an image, resize and output -1..1"""
    image = Image.open(path).convert("RGB")

    tform = transforms.Compose([
        # transforms.Resize(target_size),
        # transforms.CenterCrop(target_size),
        transforms.ToTensor(),
    ])
    image = tform(image)
    return 2. * image - 1.


def decode_to_im(samples, n_samples=1, nrow=1):
    """Decode a latent and return PIL image"""
    samples = model.decode_first_stage(samples)
    ims = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)
    x_sample = 255. * rearrange(
        ims.cpu().numpy(),
        '(n1 n2) c h w -> (n1 h) (n2 w) c',
        n1=n_samples // nrow,
        n2=nrow)
    return Image.fromarray(x_sample.astype(np.uint8))


if __name__ == '__main__':

    args.alpha_list = [float(i) for i in args.alpha_list.split(',')]

    device = "cuda"  # "cuda:0"

    # Generation parameters
    scale = 1.0
    h = 256
    w = 256
    ddim_steps = 50
    ddim_eta = 1.0

    # initialize model
    global_model = load_model_from_config(args.config, args.ckpt, device)

    input_image = args.input_image_path
    image_name = input_image.split('/')[-1]

    prompt = args.text_prompt

    torch.manual_seed(args.seed)

    model = copy.deepcopy(global_model)
    sampler = DDIMSampler(model)

    # prepare directories
    save_dir = os.path.join(args.save_folder, image_name, str(prompt))
    os.makedirs(save_dir, exist_ok=True)
    print(
        f'================================================================================'
    )
    print(f'input_image: {input_image} | text: {prompt}')

    # read input image
    init_image = load_img(input_image).to(device).unsqueeze(
        0)  # [1, 3, 256, 256]
    gaussian_distribution = model.encode_first_stage(init_image)
    init_latent = model.get_first_stage_encoding(
        gaussian_distribution)  # [1, 3, 64, 64]
    img = decode_to_im(init_latent)
    img.save(os.path.join(save_dir, 'input_image_reconstructed.png'))

    # obtain text embedding
    emb_tgt = model.get_learned_conditioning([prompt])
    emb_ = emb_tgt.clone()
    torch.save(emb_, os.path.join(save_dir, 'emb_tgt.pt'))
    emb = torch.load(os.path.join(save_dir, 'emb_tgt.pt'))  # [1, 77, 640]

    # Sample the model with a fixed code to see what it looks like
    quick_sample = lambda x, s, code: decode_to_im(
        sample_model(
            model, sampler, x, h, w, ddim_steps, s, ddim_eta, start_code=code))
    # start_code = torch.randn_like(init_latent)
    start_code = torch.randn((1, 3, 64, 64), device=device)
    torch.save(start_code, os.path.join(save_dir,
                                        'start_code.pt'))  # [1, 3, 64, 64]
    torch.manual_seed(args.seed)
    img = quick_sample(emb_tgt, scale, start_code)
    img.save(os.path.join(save_dir, 'A_start_tgtText_origDM.png'))

    # ======================= (A) Text Embedding Optimization ===================================
    print('########### Step 1 - Optimise the embedding ###########')
    emb.requires_grad = True
    opt = torch.optim.Adam([emb], lr=args.stage1_lr)
    criteria = torch.nn.MSELoss()
    history = []

    pbar = tqdm(range(args.stage1_num_iter))
    for i in pbar:
        opt.zero_grad()

        if args.set_random_seed:
            torch.seed()
        noise = torch.randn_like(init_latent)
        t_enc = torch.randint(1000, (1, ), device=device)
        z = model.q_sample(init_latent, t_enc, noise=noise)

        pred_noise = model.apply_model(z, t_enc, emb)

        loss = criteria(pred_noise, noise)
        loss.backward()
        pbar.set_postfix({"loss": loss.item()})
        history.append(loss.item())
        opt.step()

    plt.plot(history)
    plt.show()
    torch.save(emb, os.path.join(save_dir, 'emb_opt.pt'))
    emb_opt = torch.load(os.path.join(save_dir, 'emb_opt.pt'))  # [1, 77, 640]

    torch.manual_seed(args.seed)
    img = quick_sample(emb_opt, scale, start_code)
    img.save(os.path.join(save_dir, 'A_end_optText_origDM.png'))

    # Interpolate the embedding
    for idx, alpha in enumerate(args.alpha_list):
        print(f'alpha={alpha}')
        new_emb = alpha * emb_tgt + (1 - alpha) * emb_opt
        torch.manual_seed(args.seed)
        img = quick_sample(new_emb, scale, start_code)
        img.save(
            os.path.join(
                save_dir,
                f'0A_interText_origDM_{idx}_alpha={round(alpha,3)}.png'))

    # ======================= (B) Model Fine-Tuning ===================================
    print('########### Step 2 - Fine tune the model ###########')
    emb_opt.requires_grad = False
    model.train()

    opt = torch.optim.Adam(model.model.parameters(), lr=args.stage2_lr)
    criteria = torch.nn.MSELoss()
    history = []

    pbar = tqdm(range(args.stage2_num_iter))
    for i in pbar:
        opt.zero_grad()

        if args.set_random_seed:
            torch.seed()
        noise = torch.randn_like(init_latent)
        t_enc = torch.randint(model.num_timesteps, (1, ), device=device)
        z = model.q_sample(init_latent, t_enc, noise=noise)

        pred_noise = model.apply_model(z, t_enc, emb_opt)

        loss = criteria(pred_noise, noise)
        loss.backward()
        pbar.set_postfix({"loss": loss.item()})
        history.append(loss.item())
        opt.step()

    model.eval()
    plt.plot(history)
    plt.show()
    torch.manual_seed(args.seed)
    img = quick_sample(emb_opt, scale, start_code)
    img.save(os.path.join(save_dir, 'B_end_optText_optDM.png'))
    # Should look like the original image

    if args.save_checkpoint:
        ckpt = {
            "state_dict": model.state_dict(),
        }
        ckpt_path = os.path.join(save_dir, 'optDM.ckpt')
        print(f'Saving optDM to {ckpt_path}')
        torch.save(ckpt, ckpt_path)

    # ======================= (C) Generation ===================================
    print('########### Step 3 - Generate images ###########')
    # Interpolate the embedding
    for idx, alpha in enumerate(args.alpha_list):
        print(f'alpha={alpha}')
        new_emb = alpha * emb_tgt + (1 - alpha) * emb_opt
        torch.manual_seed(args.seed)
        img = quick_sample(new_emb, scale, start_code)
        img.save(
            os.path.join(
                save_dir,
                f'0C_interText_optDM_{idx}_alpha={round(alpha,3)}.png'))

    print('Done')