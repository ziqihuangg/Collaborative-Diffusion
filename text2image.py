import argparse
import os
import shutil

import numpy as np
import torch
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from torchvision.utils import make_grid

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

"""
Inference script for text-to-face generation at 512x512 resolution
"""


def parse_args():

    parser = argparse.ArgumentParser(description="")

    # conditions
    parser.add_argument(
        "--input_text",
        type=str,
        default="This man has beard of medium length. He is in his thirties.",
        help="text condition")

    # paths
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/512_text.yaml",
        help="path to model config")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="pretrained/512_text.ckpt",
        help="path to model checkpoint")
    parser.add_argument(
        "--save_folder",
        type=str,
        default="outputs/512_text2image",
        help="folder to save synthesis outputs")

    # batch size and ddim steps
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="number of images to generate")
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default="50",
        help=
        "number of ddim steps (between 20 to 1000, the larger the slower but better quality)"
    )

    # whether save intermediate outputs
    parser.add_argument(
        "--save_z",
        type=bool,
        default=False,
        help=
        "whether visualize the VAE latent codes and save them in the output folder",
    )
    parser.add_argument(
        "--return_influence_function",
        type=bool,
        default=False,
        help=
        "whether visualize the Influence Functions and save them in the output folder",
    )
    parser.add_argument(
        "--display_x_inter",
        type=bool,
        default=False,
        help=
        "whether display the intermediate DDIM outputs (x_t and pred_x_0) and save them in the output folder",
    )

    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    # ========== set up model ==========
    print(f'Set up model')
    config = OmegaConf.load(args.config_path)
    model_config = config['model']
    model = instantiate_from_config(model_config)
    model.init_from_ckpt(args.ckpt_path)
    model = model.cuda()
    model.eval()

    # ========== set output directory ==========
    os.makedirs(args.save_folder, exist_ok=True)
    # save a copy of this python script being used
    # shutil.copyfile(__file__, os.path.join(args.save_folder, __file__))

    print(
        f'================================================================================'
    )
    print(f'text: {args.input_text}')

    # prepare directories
    save_sub_folder = os.path.join(args.save_folder, str(args.input_text))
    os.makedirs(save_sub_folder, exist_ok=True)

    # ========== inference ==========
    with torch.no_grad():

        # encode condition
        condition = []
        for i in range(args.batch_size):
            condition.append(args.input_text.lower())

        with model.ema_scope("Plotting"):

            # encode condition
            condition = model.get_learned_conditioning(
                condition)  # [1, 77, 640]
            print(f'condition.shape={condition.shape}')  # [B, 77, 640]

            # DDIM sampling
            ddim_sampler = DDIMSampler(model)
            z_0_batch, intermediates = ddim_sampler.sample(
                S=args.ddim_steps,
                batch_size=args.batch_size,
                shape=(3, 64, 64),
                conditioning=condition,
                verbose=False,
                eta=1.0,
                log_every_t=1)

        # decode VAE latent z_0 to image x_0
        x_0_batch = model.decode_first_stage(z_0_batch)  # [B, 3, 256, 256]

    # ========== save outputs ==========
    for idx in range(args.batch_size):

        # ========== save synthesized image x_0 ==========
        save_x_0_path = os.path.join(save_sub_folder,
                                     f'{str(idx).zfill(6)}_x_0.png')
        x_0 = x_0_batch[idx, :, :, :].unsqueeze(0)  # [1, 3, 256, 256]
        x_0 = x_0.permute(0, 2, 3, 1).to('cpu').numpy()
        x_0 = (x_0 + 1.0) * 127.5
        np.clip(x_0, 0, 255, out=x_0)  # clip to range 0 to 255
        x_0 = x_0.astype(np.uint8)
        x_0 = Image.fromarray(x_0[0])
        x_0.save(save_x_0_path)

        # save intermediate x_t and pred_x_0
        if args.display_x_inter:
            for cond_name in ['x_inter', 'pred_x0']:
                save_conf_path = os.path.join(
                    save_sub_folder, f'{str(idx).zfill(6)}_{cond_name}.png')
                conf = intermediates[f'{cond_name}']
                conf = torch.stack(conf, dim=0)  # 50x8x3x64x64
                conf = conf[:, idx, :, :, :]  #  50x3x64x64
                print('decoding x_inter ......')
                conf = model.decode_first_stage(conf)  # [50, 3, 256, 256]
                conf = make_grid(
                    conf, nrow=10)  # 10 images per row # [3, 256x3, 256x10]
                conf = conf.permute(1, 2,
                                    0).to('cpu').numpy()  # cxhxh -> hxhxc
                conf = (conf + 1.0) * 127.5
                np.clip(conf, 0, 255, out=conf)  # clip to range 0 to 255
                conf = conf.astype(np.uint8)
                conf = Image.fromarray(conf)
                conf.save(save_conf_path)

        # save latent z_0
        if args.save_z:
            save_z_0_path = os.path.join(save_sub_folder,
                                         f'{str(idx).zfill(6)}_z_0.png')
            z_0 = z_0_batch[idx, :, :, :].unsqueeze(0)  # [1, 3, 64, 64]
            z_0 = z_0.permute(0, 2, 3, 1).to('cpu').numpy()
            z_0 = (z_0 + 40) * 4  # manually tuned denormalization
            np.clip(z_0, 0, 255, out=z_0)  # clip to range 0 to 255
            z_0 = z_0.astype(np.uint8)
            z_0 = Image.fromarray(z_0[0])
            z_0.save(save_z_0_path)


if __name__ == "__main__":
    main()