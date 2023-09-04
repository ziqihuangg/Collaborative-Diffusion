import argparse
import copy
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision.utils import make_grid

from ldm.models.diffusion.ddim_confidence import DDIMConfidenceSampler
from ldm.util import instantiate_from_config
"""
Inference script for multi-modal-driven face editing at 256x256 resolution
"""


def parse_args():

    parser = argparse.ArgumentParser(description="")

    # uni-modal editing results
    parser.add_argument(
        "--imagic_text_folder",
        type=str,
        default=
        "outputs/text_edit/27044.jpg/He is a teen. The face is covered with short pointed beard.",
        help="path to Imagic text-based editing results")
    parser.add_argument(
        "--imagic_mask_folder",
        type=str,
        default=
        "outputs/mask_edit/27044.jpg/27044_0_remove_smile_and_rings.png",
        help="path to Imagic mask-based editing results")

    # paths
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/256_codiff_mask_text.yaml",
        help="path to model config")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="pretrained/256_codiff_mask_text.ckpt",
        help="path to model checkpoint")
    parser.add_argument(
        "--save_folder",
        type=str,
        default="outputs/collaborative_edit",
        help="folder to save editing outputs")

    # batch size and ddim steps
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="number of images to generate")
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help=
        "number of ddim steps (between 20 to 1000, the larger the slower but better quality)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2,
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

    # ========== set output directory ==========
    os.makedirs(args.save_folder, exist_ok=True)

    # ========== init model ==========
    config = OmegaConf.load(args.config_path)
    model_config = config['model']
    model_config['params']['seg_mask_ldm_ckpt_path'] = os.path.join(
        args.imagic_mask_folder, 'optDM.ckpt')
    model_config['params']['text_ldm_ckpt_path'] = os.path.join(
        args.imagic_text_folder, 'optDM.ckpt')
    model = instantiate_from_config(model_config)
    model.init_from_ckpt(args.ckpt_path)
    mask_optDM_ckpt = torch.load(
        os.path.join(args.imagic_mask_folder, 'optDM.ckpt'))
    text_optDM_ckpt = torch.load(
        os.path.join(args.imagic_text_folder, 'optDM.ckpt'))

    print('Updating text and mask model to the finetuned version ......')
    state_dict = model.state_dict()
    for name in state_dict.keys():
        if 'model.seg_mask_unet.' in name:
            name_end = name[20:]
            state_dict[name] = mask_optDM_ckpt['state_dict'][
                f'model.{name_end}']
        elif 'model.text_unet.' in name:
            name_end = name[16:]
            state_dict[name] = text_optDM_ckpt['state_dict'][
                f'model.{name_end}']
    model.load_state_dict(state_dict)

    print('Pushing model to CUDA ......')
    global_model = model.cuda()
    global_model.eval()

    seed = args.seed

    for alpha_idx, alpha in enumerate([
            0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1, 1.125, 1.25,
            1.375, 1.5, 1.625, 1.75, 1.875, 2.0, 2.5
    ]):

        print(f'alpha={alpha}')

        seed = int(seed)
        torch.manual_seed(seed)
        mask_start_code = torch.load(
            os.path.join(args.imagic_mask_folder, 'start_code.pt'))
        text_start_code = torch.load(
            os.path.join(args.imagic_text_folder, 'start_code.pt'))
        start_code = mask_start_code

        # prepare directories
        save_sub_folder = os.path.join(args.save_folder, f'seed={seed}')
        os.makedirs(save_sub_folder, exist_ok=True)

        seed = int(seed)
        model = copy.deepcopy(global_model)

        # ========== inference ==========
        with torch.no_grad():

            with model.ema_scope("Plotting"):

                condition = {}

                mask_alpha = alpha
                mask_emb_tgt = torch.load(
                    os.path.join(args.imagic_mask_folder, 'emb_tgt.pt'))
                mask_emb_opt = torch.load(
                    os.path.join(args.imagic_mask_folder, 'emb_opt.pt'))
                mask_new_emb = mask_alpha * mask_emb_tgt + (
                    1 - mask_alpha) * mask_emb_opt
                condition['seg_mask'] = mask_new_emb.repeat(
                    args.batch_size, 1, 1)

                text_alpha = alpha
                text_emb_tgt = torch.load(
                    os.path.join(args.imagic_text_folder, 'emb_tgt.pt'))
                text_emb_opt = torch.load(
                    os.path.join(args.imagic_text_folder, 'emb_opt.pt'))
                text_new_emb = text_alpha * text_emb_tgt + (
                    1 - text_alpha) * text_emb_opt
                condition['text'] = text_new_emb.repeat(args.batch_size, 1, 1)

                torch.manual_seed(seed)

                ddim_sampler = DDIMConfidenceSampler(
                    model=model,
                    return_confidence_map=args.return_influence_function)

                torch.manual_seed(seed)

                z_0_batch, intermediates = ddim_sampler.sample(
                    S=args.ddim_steps,
                    batch_size=args.batch_size,
                    shape=(3, 64, 64),
                    conditioning=condition,
                    verbose=False,
                    start_code=start_code,
                    eta=1.0,
                    log_every_t=1)

            # decode VAE latent z_0 to image x_0
            x_0_batch = model.decode_first_stage(z_0_batch)  # [B, 3, 256, 256]

        # ========== save outputs ==========
        for idx in range(args.batch_size):

            # save synthesized image x_0
            save_x_0_path = os.path.join(
                save_sub_folder,
                f'{str(idx).zfill(6)}_x_0_{alpha_idx}_alpha={round(alpha, 3)}.png'
            )
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
                        save_sub_folder,
                        f'{str(idx).zfill(6)}_{cond_name}.png')
                    conf = intermediates[f'{cond_name}']
                    conf = torch.stack(conf, dim=0)  # 50x8x3x64x64
                    conf = conf[:, idx, :, :, :]  #  50x3x64x64
                    print('decoding x_inter ......')
                    conf = model.decode_first_stage(conf)  # [50, 3, 256, 256]
                    conf = make_grid(
                        conf,
                        nrow=10)  # 10 images per row # [3, 256x3, 256x10]
                    conf = conf.permute(1, 2,
                                        0).to('cpu').numpy()  # cxhxh -> hxhxc
                    conf = (conf + 1.0) * 127.5
                    np.clip(conf, 0, 255, out=conf)  # clip to range 0 to 255
                    conf = conf.astype(np.uint8)
                    conf = Image.fromarray(conf)
                    conf.save(save_conf_path)

            # save influence functions
            if args.return_influence_function:
                for cond_name in ['seg_mask', 'text']:
                    save_conf_path = os.path.join(
                        save_sub_folder,
                        f'{str(idx).zfill(6)}_{cond_name}_influence_function.png'
                    )
                    conf = intermediates[f'{cond_name}_confidence_map']
                    conf = torch.stack(conf, dim=0)  # 50x8x1x64x64
                    conf = conf[:, idx, :, :, :]  #  50x1x64x64
                    conf = torch.cat(
                        [conf, conf, conf],
                        dim=1)  # manually create 3 channels  # [50, 3, 64, 64]
                    conf = make_grid(
                        conf, nrow=10)  # 10 images per row # [3, 332, 662]
                    conf = conf.permute(1, 2,
                                        0).to('cpu').numpy()  # cxhxh -> hxhxc
                    conf = conf * 255  # manually tuned denormalization: [0,1] -> [0,255]
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