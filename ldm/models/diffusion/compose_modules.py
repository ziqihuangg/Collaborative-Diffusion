from abc import abstractmethod
from functools import partial
import math
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from ldm.modules.attention import SpatialTransformer

from ldm.modules.diffusionmodules.openaimodel import *
from omegaconf import OmegaConf

from ldm.models.diffusion.ddpm import DiffusionWrapper
import random

class ComposeUNet(nn.Module):
    """
    One diffusion denoising step (using dynamic diffusers)
    """

    def __init__(self,
        return_confidence_map=False,
        confidence_conditioning_key='crossattn',
        confidence_map_predictor_config=None,
        seg_mask_scale_factor = None,
        seg_mask_schedule = None,
        softmax_twice = False,
        boost_factor = 1.0,
        manual_prob = 1.0,
        return_each_branch = False,
        confidence_input = 'unet_output',
        conditions= ['seg_mask', 'text']
    ):
        super().__init__()

        self.conditions = conditions
        self.return_confidence_map = return_confidence_map

        # define dynamic diffusers
        if 'seg_mask' in self.conditions:
            self.seg_mask_confidence_predictor = DiffusionWrapper(confidence_map_predictor_config, confidence_conditioning_key)
        if 'text' in self.conditions:
            self.text_confidence_predictor = DiffusionWrapper(confidence_map_predictor_config, confidence_conditioning_key)
        if 'sketch' in self.conditions:
            self.sketch_confidence_predictor = DiffusionWrapper(confidence_map_predictor_config, confidence_conditioning_key)

        self.seg_mask_scale_factor = seg_mask_scale_factor #/ (seg_mask_scale_factor + text_scale_factor)
        self.seg_mask_schedule = seg_mask_schedule
        self.softmax_twice = softmax_twice
        self.boost_factor = boost_factor
        self.manual_prob = manual_prob
        self.return_each_branch = return_each_branch

        self.confidence_input = confidence_input

    def set_seg_mask_schedule(self, t):
        t = t[0]
        if self.seg_mask_schedule == None:
            schedule_scale =  1.0
        elif self.seg_mask_schedule == 'linear_decay':
            schedule_scale =   t / 1000
        elif self.seg_mask_schedule == 'cosine_decay':
            pi = torch.acos(torch.zeros(1)).item() * 2
            schedule_scale =  (torch.cos( torch.tensor((1-(t/1000)) * (pi)))+1)/2
        else:
            raise NotImplementedError
        return schedule_scale


    def forward(self, x, t, cond):
        """
        One diffusion denoising step (using dynamic diffusers)

        input:
            - x: noisy image x_t
            - t: timestep
            - cond = {'seg_mask': tensor, 'text': tensor, '...': ...}
        output:
            x_t-1
        """

        # compute individual branch's output using pretrained diffusion models
        if 'seg_mask' in self.conditions:
            seg_mask_unet_output = self.seg_mask_unet(x=x, t=t, c_crossattn=[cond['seg_mask']]) # [B, 3, 64, 64]
        if 'text' in self.conditions:
            text_unet_output = self.text_unet(x=x, t=t, c_crossattn=[cond['text']]) # [B, 3, 64, 64]

        if 'sketch' in self.conditions:
            sketch_unet_output = self.sketch_unet(x=x, t=t, c_crossattn=[cond['sketch']]) # [B, 3, 64, 64]

        # compute influence function for each branch using a dynamic diffuser for each branch
        if self.confidence_input == 'unet_output':
            if 'seg_mask' in self.conditions:
                seg_mask_confidence_map = self.seg_mask_confidence_predictor(x=seg_mask_unet_output, t=t, c_crossattn=[cond['seg_mask']]) # [B, 1, 64, 64]
            if 'text' in self.conditions:
                text_confidence_map = self.text_confidence_predictor(x=text_unet_output, t=t, c_crossattn=[cond['text']])  # [B, 1, 64, 64]
            if 'sketch' in self.conditions:
                sketch_confidence_map = self.sketch_confidence_predictor(x=sketch_unet_output, t=t, c_crossattn=[cond['sketch']])  # [B, 1, 64, 64]
                print('sketch forward')

        elif self.confidence_input == 'x_t':
            if 'seg_mask' in self.conditions:
                seg_mask_confidence_map = self.seg_mask_confidence_predictor(x=x, t=t, c_crossattn=[cond['seg_mask']]) # [B, 1, 64, 64]
            if 'text' in self.conditions:
                text_confidence_map = self.text_confidence_predictor(x=x, t=t, c_crossattn=[cond['text']])  # [B, 1, 64, 64]
            if 'sketch' in self.conditions:
                sketch_confidence_map = self.sketch_confidence_predictor(x=x, t=t, c_crossattn=[cond['sketch']])  # [B, 1, 64, 64]

        else:
            raise NotImplementedError


        # Use softmax to normalize the influence functions across all branches
        if ('seg_mask' in self.conditions) and ('text' in self.conditions) and ('sketch' not in self.conditions):
            concat_map = torch.cat([seg_mask_confidence_map, text_confidence_map], dim=1) # first mask, then text  # [B, 2, 64, 64]
            softmax_map = F.softmax(input=concat_map, dim=1) # [B, 2, 64, 64]
            seg_mask_confidence_map = softmax_map[:,0,:,:].unsqueeze(1)  # [B, 1, 64, 64]
            text_confidence_map = softmax_map[:,1,:,:].unsqueeze(1)   # [B, 1, 64, 64]
        elif ('seg_mask' in self.conditions) and ('text' in self.conditions) and ('sketch' in self.conditions):
            concat_map = torch.cat([seg_mask_confidence_map, text_confidence_map, sketch_confidence_map], dim=1) # first mask, then text  # [B, 3, 64, 64]
            softmax_map = F.softmax(input=concat_map, dim=1) # [B, 3, 64, 64]
            seg_mask_confidence_map = softmax_map[:,0,:,:].unsqueeze(1)  # [B, 1, 64, 64]
            text_confidence_map = softmax_map[:,1,:,:].unsqueeze(1)   # [B, 1, 64, 64]
            sketch_confidence_map = softmax_map[:,2,:,:].unsqueeze(1)   # [B, 1, 64, 64]
        else:
            raise NotImplementedError

        if random.random() <= self.manual_prob:

            if self.seg_mask_schedule is not None:
                seg_mask_schedule_scale = self.set_seg_mask_schedule(t)
                if 'sketch' not in self.conditions:
                    seg_mask_confidence_map = seg_mask_confidence_map * seg_mask_schedule_scale
                    text_confidence_map = 1 - seg_mask_confidence_map
                else:
                    seg_mask_confidence_map = seg_mask_confidence_map * seg_mask_schedule_scale
                    sketch_confidence_map = sketch_confidence_map * seg_mask_schedule_scale
                    text_confidence_map = 1 - seg_mask_confidence_map - sketch_confidence_map

            if self.seg_mask_scale_factor is not None:
                if 'sketch' not in self.conditions:
                    seg_mask_confidence_map = seg_mask_confidence_map * self.seg_mask_scale_factor
                    sum_map = text_confidence_map + seg_mask_confidence_map
                    seg_mask_confidence_map = seg_mask_confidence_map / sum_map
                    text_confidence_map = text_confidence_map / sum_map
                else:
                    seg_mask_confidence_map = seg_mask_confidence_map * self.seg_mask_scale_factor
                    sketch_confidence_map = sketch_confidence_map * self.seg_mask_scale_factor
                    sum_map = text_confidence_map + seg_mask_confidence_map + sketch_confidence_map
                    seg_mask_confidence_map = seg_mask_confidence_map / sum_map
                    text_confidence_map = text_confidence_map / sum_map
                    sketch_confidence_map = sketch_confidence_map / sum_map

            if self.softmax_twice:
                assert ('seg_mask' in self.conditions) and ('text' in self.conditions) and ('sketch' not in self.conditions), "softmax_twice is only implemented for two-modal controls"
                print(f'softmax_twice self.boost_factor={self.boost_factor}')
                concat_map = torch.cat([seg_mask_confidence_map, text_confidence_map], dim=1) * self.boost_factor # first mask, then text  # [B, 2, 64, 64]
                softmax_map = F.softmax(input=concat_map, dim=1) # [B, 2, 64, 64]
                seg_mask_confidence_map = softmax_map[:,0,:,:].unsqueeze(1)  # [B, 1, 64, 64]
                text_confidence_map = softmax_map[:,1,:,:].unsqueeze(1)   # [B, 1, 64, 64]


        # Compute weighted sum of all branch'es output
        if ('seg_mask' in self.conditions) and ('text' in self.conditions) and ('sketch' not in self.conditions):
            seg_mask_weighted = seg_mask_unet_output * seg_mask_confidence_map  # [B, 3, 64, 64]
            text_weighted = text_unet_output * text_confidence_map # [B, 3, 64, 64]
            output = text_weighted + seg_mask_weighted   # [B, 3, 64, 64]

            if self.return_confidence_map:
                if self.return_each_branch:
                    return {'output': output, 'seg_mask_confidence_map': seg_mask_confidence_map, 'text_confidence_map': text_confidence_map, 'text_unet_output': text_unet_output, 'seg_mask_unet_output': seg_mask_unet_output}
                else:
                    return {'output': output, 'seg_mask_confidence_map': seg_mask_confidence_map, 'text_confidence_map': text_confidence_map}
            elif self.return_each_branch:
                    return {'output': output, 'text_unet_output': text_unet_output, 'seg_mask_unet_output': seg_mask_unet_output}
            else:
                return output
        elif ('seg_mask' in self.conditions) and ('text' in self.conditions) and ('sketch' in self.conditions):
            seg_mask_weighted = seg_mask_unet_output * seg_mask_confidence_map  # [B, 3, 64, 64]
            text_weighted = text_unet_output * text_confidence_map # [B, 3, 64, 64]
            sketch_weighted = sketch_unet_output * sketch_confidence_map # [B, 3, 64, 64]
            output = text_weighted + seg_mask_weighted + sketch_weighted  # [B, 3, 64, 64]

            if self.return_confidence_map:
                if self.return_each_branch:
                    return {'output': output, 'seg_mask_confidence_map': seg_mask_confidence_map, 'text_confidence_map': text_confidence_map, 'sketch_confidence_map': sketch_confidence_map,'seg_mask_unet_output': seg_mask_unet_output, 'text_unet_output': text_unet_output, 'sketch_unet_output': sketch_unet_output }
                else:
                    return {'output': output, 'seg_mask_confidence_map': seg_mask_confidence_map, 'text_confidence_map': text_confidence_map, 'sketch_confidence_map': sketch_confidence_map}
            elif self.return_each_branch:
                    return {'output': output, 'text_unet_output': text_unet_output, 'seg_mask_unet_output': seg_mask_unet_output, 'sketch_unet_output': sketch_unet_output}
            else:
                return output


class ComposeCondStageModel(nn.Module):
    """
    Condition Encoder of Multi-Modal Conditions
    """

    def __init__(self,conditions= ['seg_mask', 'text']):
        super().__init__()
        self.conditions = conditions


    def forward(self, input):

        composed_cond = {}

        if 'seg_mask' in self.conditions:
            seg_mask_output = self.seg_mask_cond_stage_model(input['seg_mask'])
            composed_cond['seg_mask'] = seg_mask_output
        if 'text' in self.conditions:
            text_output = self.text_cond_stage_model(input['text'])
            composed_cond['text'] = text_output
        if 'sketch' in self.conditions:
            sketch_output = self.sketch_cond_stage_model(input['sketch'])
            composed_cond['sketch'] = sketch_output

        return composed_cond

    def encode(self, input):
        return self(input)
