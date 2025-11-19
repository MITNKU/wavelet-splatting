#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import time
from argparse import ArgumentParser
from os import makedirs

import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from wipes_renderer import WIPESModel, render
from scene import Scene
from utils.general_utils import safe_state



def render_set(model_path, name, iteration, views, wipes, pipeline, background):
    render_base = os.path.join(model_path, name, f"ours_{iteration}")
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    # ----------- Inference timing and memory logging ------------
    fps_list = []
    elapsed_list = []
    torch.cuda.reset_peak_memory_stats()

    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        start = time.time()
        
        rendering = render(view, wipes, pipeline, background)["render"]
        elapsed = time.time() - start

        fps = 1 / elapsed if elapsed > 0 else float('inf')
        fps_list.append(fps)
        elapsed_list.append(elapsed)
        
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{idx:05d}.png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, f"{idx:05d}.png"))
        
    avg_elapsed = sum(elapsed_list) / len(elapsed_list) if elapsed_list else 0
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    max_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
    print(f"Inference time: {avg_elapsed:.2f}s, FPS: {avg_fps:.2f}, Max GPU Memory: {max_mem} MB")
    with open(os.path.join(render_base, "inference_metrics.txt"), "w") as f:
        f.write(f"avg_elapsed {elapsed:.4f}\nfps {avg_fps:.2f}\nmax_gpu_mem {max_mem}\n")

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        wipes = WIPESModel(dataset.sh_degree)
        scene = Scene(dataset, wipes, load_iteration=iteration, shuffle=False)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if not skip_train:
            wipes.compute_rate(cameras=scene.getTrainCameras())
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), wipes, pipeline, background)
        
        if not skip_test:
            wipes.compute_rate(cameras=scene.getTestCameras())
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), wipes, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)