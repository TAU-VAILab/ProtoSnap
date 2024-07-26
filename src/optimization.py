from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.dift import device
from src.utils import flatten
from src.geometry import GroupedPoints, GeomTransform
import pandas as pd


class OptimizationLoss(nn.Module):
    def __init__(self, args, data):
        super().__init__()

        self.gpoints = GroupedPoints(data.points, args.img_size, data.labels,
                                     data.connectivity, data.connectivity_weights)

        self.saliency_mask = data.saliency_mask  # H ,W
        self.sim_tensor = data.sim_tensor  # H, W, H, W
        self.sim_res = self.sim_tensor.shape[0]
        assert self.sim_tensor.shape == (self.sim_res, self.sim_res, self.sim_res, self.sim_res), \
            f"Unrecognized shape for DIFT similarity tensor: {self.sim_tensor.shape}"

        self.img_size = args.img_size
        self.lambda_dift = args.lambda_dift
        self.lambda_reg = args.lambda_reg
        self.lambda_reg_global = args.lambda_reg_global
        self.lambda_oob = args.lambda_oob
        self.lambda_mask = args.lambda_mask

        self.n_interpolations = args.n_interpolations

    @staticmethod
    def _sample(sampling_input, warped_points):
        sampling_grid = warped_points.unsqueeze(1).unsqueeze(1)  # N, 1, 1, 2
        sampled = F.grid_sample(sampling_input, sampling_grid, align_corners=True)  # N, 1, 1, 1
        return sampled[:, 0, 0, 0]  # N,
    

    def _sample_sim4d(self, src_points, warped_points, t):
        assert src_points.shape[0] == warped_points.shape[0]
        # ^ shapes: N, 2; both are tensors with coordinates scaled to -1, 1
        N = src_points.shape[0]

        # first sample over target points
        softmax_scaled = (self.sim_tensor / t).view(self.sim_res, self.sim_res, -1).softmax(dim=-1
                                            ).view(self.sim_res, self.sim_res, self.sim_res, self.sim_res)
        trg_sim_sampling_input = softmax_scaled.view(1, -1, self.sim_res, self.sim_res).expand(N, -1, -1, -1)
        # ^ N, H*W, H, W -- dim 1 is treated as channel being sampled
        sampling_grid = warped_points.unsqueeze(1).unsqueeze(1)  # N, 1, 1, 2
        sampled1 = F.grid_sample(trg_sim_sampling_input, sampling_grid, align_corners=True)
        # ^ N, H * W, 1, 1
        sampled1 = sampled1[..., 0, 0].view(N, self.sim_res, self.sim_res)
        # ^ N, H, W

        # then sample over source points
        sampled2 = self._sample(sampled1.unsqueeze(1), warped_points)
        # N, -- similarity values
        return sampled2

    def forward(self, global_transform, stroke_transforms, t):
        src, trg, weights = self.gpoints.get_optimization_points(
            global_transform, stroke_transforms,
            n_interpolations=self.n_interpolations
        )
        # ^ src & trg: N, 2; coordinates scaled to -1, 1; no grad on src
        # src: points on source (font)
        # trg: points on target (warped keypoints)

        N = src.shape[0]
        assert N == trg.shape[0]
        saliency_sampling = self.saliency_mask.unsqueeze(0).expand(N, -1, -1, -1)
        # ^ N, 1, H, W
        sim_loss = -(self._sample_sim4d(src, trg, t) * weights).mean()
        mask_loss = -(self._sample(saliency_sampling, trg) * weights).mean()

        reg_loss = 0.
        for T in stroke_transforms.values():
            reg_loss += T.get_delta_norm()
        reg_loss /= len(stroke_transforms)  # mean

        reg_global_loss = global_transform.get_delta_norm()
        
        oob_loss = (trg.abs().max() - 1.).clip(min=0.)

        loss = sim_loss * self.lambda_dift \
            + mask_loss * self.lambda_mask \
            + reg_loss * self.lambda_reg \
            + reg_global_loss * self.lambda_reg_global \
            + oob_loss * self.lambda_oob
        
        losses_to_plot = {
            "DIFT_similarity": sim_loss.item() * self.lambda_dift,
            "mask": mask_loss.item() * self.lambda_mask,
            "affine": reg_loss.item() * self.lambda_reg,
            "global": reg_global_loss.item() * self.lambda_reg_global,
            "OOB": oob_loss.item() * self.lambda_oob,
            "total": loss.item()
        }

        return loss, losses_to_plot


def optimize(args, data, desc=None, initial_transform=None, rep=None):

    loss_func = OptimizationLoss(args, data)

    unique_labels = {L for L in data.labels}
    global_transform = GeomTransform(initial_transform=initial_transform)
    stroke_transforms = {L: GeomTransform() for L in unique_labels}

    params = global_transform.params() + flatten([
        tr.params() for tr in stroke_transforms.values()])

    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    name = data.name
    if rep is not None:
        name += f'_{rep}'
    if args.output_folder is None:
        output_path = Path("output") / name
    else:
        output_path = Path("output") / args.output_folder / name
    output_path.mkdir(parents=True, exist_ok=True)
    temp = torch.tensor(args.initial_temp, device=device)

    data.save_result(global_transform, stroke_transforms, output_path, itr_num=0)

    all_losses_to_plot = []
    global_transform_norms = []
    for itr in tqdm(range(args.iter_num), desc=desc):
        if itr == 0:
            data.visualize(global_transform, stroke_transforms, itr, output_path)
            # ^ for initial viz, show before optimization step
        optimizer.zero_grad()
        loss, losses_to_plot = loss_func(global_transform, stroke_transforms, temp)
        all_losses_to_plot.append(losses_to_plot)
        loss.backward()
        optimizer.step()
        temp *= args.temp_multiplier
        if itr > 0 and (itr % args.viz_every == 0 or (itr + 1) == args.iter_num):
            data.visualize(global_transform, stroke_transforms, itr, output_path)
        global_transform_norms.append(global_transform.get_delta_norm().item())

    data.save_result(global_transform, stroke_transforms, output_path, itr_num=args.iter_num)
    loss_df = pd.DataFrame(all_losses_to_plot)
    loss_df.to_csv(output_path / "losses.csv", index=False)

    if args.save_losses:
        loss_df.plot()
        plt.savefig(output_path / "losses.png")
        plt.title("All losses")
        plt.close()

        for loss_name in loss_df.columns:
            col = loss_df[loss_name]
            col.plot()
            plt.title(f"{loss_name} loss")
            plt.savefig(output_path / f"loss_{loss_name}.png")
            plt.close()

        pd.Series(global_transform_norms).plot()
        plt.title("Global transform norms")
        plt.savefig(output_path / "global_norms.png")
        plt.close()