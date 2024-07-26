import pandas as pd
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import re
from pathlib import Path
import torch.nn.functional as F
import cv2

from src.utils import get_proto_img, COLORS
from src.geometry import GroupedPoints
from src.dift import DiftWrapper


class Prototype:

    def __init__(self, args, prompt):
        df = pd.read_csv(args.df_filename)
        proto, cs, rs, orig_w, orig_h = get_proto_img(args, prompt, df)

        unicode = df[df['name'] == prompt]['hex'].values[0]
        points_fn = Path(args.con_dir) / f"{unicode}_adf.csv"
        points_df = pd.read_csv(points_fn)
        points_df.x = points_df.x.apply(self.global_adjustment, args=(cs, (args.img_size / orig_w)))
        points_df.y = points_df.y.apply(self.global_adjustment, args=(rs, (args.img_size / orig_h)))
        con_df = pd.read_csv(Path(args.con_dir) / f"{unicode}_con.csv")

        self.points = [(int(row.x), int(row.y)) for _, row in points_df.iterrows()]
        self.connectivity, self.connectivity_weights = \
            self.get_connectivity(points_df, con_df, line_weighting=args.line_weighting)

        self.labels = [int(row.label.split()[1]) for _, row in points_df.iterrows()]
        self.proto = proto

    @staticmethod
    def global_adjustment(value, start, scale, padding=10):
        value -= start
        value += padding
        value *= scale
        return value

    @staticmethod
    def get_connectivity(points_df, con_df, line_weighting=1.0):
        connectivity = [[] for _ in range(len(points_df))]
        connectivity_weights = [[] for _ in range(len(points_df))]
        n_strokes = points_df.label.nunique()
        n_per_stroke_dict = points_df.label.value_counts().to_dict()
        n_per_stroke = [n_per_stroke_dict[f'Stroke {i + 1}'] for i in range(n_strokes)]
        starting_indices = [0] + np.cumsum(n_per_stroke).tolist()[:-1]
        for row in con_df.itertuples():
            label, i, j = row.label, row.i, row.j
            m = re.match(r'Stroke (\d+)', label)
            stroke_idx = int(m.group(1)) - 1
            start_idx = starting_indices[stroke_idx]
            i_ = i + start_idx
            j_ = j + start_idx
            connectivity[i_] += [j_]
            dx = points_df.x.iloc[j_] - points_df.x.iloc[i_]
            dy = points_df.y.iloc[j_] - points_df.y.iloc[i_]
            weight = (dx ** 2 + dy ** 2) ** 0.5
            connectivity_weights[i_] += [weight ** line_weighting]
            # ^ only one direction is needed for losses and visualization
            # (they are symmetric in the indices)
        mean_weight = np.mean([y for x in connectivity_weights for y in x])
        connectivity_weights = [[y / mean_weight for y in x] for x in connectivity_weights]
        return connectivity, connectivity_weights



class Data:
    def __init__(self, args, dift):
        self.prompt = args.prompt
        self.img_size = args.img_size

        target_image_path = Path(args.target_image_path)
        if target_image_path.is_dir():
            target_image_path = target_image_path / f"{self.prompt}.png"
        self.target = self._load_target(target_image_path)
        self.name = f"{target_image_path.stem}_{args.suffix}"

        prototype = Prototype(args, args.prompt)
        self.points, self.connectivity, self.connectivity_weights, self.labels, self.proto = \
            prototype.points, prototype.connectivity, prototype.connectivity_weights, prototype.labels, prototype.proto


        self.scatter_size = args.scatter_size
        self.line_alpha = args.line_alpha

        if dift is not None:
            dift_wrapper = DiftWrapper(args, dift)
            self.sim_tensor = dift_wrapper.get_similaritity_tensor(self.proto, self.target)
            # H, W, H, W
            
            sim_res = self.sim_tensor.shape[0]
            mask = np.uint8(self.proto.resize((sim_res, sim_res)).convert('L')) != 255
            deltas = self.sim_tensor[mask, :, :].mean(dim=0) - self.sim_tensor[~mask, :, :].mean(dim=0)
            deltas -= deltas.min()
            deltas /= deltas.max()
            clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(2, 2))
            eq = clahe.apply(np.uint8(deltas.cpu() * 255)) # histogram-equalized
            eq = eq - eq.mean()
            eq = eq.clip(min=0.)
            eq = eq / eq.max()
            self.saliency_mask = torch.tensor(eq,
                    dtype=torch.float32, device=self.sim_tensor.device) # H, W

    def _load_target(self, target_images_path):
        return Image.open(target_images_path).resize((self.img_size, self.img_size))

    @staticmethod
    def _find_head_center(label_points, label_cons):
        assert len(label_cons) == len(label_points)
        # removing the tail point
        if len(label_cons) == 4:
            con_flatten = label_cons.i.to_list() + label_cons.j.to_list()
            con_counts = {i: con_flatten.count(i) for i in range(len(label_cons))}
            con_counts = sorted(con_counts, key=con_counts.get)
            tail = con_counts[0]
            center = con_counts[-1]
            label_points = label_points.drop(tail)
        else:
            assert len(label_cons) == 3
            center = 0
        base_points = label_points.loc[[i for i in label_points.index if i != center]]
        center = label_points.loc[center]
        mid_base_x, mid_base_y = np.mean(base_points.x), np.mean(base_points.y)
        return np.mean([mid_base_x, center.x]), np.mean([mid_base_y, center.y])

    @torch.no_grad()
    def visualize(self, global_transform, stroke_transforms, itr_num, output_path):

        gpoints = GroupedPoints(self.points, self.img_size, self.labels, self.connectivity)
        src_points, dst_points, dst_points_global, dst_points_init = \
            gpoints.get_viz_points(global_transform, stroke_transforms)
        saliency_mask = self.saliency_mask.cpu()
        # for visualization: needs to be right size
        saliency_mask = F.interpolate(saliency_mask[None, None],
                        size=(self.img_size, self.img_size),
                        mode='bilinear', align_corners=True)[0, 0]

        fig, axs = plt.subplots(3, 3, figsize=(8, 6))
        plt.tight_layout()
        for row in axs:
            for ax in row:
                ax.axis('off')

        axs[0][0].imshow(self.target)
        axs[0][1].imshow(self.proto)
        axs[0][2].imshow(self.proto)
        self.plot_coords(axs[0][2], src_points)

        axs[1][0].imshow(self.target)
        self.plot_coords(axs[1][0], dst_points_init)
        axs[1][0].set_title("initialization")
        axs[1][1].imshow(self.target)
        self.plot_coords(axs[1][1], dst_points_global)
        axs[1][1].set_title("global")
        axs[1][2].imshow(self.target)
        self.plot_coords(axs[1][2], dst_points)
        axs[1][2].set_title("global+local")

        axs[2][0].imshow(saliency_mask)
        self.plot_coords(axs[2][0], dst_points_init)
        axs[2][1].imshow(saliency_mask)
        self.plot_coords(axs[2][1], dst_points_global)
        axs[2][2].imshow(saliency_mask)
        self.plot_coords(axs[2][2], dst_points)

        plt.savefig(output_path / f"itr{itr_num}_viz.png")
        plt.close()

    def plot_coords(self, ax, coords):
        for i, indices in enumerate(self.connectivity):
            label = self.labels[i]
            for j in indices:
                xs = [coords[i, 0], coords[j, 0]]
                ys = [coords[i, 1], coords[j, 1]]
                ax.plot(xs, ys, color=COLORS[(label - 1) % len(COLORS)], alpha=self.line_alpha)
        ax.scatter(coords[:, 0], coords[:, 1],
                   c=[COLORS[(i - 1) % len(COLORS)] for i in self.labels], s=self.scatter_size)
        
    @torch.no_grad()
    def save_result(self, global_transform, stroke_transforms, output_path, itr_num, save_artifacts=True):
        gpoints = GroupedPoints(self.points, self.img_size, self.labels, self.connectivity)
        _, dst_points, dst_points_global, _ = gpoints.get_viz_points(global_transform, stroke_transforms)
        dst_points[:, 1] = self.img_size - dst_points[:, 1]
        dst_points_global[:, 1] = self.img_size - dst_points_global[:, 1]

        def plot_coords(coords, name):
            fig, ax = plt.subplots(1, 1, figsize=(self.img_size / 100, self.img_size / 100))
            ax.set_ylim([0, self.img_size])
            ax.set_xlim([0, self.img_size])
            ax.axis('off')
            plt.tight_layout()
            self.plot_coords(ax, coords)
            plt.savefig(output_path / f"{name}_{itr_num}.png", bbox_inches='tight', pad_inches=0)
            plt.close()

        plot_coords(dst_points, "final_result")
        plot_coords(dst_points_global, "final_result_global")
        if save_artifacts:
            torch.save(dst_points, output_path / f"dst_points_{itr_num}.pt")
            torch.save(self.saliency_mask, output_path / f"saliency_mask_{itr_num}.pt")
            torch.save(self.labels, output_path / f"labels_{itr_num}.pt")
            torch.save(self.connectivity, output_path / f"connectivity_{itr_num}.pt")
            torch.save(global_transform, output_path / f"global_transform_{itr_num}.pt")
            torch.save(stroke_transforms, output_path / f"stroke_transforms_{itr_num}.pt")

