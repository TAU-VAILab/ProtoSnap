import torch
import numpy as np
from src.dift import device
from collections import namedtuple


class Points:
    def __init__(self, raw_points, img_size):
        self.raw_points = torch.tensor(raw_points)  # N, 2
        # raw pixel coordinates, on CPU
        self.img_size = img_size
    
        self.projective_points = torch.tensor([
            (self._scale(x), self._scale(y), 1.)
            for x, y in self.raw_points
        ], device=device)  # N, 3

    def _scale(self, coord):
        return (coord / (self.img_size - 1)) * 2 - 1
    
    def _unscale(self, coord):
        return (self.img_size - 1) * (coord + 1) / 2
    
    def get_stacked_scaled_points(self):
        return torch.tensor([
            (self._scale(x), self._scale(y))
            for x, y in self.raw_points
        ], device=device)  # N, 2
    
    def apply_transforms(self, transforms, unscale=False, do_init=False):
        warped_projective = self.projective_points
        for T in transforms:
            M = T.init if do_init else T.build_matrix()
            warped_projective = warped_projective @ M.T
        # ^ N, 3 (last of three coordinates is projective z)
        z = warped_projective[:, -1].unsqueeze(-1) # N, 1
        warped_points = warped_projective[:, :2] / z # N, 2
        return self._unscale(warped_points) if unscale else warped_points


SrcTrgPair = namedtuple("SrcTrgPair", "src trg trg_global trg_init")

class GroupedPoints:
    def __init__(self, raw_points, img_size, labels, connectivity, connectivity_weights=None):

        unique_labels = set(labels)

        global_idx2stroke_idx = {}
        stroke_counts = {label: 0 for label in unique_labels}
        for i in range(len(raw_points)):
            label = labels[i]
            global_idx2stroke_idx[i] = stroke_counts[label]
            stroke_counts[label] += 1
        self.grouped_points = {
            ul: Points(
                [p for p, L in zip(raw_points, labels) if L == ul],
                img_size,
            ) for ul in unique_labels
        }

        self.img_size = img_size
        self.connectivity = connectivity
        self.connectivity_weights = connectivity_weights
    
    def _scale(self, coord):
        return (coord / (self.img_size - 1)) * 2 - 1
    
    def _unscale(self, coord):
        return (self.img_size - 1) * (coord + 1) / 2
    
    def get_stacked_raw_points(self):
        return torch.vstack([v.raw_points for v in self.grouped_points.values()])

    def _apply_transforms(self, global_transform, stroke_transforms, do_global=False, do_init=False):
        return {
            L: SrcTrgPair(
                p.get_stacked_scaled_points(),
                p.apply_transforms([global_transform, stroke_transforms[L]]),
                p.apply_transforms([global_transform]) if do_global else None,
                p.apply_transforms([global_transform], do_init=True) if do_init else None
            ) for L, p in self.grouped_points.items()
        }
    
    def get_viz_points(self, global_transform, stroke_transforms):
        pairs = self._apply_transforms(global_transform, stroke_transforms, do_global=True, do_init=True)
        labels = list(pairs.keys()) # list so order is guaranteed to be consistent
        src = torch.vstack([pairs[L].src for L in labels])
        trg = torch.vstack([pairs[L].trg for L in labels])
        trg_global = torch.vstack([pairs[L].trg_global for L in labels])
        trg_init = torch.vstack([pairs[L].trg_init for L in labels])
        return self._unscale(src).cpu(), self._unscale(trg).cpu(), \
            self._unscale(trg_global).cpu(), self._unscale(trg_init).cpu()
    
    def get_optimization_points(self, global_transform, stroke_transforms,
                                n_interpolations=0):

        pairs = self._apply_transforms(global_transform, stroke_transforms)
        labels = list(pairs.keys()) # list so order is guaranteed to be consistent
        src = torch.vstack([pairs[L].src for L in labels]) # N, 2
        trg = torch.vstack([pairs[L].trg for L in labels]) # N, 2
        weights = [1. for _  in range(src.shape[0])]

        if n_interpolations > 0:
            # TODO: add check that connectivity still has correct indices (assumes labels are sorted)
            src_interp = []
            trg_interp = []
            for i, indices in enumerate(self.connectivity):
                for k, j in enumerate(indices):
                    for _ in range(n_interpolations):
                        c = np.random.random()
                        # interpolation coefficients: uniform in [0, 1]
                        src_interp.append(src[i] * c + src[j] * (1 - c))
                        trg_interp.append(trg[i] * c + trg[j] * (1 - c))
                        weights.append(self.connectivity_weights[i][k])
            
            src_interp = torch.vstack(src_interp) # N', 2
            trg_interp = torch.vstack(trg_interp) # N', 2

            src = torch.vstack([src, src_interp]) # N'', 2
            trg = torch.vstack([trg, trg_interp]) # N'', 2
        
        weights = torch.tensor(weights, device=device)

        return src, trg, weights


class GeomTransform:
    def __init__(self, initial_transform=None):
        self.a = torch.tensor(0., requires_grad=True, device=device)
        self.b = torch.tensor(0., requires_grad=True, device=device)
        self.c = torch.tensor(0., requires_grad=True, device=device)
        self.d = torch.tensor(0., requires_grad=True, device=device)
        self.e = torch.tensor(0., requires_grad=True, device=device)
        self.f = torch.tensor(0., requires_grad=True, device=device)
        self.tx = torch.tensor(0., requires_grad=True, device=device)
        self.ty = torch.tensor(0., requires_grad=True, device=device)

        self.init = torch.eye(3, device=device) if initial_transform is None \
            else torch.tensor(initial_transform, device=device, dtype=torch.float32)
    
    def params(self):
        return [self.a, self.b, self.c, self.d, self.e, self.f, self.tx, self.ty]
    
    def build_matrix(self):
        return self.init + (
            torch.tensor([[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]], device=device) * self.a
            + torch.tensor([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]], device=device) * self.b
            + torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 0., 0.]], device=device) * self.c
            + torch.tensor([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]], device=device) * self.d
            + torch.tensor([[0., 0., 0.], [0., 0., 0.], [1., 0., 0.]], device=device) * self.e
            + torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 1., 0.]], device=device) * self.f
            + torch.tensor([[0., 0., 1.], [0., 0., 0.], [0., 0., 0.]], device=device) * self.tx
            + torch.tensor([[0., 0., 0.], [0., 0., 1.], [0., 0., 0.]], device=device) * self.ty
        )
    
    def get_delta_norm(self):
        return self.a.abs() + self.b.abs() + self.c.abs() \
            + self.d.abs() + self.e.abs() + self.f.abs() \
            + self.tx.abs() + self.ty.abs()