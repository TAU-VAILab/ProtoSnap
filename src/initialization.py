import torch
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from PIL import ImageOps
import cv2
from pathlib import Path
from tqdm.auto import trange

from src.affine_ransac import Ransac
from src.data import Data

def get_cmap(alpha=0.9):
    C = np.zeros((256, 4), dtype=np.float32)
    C[:, 1] = 1.0  # green component
    C[:, -1] = np.linspace(0, alpha, 256)
    cmap = ListedColormap(C)
    return cmap

def idx2img_coords(i, sim_res, img_size):
    x = i % sim_res / sim_res * img_size
    y = i // sim_res / sim_res * img_size
    return x, y

def get_best_buddies(sim_tensor, img_size):
    sim_res = sim_tensor.shape[0]
    assert sim_tensor.shape == (sim_res, sim_res, sim_res, sim_res), \
            f"Unrecognized shape for DIFT similarity tensor: {sim_tensor.shape}"
    # sim_tensor: proto <-> target similarities

    S = sim_tensor.view(sim_res ** 2, sim_res ** 2) # H*W, H*W

    img_nn = S.argmax(dim=0).cpu().numpy() # for each proto feature, NN in target
    proto_nn = S.argmax(dim=1).cpu().numpy() # for each target feature, NN in proto

    bbs = []
    for i in range(len(img_nn)):
        for j in range(len(proto_nn)):
            if img_nn[i] == j and proto_nn[j] == i:
                i_coords = idx2img_coords(i, sim_res, img_size)
                j_coords = idx2img_coords(j, sim_res, img_size)
                bbs.append((i_coords, j_coords))

    return bbs

def estimate_affine(src_pts, dst_pts, args):
    r = Ransac(K=args.ransac_k, threshold=args.ransac_threshold)
    A, t, inliers = r.ransac_fit(dst_pts.T, src_pts.T)
    inliers = inliers[0]

    H = np.zeros((3, 3), dtype=np.float32)
    H[:2, :2] = A
    H[:2, -1] = t[:, 0]
    H[-1, -1] = 1.

    return H, inliers

def visualize(args, data, bbs, H, inliers, hull_mask1, hull_mask2, score1, score2, agg_score, rep=None):

    name = data.name
    if rep is not None:
        name += f'_{rep}'
    if args.output_folder is None:
        output_path = Path("output") / name
    else:
        output_path = Path("output") / args.output_folder / name
    output_path.mkdir(parents=True, exist_ok=True)
    torch.save(agg_score, output_path / "agg_score.pt")

    proto, img = data.proto, data.target

    img_ = np.array(ImageOps.invert(proto))
    warped = Image.fromarray(cv2.warpPerspective(img_, H, img.size))
    warped = ImageOps.invert(warped)

    cmap = get_cmap()

    def display(img, proto, ax):
        ax.imshow(img)
        ax.imshow(255 - np.asarray(proto.convert('L')), cmap=cmap, alpha=0.5)
        ax.axis('off')

    # color area of prototype covered by convex hull
    P = np.uint8(ImageOps.invert(proto)).copy()
    P[hull_mask2, 1] = P[hull_mask2, 1].clip(max=100)
    P = ImageOps.invert(Image.fromarray(P))

    fig, axs = plt.subplots(2, 4, figsize=(8, 4))
    plt.suptitle(f'Initialization; scores: {score1:.4f} {score2:.4f}; agg_score: {agg_score:.4f}')
    axs[0, 0].imshow(img)
    axs[0, 1].imshow(proto)
    axs[0, 2].imshow(img)
    axs[0, 3].imshow(proto)
    for (xi, yi), (xj, yj) in bbs:
        axs[0, 2].scatter(xi, yi, s=args.init_scatter_size)
        axs[0, 3].scatter(xj, yj, s=args.init_scatter_size)
    display(img, proto, axs[1, 0])
    display(img, warped, axs[1, 1])
    axs[1, 2].imshow(img)
    axs[1, 2].imshow(hull_mask1, cmap='gray', alpha=np.uint8(hull_mask1) * 0.5)
    axs[1, 3].imshow(P)
    for i, ((xi, yi), (xj, yj)) in enumerate(bbs):
        color = None if i in inliers else 'gray'
        axs[1, 2].scatter(xi, yi, s=args.init_scatter_size, c=color)
        axs[1, 3].scatter(xj, yj, s=args.init_scatter_size, c=color)
    for row in axs:
        for ax in row:
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path / f"init.png")
    plt.close()


def scale_H(H, img_size):
    # from [0, 512) to [-1, 1] scale
    r = 2. / (img_size - 1.)
    r_ = (img_size - 1.) / 2.
    scale_matrix = np.array([
        [r, 0., -1.],
        [0., r, -1.],
        [0., 0., 1.]
    ])
    unscale_matrix = np.array([
        [r_, 0., r_],
        [0., r_, r_],
        [0., 0., 1.]
    ])
    return scale_matrix @ H @ unscale_matrix

def get_hull_scores(data, bbs, inliers):
    pL = np.uint8(ImageOps.invert(data.proto.convert('L'))) / 255
    hull_masks = []
    for idx in [0, 1]:
        hull_pts = np.array([
            (int(pairs[idx][0]), int(pairs[idx][1]))
            for i, pairs in enumerate(bbs)
            if i in inliers
        ])
        hull = cv2.convexHull(hull_pts)
        hull_mask = np.zeros_like(pL)
        cv2.drawContours(hull_mask, [hull], -1, 1, -1)
        hull_mask = hull_mask == 1
        hull_masks.append(hull_mask)
    score1 = hull_masks[0].mean()
    score2 = pL[hull_masks[1]].sum() / pL.sum()
    return hull_masks[0], hull_masks[1], score1, score2

def initialize(args, dift, rep=None):

    vals = []
    scores = []

    for _ in trange(args.init_tries, desc="Trying initializations"):

        data = Data(args, dift)

        bbs = get_best_buddies(data.sim_tensor, args.img_size)

        src_pts = np.array([(xi, yi) for (xi, yi), _ in bbs])
        dst_pts = np.array([(xj, yj) for _, (xj, yj) in bbs])

        H, inliers = estimate_affine(src_pts, dst_pts, args)

        hull_mask1, hull_mask2, score1, score2 = get_hull_scores(data, bbs, inliers)

        agg_score = (score1 * score2) ** 0.5

        vals.append([bbs, H, inliers, hull_mask1, hull_mask2, score1, score2, agg_score])
        scores.append(agg_score)
    
    idx = np.argmax(scores)
    H = vals[idx][1]

    visualize(args, data, *vals[idx], rep=rep)

    return data, scale_H(H, args.img_size)