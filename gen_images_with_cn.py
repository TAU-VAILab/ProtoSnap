import torch
import numpy as np
from pathlib import Path
from PIL import Image
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

from src.data import Prototype
from src.geometry import GeomTransform, GroupedPoints


def get_opts():
    parser = ArgumentParser()
    parser.add_argument('sign_name', type=str)
    parser.add_argument('--num_of_samples', type=int, default=50)
    parser.add_argument('--output_path', type=str, default='output_cn')
    parser.add_argument('--sd_checkpoint', type=str, default='weights/SD_without_prompt')
    parser.add_argument('--cn_checkpoint', type=str, default='weights/ControlNet')
    parser.add_argument('--con_dir', type=str, default='skeletons/Santakku')
    parser.add_argument('--font_dir', type=str, default='prototypes/Santakku')
    parser.add_argument('--df_filename', type=str, default='prototypes/metadata.csv')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--line_weighting', '-lw', type=float, default=1.0)
    return parser.parse_args()


def build_random_trans():
    a, b, c, d, e, f, tx, ty = (torch.rand(8) * 2 - 1) / 10
    return torch.eye(3) + torch.tensor([[a, b, tx], [c, d, ty], [e, f, 0]])


def visualize_control(proto, output_path, idx):
    fig, ax = plt.subplots()
    ax.imshow(np.ones((512, 512, 3), dtype=np.uint8) * 255)
    for i, indices in enumerate(proto.connectivity):
        for j in indices:
            xs = [proto.points[i][0], proto.points[j][0]]
            ys = [proto.points[i][1], proto.points[j][1]]
            plt.plot(xs, ys, c='black', linewidth=8)
    plt.tight_layout()
    ax.axis('off')
    plt.savefig(output_path / f"{idx}.png", bbox_inches='tight', pad_inches=0)
    plt.close()


def create_control(proto, args, output_path, idx):
    global_transform = GeomTransform(initial_transform=build_random_trans())
    labels = set(proto.labels)
    stroke_transforms = {label: GeomTransform(initial_transform=build_random_trans()) for label in labels}
    gpoints = GroupedPoints(proto.points, args.img_size, proto.labels, proto.connectivity)
    _, dst_points, dst_points_global, _ = gpoints.get_viz_points(global_transform, stroke_transforms)
    if dst_points.min() >= 0 and dst_points.max() < args.img_size:
        proto.points = dst_points.detach().numpy()
        visualize_control(proto, output_path, idx)
        return True
    return False


def create_all_controls(proto, output_path, args):
    count = 0
    controls_output = output_path / "controls"
    controls_output.mkdir(exist_ok=True)
    while count < args.num_of_samples:
        res = create_control(proto, args, controls_output, count)
        if res:
            count += 1

    return controls_output


def generate_images(args, controls_dir):
    controlnet = ControlNetModel.from_pretrained(args.cn_checkpoint, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(args.sd_checkpoint,
                                                             controlnet=controlnet,
                                                             torch_dtype=torch.float16,
                                                             safety_checker=None).to('cuda')
    control_paths = controls_dir.glob("*png")
    controls = [Image.open(path).convert("RGB").resize((512, 512)) for path in control_paths]
    assert len(controls) == args.num_of_samples

    prompt = "cuneiform single ancient icon"
    images_path = controls_dir.parent / "images"
    images_path.mkdir(exist_ok=True)
    for i in range(0, len(controls), 10):
        images = pipe([prompt] * 10, image=controls[i: i + 10]).images
        for j, img in enumerate(images):
            img.save(images_path / f"{i+j}.png")


def main():
    args = get_opts()
    proto = Prototype(args, args.sign_name)
    output_path = Path(args.output_path) / args.sign_name
    output_path.mkdir(exist_ok=True, parents=True)

    controls_output = create_all_controls(proto, output_path, args)
    generate_images(args, controls_output)

if __name__ == '__main__':
    main()

