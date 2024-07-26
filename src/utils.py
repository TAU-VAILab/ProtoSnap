import numpy as np
from PIL import Image
from torchvision.transforms import v2
import os

COLORS = ['red', 'blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']


def find_boundary(ar):
    # find indices of first and last "True" elements
    lar = list(ar)
    start = lar.index(True)
    end = len(lar) - lar[::-1].index(True)
    return start, end


def crop_img(img, is_pil=False):
    if is_pil:
        img = np.array(img)
    has_content = img != 255
    has_content_col = has_content.any(axis=0)
    has_content_row = has_content.any(axis=1)

    cs, ce = find_boundary(has_content_col)
    rs, re = find_boundary(has_content_row)

    if is_pil:
        return Image.fromarray(img[rs:re, cs:ce]), cs, ce, rs, re
    return img[rs:re, cs:ce], cs, ce, rs, re


def get_path_to_proto(args, prompt, df):
    unicode = df[df['name'] == prompt]['hex'].values[0]
    prefix = unicode[:5]
    return os.path.join(args.font_dir, f"{prefix}xx/{unicode}.png")


def get_proto_img(args, prompt, df):
    proto = Image.open(get_path_to_proto(args, prompt, df)).convert("L")
    proto, cs, ce, rs, re = crop_img(proto, is_pil=True)
    proto = v2.Pad(10, fill=(255, 255, 255))(proto.convert("RGB"))
    w, h = proto.size
    return v2.Resize((512, 512))(proto), cs, rs, w, h


def flatten(array_of_arrays):
    return [y for x in array_of_arrays for y in x]

