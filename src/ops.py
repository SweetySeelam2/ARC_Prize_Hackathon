from collections import Counter
import numpy as np
from .utils import (bg_color, pad_to, rotate, mirror_h, mirror_v, transpose, 
                    crop_to_bbox, connected_components, bbox_of_pixels, 
                    majority_color, resize_integer_scale, smallest_tile)

def learn_color_map(train_pairs):
    mapping = {}
    for xin, xout in train_pairs:
        cin = Counter(xin.flatten())
        cout = Counter(xout.flatten())
        if not cin or not cout:
            continue
        bg_in  = max(cin, key=cin.get)
        bg_out = max(cout, key=cout.get)
        in_sorted  = [c for c,_ in cin.most_common() if c!=bg_in]
        out_sorted = [c for c,_ in cout.most_common() if c!=bg_out]
        for i, c in enumerate(in_sorted):
            if i < len(out_sorted):
                mapping[c] = out_sorted[i]
        mapping[bg_in] = bg_out
    return mapping

def apply_color_map(arr, cmap):
    out = arr.copy()
    for c_in, c_out in cmap.items():
        out[arr==c_in] = c_out
    return out

# ---- 15 Ops ----
def op_identity(arr, H, W, train_pairs=None):
    return pad_to(arr, H, W, bg=bg_color(arr))

def op_rotate90(arr, H, W, train_pairs=None):
    return pad_to(rotate(arr,1), H, W, bg=bg_color(arr))

def op_rotate180(arr, H, W, train_pairs=None):
    return pad_to(rotate(arr,2), H, W, bg=bg_color(arr))

def op_rotate270(arr, H, W, train_pairs=None):
    return pad_to(rotate(arr,3), H, W, bg=bg_color(arr))

def op_mirror_h(arr, H, W, train_pairs=None):
    return pad_to(mirror_h(arr), H, W, bg=bg_color(arr))

def op_mirror_v(arr, H, W, train_pairs=None):
    return pad_to(mirror_v(arr), H, W, bg=bg_color(arr))

def op_transpose(arr, H, W, train_pairs=None):
    return pad_to(transpose(arr), H, W, bg=bg_color(arr))

def op_crop_center_pad(arr, H, W, train_pairs=None):
    cropped = crop_to_bbox(arr, bg=bg_color(arr))
    return pad_to(cropped, H, W, bg=bg_color(arr), align='center')

def op_recolor_learned(arr, H, W, train_pairs=None):
    cmap = learn_color_map(train_pairs or [])
    recol = apply_color_map(arr, cmap)
    return pad_to(recol, H, W, bg=bg_color(recol))

def op_largest_cc_center(arr, H, W, train_pairs=None):
    comps = connected_components(arr, bg=bg_color(arr))
    if not comps:
        return pad_to(arr, H, W, bg=bg_color(arr))
    color, pixels = max(comps, key=lambda cp: len(cp[1]))
    y0,x0,y1,x1 = bbox_of_pixels(pixels)
    obj = arr[y0:y1+1, x0:x1+1]
    return pad_to(obj, H, W, bg=bg_color(arr), align='center')

def op_integer_resize_fg(arr, H, W, train_pairs=None):
    fg = crop_to_bbox(arr, bg=bg_color(arr))
    return resize_integer_scale(fg, H, W, bg=bg_color(arr))

def op_majority_fill(arr, H, W, train_pairs=None):
    bg = bg_color(arr)
    maj = majority_color(arr)
    out = pad_to(arr, H, W, bg=bg)
    out[out==bg] = maj
    return out

def op_complete_by_reflection(arr, H, W, train_pairs=None):
    bg = bg_color(arr)
    a = pad_to(arr, H, W, bg=bg)
    out = a.copy()
    Hh, Ww = a.shape
    left = a[:, :Ww//2]
    right = a[:, Ww - (Ww//2):]
    if np.sum(right!=bg) < np.sum(left!=bg)//4:
        out[:, Ww - (Ww//2):] = mirror_h(left)
        return out
    if np.sum(left!=bg) < np.sum(right!=bg)//4:
        out[:, :Ww//2] = mirror_h(right)
        return out
    top = a[:Hh//2, :]
    bot = a[Hh - (Hh//2):, :]
    if np.sum(bot!=bg) < np.sum(top!=bg)//4:
        out[Hh - (Hh//2):, :] = mirror_v(top)
        return out
    if np.sum(top!=bg) < np.sum(bot!=bg)//4:
        out[:Hh//2, :] = mirror_v(bot)
        return out
    return out

def op_project_rows_or_cols(arr, H, W, train_pairs=None):
    a = pad_to(arr, H, W, bg=bg_color(arr))
    rows = [tuple(r) for r in a]
    cols = [tuple(c) for c in a.T]
    from collections import Counter
    row_counts = Counter(rows)
    col_counts = Counter(cols)
    out = a.copy()
    if row_counts and (row_counts.most_common(1)[0][1] >= H//2):
        best = np.array(row_counts.most_common(1)[0][0], dtype=int)
        out = np.tile(best, (H,1))
        return out
    if col_counts and (col_counts.most_common(1)[0][1] >= W//2):
        best = np.array(col_counts.most_common(1)[0][0], dtype=int)
        out = np.tile(best.reshape(-1,1), (1,W))
        return out
    return out

def op_tile_smallest_patch(arr, H, W, train_pairs=None):
    tile = smallest_tile(arr)
    if tile is None:
        return pad_to(arr, H, W, bg=bg_color(arr))
    th, tw = tile.shape
    import math
    tiled = np.tile(tile, (max(1, math.ceil(H/th)), max(1, math.ceil(W/tw))))
    return tiled[:H, :W]

def op_outline_cc(arr, H, W, train_pairs=None):
    bg = bg_color(arr)
    a = pad_to(arr, H, W, bg=bg)
    out = np.full_like(a, bg)
    Hh, Ww = a.shape
    for y in range(Hh):
        for x in range(Ww):
            if a[y,x]==bg: 
                continue
            for dy,dx in [(1,0),(-1,0),(0,1),(0,-1)]:
                ny,nx = y+dy, x+dx
                if not (0<=ny<Hh and 0<=nx<Ww) or a[ny,nx]==bg:
                    out[y,x] = a[y,x]
                    break
    return out

BASIC_OPS = [
    op_identity,
    op_rotate90, op_rotate180, op_rotate270,
    op_mirror_h, op_mirror_v,
    op_transpose,
    op_crop_center_pad,
    op_recolor_learned,
    op_largest_cc_center,
    op_integer_resize_fg,
    op_majority_fill,
    op_complete_by_reflection,
    op_project_rows_or_cols,
    op_tile_smallest_patch,
    op_outline_cc,
]