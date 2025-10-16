import json, math, numpy as np
from collections import Counter

def to_np(grid): 
    return np.array(grid, dtype=int)

def to_list(arr): 
    return arr.tolist()

def bg_color(arr):
    vals, cnts = np.unique(arr, return_counts=True)
    return int(vals[np.argmax(cnts)]) if len(vals)>0 else 0

def hamming(a, b): 
    if a.shape != b.shape: 
        return 10**9
    return int(np.sum(a != b))

def rotate(arr, k=1): 
    return np.rot90(arr, k=k)

def mirror_h(arr): 
    return arr[:, ::-1]

def mirror_v(arr): 
    return arr[::-1, :]

def transpose(arr): 
    return arr.T

def crop_to_bbox(arr, bg=None):
    if bg is None: bg = bg_color(arr)
    ys, xs = np.where(arr != bg)
    if len(ys)==0: 
        return arr.copy()
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    return arr[y0:y1+1, x0:x1+1]

def bbox_of_pixels(pixels):
    ys = [p[0] for p in pixels]; xs = [p[1] for p in pixels]
    return min(ys), min(xs), max(ys), max(xs)

def pad_to(arr, H, W, bg=None, align='center'):
    if bg is None: bg = bg_color(arr)
    h, w = arr.shape
    out = np.full((H, W), bg, dtype=int)
    if align=='center':
        y0 = (H - h)//2; x0 = (W - w)//2
    else:
        y0 = 0; x0 = 0
    y1 = max(0, min(H, y0 + h))
    x1 = max(0, min(W, x0 + w))
    out[y0:y1, x0:x1] = arr[:y1-y0, :x1-x0]
    return out

def connected_components(arr, bg=None):
    if bg is None: bg = bg_color(arr)
    H, W = arr.shape
    seen = np.zeros((H,W), dtype=bool)
    comps = []
    for y in range(H):
        for x in range(W):
            if seen[y,x] or arr[y,x]==bg: 
                continue
            color = arr[y,x]
            stack = [(y,x)]
            seen[y,x] = True
            pix = []
            while stack:
                cy,cx = stack.pop()
                pix.append((cy,cx))
                for dy,dx in [(1,0),(-1,0),(0,1),(0,-1)]:
                    ny,nx = cy+dy, cx+dx
                    if 0<=ny<H and 0<=nx<W and not seen[ny,nx] and arr[ny,nx]==color:
                        seen[ny,nx]=True
                        stack.append((ny,nx))
            comps.append((color, pix))
    return comps

def majority_color(arr):
    vals, cnts = np.unique(arr, return_counts=True)
    return int(vals[np.argmax(cnts)]) if len(vals)>0 else 0

def resize_integer_scale(obj_arr, target_h, target_w, bg=None):
    if bg is None: bg = bg_color(obj_arr)
    h, w = obj_arr.shape
    if h==0 or w==0:
        return np.full((target_h, target_w), bg, dtype=int)
    sy = max(1, round(target_h / h))
    sx = max(1, round(target_w / w))
    scaled = np.repeat(np.repeat(obj_arr, sy, axis=0), sx, axis=1)
    sh, sw = scaled.shape
    if sh >= target_h and sw >= target_w:
        y0 = (sh - target_h)//2; x0 = (sw - target_w)//2
        scaled = scaled[y0:y0+target_h, x0:x0+target_w]
    else:
        scaled = pad_to(scaled, target_h, target_w, bg=bg, align='center')
    return scaled

def smallest_tile(arr):
    H, W = arr.shape
    for th in range(1, H+1):
        if H % th != 0: 
            continue
        for tw in range(1, W+1):
            if W % tw != 0:
                continue
            tile = arr[0:th, 0:tw]
            tiled = np.tile(tile, (H//th, W//tw))
            if np.array_equal(tiled, arr):
                return tile
    return None

def load_json(path):
    with open(path, "r") as f:
        txt = f.read().strip()
    if txt.startswith("["):
        return json.loads(txt)
    else:
        rows = []
        for line in txt.splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows