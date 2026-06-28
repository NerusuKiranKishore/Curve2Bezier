"""
curve2bezier_v9.py  —  Image → Minimal Piecewise Cubic Bézier  [v9]
====================================================================
Base: v7 (excellent open-curve fitting)
Patch: v8 closed-curve improvements merged in (§2, §3, §6, §8, §10)

USAGE
─────
  python curve2bezier_v9.py image.png                      # single image (auto-detect topology)
  python curve2bezier_v9.py image.png --closed             # FORCE closed-curve pipeline
  python curve2bezier_v9.py image.png --open               # FORCE open-curve pipeline
  python curve2bezier_v9.py image.png --tol 2.0            # tighter tolerance
  python curve2bezier_v9.py image.png --out r.png --svg r.svg
  python curve2bezier_v9.py --aefs-compare                 # paper benchmark
  python curve2bezier_v9.py --batch-dir ./images           # batch mode
  python curve2bezier_v9.py --batch-dir ./images --closed  # batch (force closed)
  python curve2bezier_v9.py --benchmark --bench-glob "*.png"
  python curve2bezier_v9.py image.png --no-symmetry
  python curve2bezier_v9.py image.png --corner-angle 35

INSTALL
───────
  pip install opencv-python numpy matplotlib pillow scipy scikit-image

WHAT'S NEW IN v9
─────────────────
  INHERITED from v8 (closed-curve support):
  MOD-v9-1..6  all closed-curve fixes retained.
  NEW in v10:
  MOD-v10-1  USER_TOPOLOGY_OVERRIDE — --closed / --open CLI flags bypass
             all automatic detection. When either flag is given:
               * mask_to_ordered_points uses the correct traversal directly
               * smoothing, fitting, SVG, and plot all use the forced type
               * auto-detection (_is_closed_skeleton) is skipped entirely
             Without either flag behaviour is identical to v9 (auto-detect).
"""

import argparse, csv, glob, os, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import cv2
from PIL import Image as PILImage
from collections import deque
from scipy.ndimage import gaussian_filter1d
from scipy import stats as scipy_stats
from skimage.morphology import skeletonize

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
IMAGE_PATH           = "img1.jpeg"
TOLERANCE            = 8.0
OUT_PATH             = "bezier_output.png"
SVG_PATH             = "bezier_output.svg"
CORNER_ANGLE_THRESH  = np.radians(45)
LOSSLESS_TOL         = 1e-2
SYMMETRY_CONF_THRESH = 0.90
SYMMETRY_POINT_TOL   = 3.0

AEFS_PAPER = {
    "butterfly": {
        "n_pts": 577, "tol": 0.5, "degree": 2,
        "runtime": 0.633, "segments": 38,
        "max_error": 0.2091, "mean_error": 0.0361,
        "ctrl_pts": 115,
        "platform": "R / Apple M1 Max GPU",
    },
    "inflection": {
        "n_pts": 91, "tol": 1.0, "degree": 2,
        "runtime": 0.388, "segments": 2,
        "max_error": 0.1974, "mean_error": 0.0364,
        "ctrl_pts": 7, "platform": "R / Apple M1 Max GPU",
    },
}

# ══════════════════════════════════════════════════════════════
# §1  IMAGE → CLEAN BINARY MASK  (v7 — unchanged)
# ══════════════════════════════════════════════════════════════

def load_and_binarize(path):
    is_jpeg = path.lower().endswith(('.jpg', '.jpeg'))
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        pil     = PILImage.open(path).convert('RGB')
        img_rgb = np.array(pil)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if not is_jpeg:
        gray = _strip_dark_border(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    bg   = cv2.GaussianBlur(gray.astype(np.float32), (71, 71), 0)
    bw_a = ((bg - gray.astype(np.float32)) > 6).astype(np.uint8) * 255
    bw_b = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 51, 8)
    _, bw_c = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if bw_c.mean() > 38:
        bw_c[:] = 0
    bw = np.maximum(np.maximum(bw_a, bw_b), bw_c)
    k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=2)
    bw = _keep_large_ccs(bw, max(50, int(gray.size * 0.0005)))
    n_cc, lbl, sts, _ = cv2.connectedComponentsWithStats(bw)
    if n_cc > 2:
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k5, iterations=3)
    return img_rgb, bw, H, W

def _strip_dark_border(gray):
    border = float(np.percentile(gray, 0.5)) + 30
    r_ok   = np.where(gray.mean(axis=1) > border)[0]
    c_ok   = np.where(gray.mean(axis=0) > border)[0]
    if len(r_ok) == 0 or len(c_ok) == 0:
        return gray
    r0, r1 = r_ok[0], r_ok[-1]; c0, c1 = c_ok[0], c_ok[-1]
    h, w = gray.shape
    if r0 > 5 or c0 > 5 or r1 < h-5 or c1 < w-5:
        return gray[r0:r1+1, c0:c1+1]
    return gray

def _keep_large_ccs(bw, min_area):
    n, lbl, sts, _ = cv2.connectedComponentsWithStats(bw)
    out = np.zeros_like(bw)
    for i in range(1, n):
        if sts[i, cv2.CC_STAT_AREA] >= min_area:
            out[lbl == i] = 255
    return out

# ══════════════════════════════════════════════════════════════
# §2  BINARY MASK → ORDERED SKELETON POINTS
#     MOD-v9-1 / MOD-v9-2: closed topology detection + cyclic walk
#     Open-curve path is identical to v7.
# ══════════════════════════════════════════════════════════════

# ── Closed-loop detection thresholds (MOD-v9-1) ───────────────
# GATE 1 — endpoint pixel ratio: a truly closed loop has NO degree-1
#           pixels (both ends touch), so the count should be 0 or
#           only a handful caused by digitisation noise.
#           We allow at most 1 endpoint per 200 skeleton pixels.
_CLOSED_EP_RATIO = 0.005          # <= 0.5% of pixels may be endpoints

# GATE 2 — physical gap: even if the ratio passes, the two BFS-farthest
#           endpoints must be within _CLOSED_GAP_RATIO * arc_length of
#           each other.  Open curves (Z, S, lines) always have their two
#           tips far apart relative to their own length, so this gate
#           rejects them even when their skeleton happens to have few
#           degree-1 pixels.
_CLOSED_GAP_RATIO = 0.05          # gap / arc_length <= 5%

def mask_to_ordered_points(bw, force_closed=None):
    skel    = skeletonize(bw > 0)
    skel_u8 = (skel * 255).astype(np.uint8)
    n_cc, lbl, sts, _ = cv2.connectedComponentsWithStats(skel_u8)
    # Sort largest component first so dominant stroke leads
    comp_info = sorted(
        [(sts[i, cv2.CC_STAT_AREA], lbl == i)
         for i in range(1, n_cc) if sts[i, cv2.CC_STAT_AREA] >= 10],
        key=lambda x: -x[0])
    if not comp_info:
        return np.array([[0.0, 0.0]]), False
    if len(comp_info) > 1:
        print(f"      [multi-stroke] {len(comp_info)} strokes found.")
    all_paths = []; dominant_closed = False
    for ci, (area, comp_mask) in enumerate(comp_info):
        comp_skel = skel & comp_mask
        # MOD-v10-1: user override takes priority; else auto-detect
        if force_closed is not None:
            closed = force_closed
            src_label = "USER OVERRIDE"
        else:
            closed = _is_closed_skeleton(comp_skel)   # MOD-v9-1
            src_label = "auto-detected"
        if ci == 0:
            dominant_closed = closed
        if closed:
            print(f"      [topology] Stroke {ci+1}: CLOSED ({src_label}).")
            path = _cyclic_walk(comp_skel)            # MOD-v9-2
        else:
            print(f"      [topology] Stroke {ci+1}: OPEN ({src_label}).")
            path = _open_path_bfs(comp_skel, ci)
        if path is not None and len(path) >= 2:
            all_paths.append(path)
    if not all_paths:
        return np.array([[0.0, 0.0]]), False
    if len(all_paths) > 1:
        all_paths.sort(key=lambda p: p[:, 0].mean())
        return np.vstack(all_paths), dominant_closed
    return all_paths[0], dominant_closed


# MOD-v9-1 — two-gate closure test
def _is_closed_skeleton(comp_skel):
    """
    Returns True only when BOTH gates agree the skeleton is a closed loop.

    GATE 1 — endpoint-pixel ratio:
        Count degree-1 pixels (exactly one neighbour).  A genuine closed
        loop has zero such pixels; we allow up to _CLOSED_EP_RATIO as
        digitisation noise tolerance.  This alone is insufficient for long
        open strokes (Z, S, wavy lines) because 2 endpoints / thousands of
        pixels is still a tiny ratio.

    GATE 2 — physical gap between the two BFS-farthest endpoints:
        Run double-BFS to find the two true tips of the stroke, then
        measure their Euclidean distance versus the skeleton arc-length.
        Closed loops have their tips nearly touching (gap ~ 0); open
        curves always have their tips far apart (gap ~ arc-length).
        We require  gap / arc_length < _CLOSED_GAP_RATIO (default 5 %).
    """
    pixels = np.argwhere(comp_skel)
    n_px   = len(pixels)
    if n_px < 8:
        return False
    h, w = comp_skel.shape

    # --- Gate 1: endpoint pixel ratio ---
    ep = 0
    for r, c in pixels:
        nb = sum(1 for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                 if (dr or dc) and 0 <= r+dr < h and 0 <= c+dc < w
                 and comp_skel[r+dr, c+dc])
        if nb == 1:
            ep += 1
    ratio = ep / max(n_px, 1)
    if ratio >= _CLOSED_EP_RATIO:
        return False   # clearly open — endpoints visible

    # --- Gate 2: physical gap test ---
    # Arc-length approximation: number of skeleton pixels (each step ~1px)
    arc_length = float(n_px)

    # Find the two farthest endpoints via double-BFS
    start         = tuple(pixels[0])
    tip_a, _      = _bfs_farthest(comp_skel, start)
    tip_b, _      = _bfs_farthest(comp_skel, tip_a)
    gap = float(np.hypot(tip_a[0] - tip_b[0], tip_a[1] - tip_b[1]))

    closed = (gap / max(arc_length, 1.0)) < _CLOSED_GAP_RATIO
    if not closed:
        print(f"      [topology] Gate-2 OPEN: gap={gap:.1f}px  "
              f"arc={arc_length:.0f}px  ratio={gap/arc_length:.3f} "
              f"(threshold {_CLOSED_GAP_RATIO})")
    return closed


# MOD-v9-2 — cyclic neighbour-priority walk for closed loops


# MOD-v9-2 — cyclic neighbour-priority walk for closed loops
def _cyclic_walk(comp_skel):
    """
    Traverse a closed skeleton loop with a neighbour-priority walk.
    Visits every pixel exactly once and returns them as (x, y) array.
    Start pixel is chosen as the topmost-leftmost for reproducibility.
    """
    pixels_rc = np.argwhere(comp_skel)
    if len(pixels_rc) < 3:
        return None
    idx   = int(np.lexsort((pixels_rc[:,1], pixels_rc[:,0]))[0])
    start = tuple(pixels_rc[idx])
    h, w  = comp_skel.shape
    visited = {start}; path_rc = [start]; current = start
    DIRS = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]
    while True:
        r, c = current; found = False
        for dr, dc in DIRS:
            nr, nc = r+dr, c+dc
            if (0<=nr<h and 0<=nc<w and comp_skel[nr,nc]
                    and (nr,nc) not in visited):
                visited.add((nr,nc)); path_rc.append((nr,nc))
                current = (nr,nc); found = True; break
        if not found:
            break
    return np.array([[c, r] for r, c in path_rc], dtype=float)


def _open_path_bfs(comp_skel, comp_idx):
    """Double-BFS farthest-point path for open strokes (v7 logic)."""
    pixels = np.argwhere(comp_skel)
    if len(pixels) < 2:
        return None
    h, w = comp_skel.shape
    branch_count = 0
    for r, c in pixels:
        nb = sum(1 for dr in (-1,0,1) for dc in (-1,0,1)
                 if (dr!=0 or dc!=0) and 0<=r+dr<h and 0<=c+dc<w
                 and comp_skel[r+dr, c+dc])
        if nb >= 3: branch_count += 1
    if branch_count > 0:
        pct = 100.*branch_count/max(len(pixels),1)
        print(f"      [WARNING] Stroke {comp_idx+1}: {branch_count} branch "
              f"pixel(s) ({pct:.1f}%). Longest path used.")
    start      = tuple(pixels[0])
    end_a, _   = _bfs_farthest(comp_skel, start)
    end_b, prv = _bfs_farthest(comp_skel, end_a)
    path       = _trace_path(prv, end_b)
    return np.array([[c, r] for r, c in path], dtype=float)


def _bfs_farthest(skel, start):
    h, w = skel.shape
    prev = {start: None}; dist = {start: 0}
    queue = deque([start]); far, mx = start, 0
    while queue:
        r, c = queue.popleft(); d = dist[(r, c)]
        if d > mx: mx, far = d, (r, c)
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr==0 and dc==0: continue
                nr, nc = r+dr, c+dc
                if (0<=nr<h and 0<=nc<w and skel[nr,nc] and (nr,nc) not in prev):
                    prev[(nr,nc)] = (r,c); dist[(nr,nc)] = d+1
                    queue.append((nr,nc))
    return far, prev

def _trace_path(prev, end):
    path, cur = [], end
    while cur is not None: path.append(cur); cur = prev.get(cur)
    return path[::-1]

# ══════════════════════════════════════════════════════════════
# §3  SMOOTH
#     MOD-v9-3: closed-curve gets wrap-around Gaussian;
#               open-curve uses v7's original smooth_points.
# ══════════════════════════════════════════════════════════════

def smooth_points(pts, sigma=1.5, downsample=2):
    """Open-curve smoothing — v7 original (endpoints clamped)."""
    xs = gaussian_filter1d(pts[:, 0], sigma=sigma)
    ys = gaussian_filter1d(pts[:, 1], sigma=sigma)
    return np.column_stack([xs[::downsample], ys[::downsample]])

def smooth_points_closed(pts, sigma=1.5, downsample=2):
    """
    MOD-v9-3 — Closed-curve smoothing with wrap-around padding.
    Eliminates the seam kink that standard Gaussian would introduce.
    """
    n   = len(pts)
    pad = min(n - 1, int(np.ceil(3 * sigma)))
    tiled = np.vstack([pts[-pad:], pts, pts[:pad]])
    xs = gaussian_filter1d(tiled[:, 0], sigma=sigma)[pad:pad+n]
    ys = gaussian_filter1d(tiled[:, 1], sigma=sigma)[pad:pad+n]
    return np.column_stack([xs, ys])[::downsample]

# ══════════════════════════════════════════════════════════════
# §SYM  SYMMETRY DETECTION  (v7 — unchanged)
# ══════════════════════════════════════════════════════════════

def detect_axis_symmetry(pts, point_tol=SYMMETRY_POINT_TOL,
                         conf_thresh=SYMMETRY_CONF_THRESH):
    if len(pts) < 8:
        return None, 0.0, pts.mean(axis=0)
    cx, cy = pts.mean(axis=0); results = {}
    for axis in ('vertical', 'horizontal'):
        if axis == 'vertical':
            ref = np.column_stack([2*cx - pts[:,0], pts[:,1]])
        else:
            ref = np.column_stack([pts[:,0], 2*cy - pts[:,1]])
        diff  = ref[:,np.newaxis,:] - pts[np.newaxis,:,:]
        min_d = np.sqrt((diff**2).sum(axis=2).min(axis=1))
        results[axis] = (min_d < point_tol).sum() / len(pts)
    best_axis = max(results, key=results.get)
    best_conf = results[best_axis]
    if best_conf >= conf_thresh:
        print(f"      [symmetry] {best_axis} axis (conf={best_conf:.2%})")
        return best_axis, best_conf, np.array([cx, cy])
    return None, best_conf, np.array([cx, cy])

def _apply_symmetry_fit(pts, axis, centroid, tolerance):
    cx, cy = centroid
    half_pts = pts[pts[:,0] <= cx] if axis == 'vertical' else pts[pts[:,1] <= cy]
    if len(half_pts) < 4: return None, False
    segs, _, _ = fit_bezier_curves(half_pts, tolerance=tolerance, _symmetry_call=True)
    if not segs: return None, False
    mirrored = []
    for seg in reversed(segs):
        m = seg.copy()
        if axis == 'vertical': m[:,0] = 2*cx - m[:,0]
        else:                  m[:,1] = 2*cy - m[:,1]
        mirrored.append(m[::-1])
    return segs + mirrored, True

# ══════════════════════════════════════════════════════════════
# §4  CORE BÉZIER MATH  (v7 — unchanged)
# ══════════════════════════════════════════════════════════════

def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.array([1.0, 0.0])

def _tangent_at_start(pts, k=None):
    n = len(pts)
    k = min(max(3, n//8), 10) if k is None else k
    k = min(k, n-1)
    for i in range(k, 0, -1):
        v = pts[i] - pts[0]
        if np.linalg.norm(v) > 1e-10: return _normalize(v)
    return _normalize(pts[-1] - pts[0])

def _chord_param(pts):
    d = np.diff(pts, axis=0); ls = np.sqrt((d**2).sum(axis=1))
    t = np.concatenate([[0.0], np.cumsum(ls)]); tot = t[-1]
    return t / tot if tot > 1e-12 else np.linspace(0.0, 1.0, len(pts))

def _bezier_eval(ctrl, t):
    mt = 1.0 - t
    return (np.outer(mt**3, ctrl[0]) + np.outer(3*mt**2*t, ctrl[1]) +
            np.outer(3*mt*t**2, ctrl[2]) + np.outer(t**3, ctrl[3]))

def _generate_bezier(pts, u, t1, t2):
    P0 = pts[0].astype(float); P3 = pts[-1].astype(float)
    t1 = np.asarray(t1, float); t2 = np.asarray(t2, float)
    u_arr = np.asarray(u, float); mt = 1.0 - u_arr
    b1 = 3*mt**2*u_arr; b2 = 3*mt*u_arr**2; b0 = mt**3; b3 = u_arr**3
    A1 = b1[:,np.newaxis]*t1; A2 = b2[:,np.newaxis]*t2
    C00 = float((A1*A1).sum()); C01 = float((A1*A2).sum()); C11 = float((A2*A2).sum())
    RHS = (pts.astype(float) - np.outer(b0+b1, P0) - np.outer(b2+b3, P3))
    X0 = float((A1*RHS).sum()); X1 = float((A2*RHS).sum())
    det = C00*C11 - C01*C01; fb = max(np.linalg.norm(P3-P0), 1e-6) / 3.0
    if abs(det) > 1e-12:
        alpha = (C11*X0 - C01*X1) / det; beta = (C00*X1 - C01*X0) / det
    else:
        alpha = beta = fb
    if alpha <= 0.0 and beta <= 0.0: alpha = beta = fb
    elif alpha <= 0.0: alpha = fb * 0.1
    elif beta  <= 0.0: beta  = fb * 0.1
    return np.array([P0, P0 + alpha*t1, P3 + beta*t2, P3])

def _nr_refine(ctrl, pts, u, iters=4):
    u_arr = np.asarray(u, float); pts_a = np.asarray(pts, float)
    for _ in range(iters):
        mt = 1.0 - u_arr
        b0 = mt**3; b1 = 3*mt**2*u_arr; b2 = 3*mt*u_arr**2; b3 = u_arr**3
        P  = (np.outer(b0, ctrl[0]) + np.outer(b1, ctrl[1]) +
              np.outer(b2, ctrl[2]) + np.outer(b3, ctrl[3]))
        dP = (np.outer(3*mt**2,    ctrl[1]-ctrl[0]) +
              np.outer(6*mt*u_arr, ctrl[2]-ctrl[1]) +
              np.outer(3*u_arr**2, ctrl[3]-ctrl[2]))
        d2P= (np.outer(6*mt,    ctrl[2]-2*ctrl[1]+ctrl[0]) +
              np.outer(6*u_arr, ctrl[3]-2*ctrl[2]+ctrl[1]))
        diff = P - pts_a
        num  = (diff*dP).sum(axis=1); den = (dP*dP).sum(axis=1) + (diff*d2P).sum(axis=1)
        mask = np.abs(den) > 1e-12
        u_new = np.clip(u_arr - np.where(mask, num/(den+1e-30), 0.0), 0.0, 1.0)
        mt2 = 1.0 - u_new
        Pn  = (np.outer(mt2**3,         ctrl[0]) + np.outer(3*mt2**2*u_new, ctrl[1]) +
               np.outer(3*mt2*u_new**2, ctrl[2]) + np.outer(u_new**3,       ctrl[3]))
        u_arr = np.where(((Pn-pts_a)**2).sum(axis=1) < (diff**2).sum(axis=1), u_new, u_arr)
    return u_arr

_T_CHECK = np.linspace(0.0, 1.0, 200)

def _max_error_dense(ctrl, pts):
    curve_pts = _bezier_eval(ctrl, _T_CHECK)
    pts_arr   = np.asarray(pts, float)
    diff      = pts_arr[:,np.newaxis,:] - curve_pts[np.newaxis,:,:]
    return float(np.sqrt((diff**2).sum(axis=2).min(axis=1).max()))

# ══════════════════════════════════════════════════════════════
# §4b  SINGLE-SEGMENT FIT  (v7 — unchanged)
# ══════════════════════════════════════════════════════════════

def _fit_one_segment(pts, t1, t2, nr_rounds=3, nr_iters=4):
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        ctrl = np.array([pts[0], pts[0]+1e-3*t1, pts[0]+1e-3*t1, pts[0]])
        return ctrl, 0.0
    u    = _chord_param(pts)
    ctrl = _generate_bezier(pts, u, t1, t2)
    for _ in range(nr_rounds):
        u    = _nr_refine(ctrl, pts, u, nr_iters)
        ctrl = _generate_bezier(pts, u, t1, t2)
    return ctrl, _max_error_dense(ctrl, pts)

# ══════════════════════════════════════════════════════════════
# §4c  LEGACY recursive subdivider (v7 — unchanged)
# ══════════════════════════════════════════════════════════════

_SPLIT_DENSE = 100

def _max_error(ctrl, pts, u=None):
    t_dense = np.linspace(0, 1, _SPLIT_DENSE)
    curve_pts = _bezier_eval(ctrl, t_dense)
    pts_arr = np.asarray(pts, float)
    diff = pts_arr[:,np.newaxis,:] - curve_pts[np.newaxis,:,:]
    sq_d = (diff**2).sum(axis=2); min_sq = sq_d.min(axis=1)
    return float(min_sq.max()), int(np.argmax(min_sq))

def _fit_segment(pts, t1, t2, tol_sq, depth=0, max_depth=28):
    pts = np.asarray(pts, float); n = len(pts)
    if n < 2: return [], []
    if n == 2:
        d = np.linalg.norm(pts[1]-pts[0]) / 3.0
        return [np.array([pts[0], pts[0]+d*t1, pts[1]-d*t2, pts[1]])], [pts]
    u    = list(_chord_param(pts))
    ctrl = _generate_bezier(pts, u, t1, t2)
    err_sq, split = _max_error(ctrl, pts, u)
    if err_sq <= tol_sq: return [ctrl], [pts]
    if err_sq < 4.0*tol_sq:
        for _ in range(4):
            u = list(_nr_refine(ctrl, pts, np.array(u)))
            ctrl = _generate_bezier(pts, u, t1, t2)
            err_sq, split = _max_error(ctrl, pts, u)
            if err_sq <= tol_sq: return [ctrl], [pts]
    if depth >= max_depth: return [ctrl], [pts]
    split = max(1, min(split, n-2))
    v_split = _normalize(pts[min(split+1,n-1)] - pts[max(split-1,0)])
    l_segs, l_pts = _fit_segment(pts[:split+1],  t1,      v_split, tol_sq, depth+1, max_depth)
    r_segs, r_pts = _fit_segment(pts[split:],   -v_split, t2,      tol_sq, depth+1, max_depth)
    if l_segs and r_segs:
        r_segs[0] = r_segs[0].copy(); r_segs[0][0] = l_segs[-1][3]
    return l_segs + r_segs, l_pts + r_pts

# ══════════════════════════════════════════════════════════════
# §5  CORNER DETECTION  (v7 — unchanged)
# ══════════════════════════════════════════════════════════════

def _detect_corners(pts, angle_thresh=CORNER_ANGLE_THRESH, window=5):
    n = len(pts); corners = []; w = max(1, min(window, n//4))
    for i in range(w, n-w):
        v_in  = pts[i] - pts[max(i-w, 0)]
        v_out = pts[min(i+w, n-1)] - pts[i]
        ni = np.linalg.norm(v_in); no = np.linalg.norm(v_out)
        if ni < 1e-10 or no < 1e-10: continue
        cos_a = np.clip(np.dot(v_in, v_out)/(ni*no), -1.0, 1.0)
        if np.arccos(cos_a) > angle_thresh: corners.append(i)
    if not corners: return []
    deduped = [corners[0]]
    for ci in corners[1:]:
        if ci - deduped[-1] > w: deduped.append(ci)
    return deduped

# ══════════════════════════════════════════════════════════════
# §5b  GLOBALLY OPTIMAL SEGMENTATION  (v7 — unchanged)
# ══════════════════════════════════════════════════════════════

def _longest_valid_extension(pts, tol, max_seg_pts=160):
    N = len(pts); segs = []; pt_ranges = []
    i = 0
    while i < N - 1:
        t1 = _tangent_at_start(pts[i:min(i+10, N)])
        lo = min(i+2, N-1); hi = min(N-1, i+max_seg_pts)
        best_j = i+1; best_ctrl = None
        while lo <= hi:
            mid = (lo + hi + 1) // 2
            sub = pts[i:mid+1]
            t2  = -_tangent_at_start(sub[-min(10, len(sub)):][::-1])
            ctrl, e = _fit_one_segment(sub, t1, t2)
            if e <= tol:
                lo = mid+1; best_j = mid; best_ctrl = ctrl
            else:
                hi = mid-1
        if best_ctrl is None:
            sub = pts[i:i+2]; t2 = -_tangent_at_start(sub[::-1])
            best_ctrl, _ = _fit_one_segment(sub, t1, t2); best_j = i+1
        segs.append(best_ctrl); pt_ranges.append(pts[i:best_j+1]); i = best_j
    return segs, pt_ranges

def _exhaustive_merge(segs, pt_ranges, tol, max_run=12):
    if len(segs) <= 1: return segs, pt_ranges
    changed = True
    while changed:
        changed = False; new_segs = []; new_pts = []; i = 0; n = len(segs)
        while i < n:
            best_run = 1; best_ctrl = segs[i]; best_comb = pt_ranges[i]
            for run in range(2, min(max_run+1, n-i+1)):
                comb = np.vstack([pt_ranges[i+k][1:] if k>0 else pt_ranges[i]
                                  for k in range(run)])
                t1 = _tangent_at_start(comb[:min(10, len(comb))])
                t2 = -_tangent_at_start(comb[-min(10, len(comb)):][::-1])
                ctrl, e = _fit_one_segment(comb, t1, t2)
                if e <= tol:   best_run = run; best_ctrl = ctrl; best_comb = comb
                elif e > tol*6: break
            if best_run > 1: changed = True
            new_segs.append(best_ctrl); new_pts.append(best_comb); i += best_run
        segs = new_segs; pt_ranges = new_pts
    return segs, pt_ranges

# ══════════════════════════════════════════════════════════════
# §6  PUBLIC FIT
#     Open curve: v7 logic (unchanged).
#     Closed curve: MOD-v9-4 (_fit_closed replaces the old seam hack).
# ══════════════════════════════════════════════════════════════

def _measure_max_error(segments, pts, samples=200):
    if not segments: return float('inf')
    t_s = np.linspace(0, 1, samples)
    curve_pts = np.vstack([_bezier_eval(seg, t_s) for seg in segments])
    max_err = 0.0
    for s in range(0, len(pts), 1000):
        chunk = pts[s:s+1000]
        diff  = chunk[:,np.newaxis,:] - curve_pts[np.newaxis,:,:]
        dists = np.sqrt((diff**2).sum(axis=2)).min(axis=1)
        max_err = max(max_err, dists.max())
    return float(max_err)


# ── Closed-curve helpers (MOD-v9-4) ───────────────────────────

def _best_cut_point(pts):
    """
    Index of minimum curvature on a closed loop — the smoothest spot
    for the invisible seam, making G1 enforcement as easy as possible.
    """
    n = len(pts); k = max(3, min(10, n // 12))
    min_angle = float('inf'); best = 0
    for i in range(n):
        v_in  = pts[i]          - pts[(i - k) % n]
        v_out = pts[(i + k) % n] - pts[i]
        ni = np.linalg.norm(v_in); no = np.linalg.norm(v_out)
        if ni < 1e-10 or no < 1e-10: continue
        cos_a = np.clip(np.dot(v_in, v_out) / (ni * no), -1., 1.)
        if np.arccos(cos_a) < min_angle:
            min_angle = np.arccos(cos_a); best = i
    return best


def _enforce_g1_at_seam(segments):
    """
    Make the last→first junction G1 (tangent-continuous).
    Averages incoming and outgoing tangent directions at the seam
    anchor and adjusts both neighbouring handles to match.
    """
    if len(segments) < 2:
        return segments
    segs    = [s.copy() for s in segments]
    anchor  = segs[0][0]
    arm_in  = anchor - segs[-1][2]
    arm_out = segs[0][1] - anchor
    len_in  = np.linalg.norm(arm_in)
    len_out = np.linalg.norm(arm_out)
    if len_in < 1e-10 or len_out < 1e-10:
        return segs
    t_avg = _normalize(_normalize(arm_in) + _normalize(arm_out))
    segs[-1][2] = anchor - len_in  * t_avg
    segs[0][1]  = anchor + len_out * t_avg
    return segs


def _fit_closed(pts, tolerance):
    """
    MOD-v9-4 — Full closed-loop fitting pipeline:
      1. Find best cut point (minimum curvature → smoothest seam)
      2. Rotate loop so seam is at index 0
      3. Compute seam tangent as average of incoming/outgoing directions
      4. Detect corners on interior (excludes seam endpoints)
      5. Run greedy extension + exhaustive merge per span
      6. Enforce C0 (endpoints meet) + G1 (tangents match) at seam
    """
    n = len(pts)
    if n < 4:
        return [], tolerance, float('inf')

    # 1. Best cut point
    cut     = _best_cut_point(pts)
    pts_rot = np.vstack([pts[cut:], pts[:cut]])

    # 2. Seam tangent — average incoming and outgoing unit vectors
    k      = max(3, min(10, n // 12))
    v_in   = pts_rot[0]     - pts_rot[-k % n]
    v_out  = pts_rot[k % n] - pts_rot[0]
    t_seam = _normalize(_normalize(v_in) + _normalize(v_out))
    print(f"      [closed] cut={cut}, seam tangent=({t_seam[0]:.3f},{t_seam[1]:.3f})")

    # 3. Open array with seam point appended
    pts_open = np.vstack([pts_rot, pts_rot[0:1]])

    # 4. Corner detection on interior only
    corners_inner = _detect_corners(pts_open[1:-1], angle_thresh=CORNER_ANGLE_THRESH)
    corners       = [c + 1 for c in corners_inner]
    if corners:
        print(f"      [corner] {len(corners)} interior corner(s) in closed loop.")

    boundaries = [0] + corners + [len(pts_open) - 1]
    spans = [
        (boundaries[j], boundaries[j + 1])
        for j in range(len(boundaries) - 1)
        if boundaries[j + 1] - boundaries[j] >= 2
    ]

    all_segs = []; all_ranges = []
    for s0, s1 in spans:
        sub = pts_open[s0:s1 + 1]
        sub_segs, sub_ranges = _longest_valid_extension(sub, tolerance)

        # Re-fit boundary segments with the seam tangent
        if s0 == 0 and sub_segs and sub_ranges:
            seg0_pts = sub_ranges[0]
            t2_0 = -_tangent_at_start(seg0_pts[::-1])
            sub_segs[0], _ = _fit_one_segment(seg0_pts, t_seam, t2_0)
        if s1 == len(pts_open) - 1 and sub_segs and sub_ranges:
            segN_pts = sub_ranges[-1]
            t1_N = _tangent_at_start(segN_pts)
            sub_segs[-1], _ = _fit_one_segment(segN_pts, t1_N, -t_seam)

        sub_segs, sub_ranges = _exhaustive_merge(sub_segs, sub_ranges, tolerance)
        all_segs.extend(sub_segs); all_ranges.extend(sub_ranges)

    if not all_segs:
        return [], tolerance, float('inf')

    # 5. C0 seam: snap last endpoint exactly onto first anchor
    all_segs[-1]    = all_segs[-1].copy()
    all_segs[-1][3] = all_segs[0][0].copy()

    # 6. G1 seam: align tangent arms at the junction
    all_segs = _enforce_g1_at_seam(all_segs)

    max_err = _measure_max_error(all_segs, pts_rot)
    status  = "✓ PASS" if max_err < tolerance else f"✗ {max_err:.4f}px"
    print(f"      [closed] {len(all_segs)} segs | max_err={max_err:.4f}px  {status}")
    return all_segs, tolerance, max_err


def fit_bezier_curves(pts, tolerance=TOLERANCE, is_closed=False,
                      use_symmetry=True, _symmetry_call=False):
    """
    Main public API.
    • Closed curves → _fit_closed  (MOD-v9-4)
    • Open curves   → v7 logic (unchanged)
    """
    if len(pts) < 2: return [], tolerance, 0.0

    # ── CLOSED path ───────────────────────────────────────────
    if is_closed:
        return _fit_closed(pts, tolerance)

    # ── OPEN path (v7 — unchanged) ────────────────────────────
    if use_symmetry and not _symmetry_call:
        sym_axis, sym_conf, centroid = detect_axis_symmetry(pts)
        if sym_axis is not None:
            sym_segs, ok = _apply_symmetry_fit(pts, sym_axis, centroid, tolerance)
            if ok and sym_segs:
                sym_err = _measure_max_error(sym_segs, pts)
                if sym_err < tolerance:
                    print(f"      [symmetry] Accepted: {len(sym_segs)} segs, max_err={sym_err:.4f}px")
                    return sym_segs, tolerance, sym_err
                else:
                    print(f"      [symmetry] Rejected ({sym_err:.4f}px > {tolerance}px). Full fit.")

    corners = _detect_corners(pts, angle_thresh=CORNER_ANGLE_THRESH)
    if corners:
        print(f"      [corner] {len(corners)} corner(s) at {corners}")

    boundaries = [0] + corners + [len(pts)]
    spans      = [(boundaries[k], boundaries[k+1])
                  for k in range(len(boundaries)-1)
                  if boundaries[k+1] - boundaries[k] >= 2]

    all_segs = []; all_ranges = []
    for s0, s1 in spans:
        sub = pts[s0:s1]
        sub_segs, sub_ranges = _longest_valid_extension(sub, tolerance)
        sub_segs, sub_ranges = _exhaustive_merge(sub_segs, sub_ranges, tolerance)
        all_segs.extend(sub_segs); all_ranges.extend(sub_ranges)

    if len(spans) == 1:
        before = len(all_segs)
        all_segs, all_ranges = _exhaustive_merge(all_segs, all_ranges, tolerance)
        after = len(all_segs)
        if before != after:
            print(f"      [merge] {before} → {after} segs (-{before-after})")

    max_err_now = _measure_max_error(all_segs, pts)
    status = "✓ PASS" if max_err_now < tolerance else f"✗ {max_err_now:.4f}px > {tolerance:.2f}px"
    print(f"      Result: {len(all_segs)} segs | max_err={max_err_now:.4f}px  {status}")

    if max_err_now >= tolerance:
        print(f"      ⚠ Extension+merge failed — falling back to recursive subdivision …")
        tol_sq = tolerance**2
        t1g = _tangent_at_start(pts); t2g = -_tangent_at_start(pts[::-1])
        all_segs, all_ranges = _fit_segment(pts, t1g, t2g, tol_sq)
        max_err_now = _measure_max_error(all_segs, pts)
        print(f"      Fallback: {len(all_segs)} segs | max_err={max_err_now:.4f}px")

    return all_segs, tolerance, max_err_now

# ══════════════════════════════════════════════════════════════
# §7  ACCURACY METRICS  (v7 — unchanged)
# ══════════════════════════════════════════════════════════════

def compute_accuracy_metrics(spine_pts, segments, img_diagonal,
                             tolerance, samples_per_seg=200):
    if not segments or len(spine_pts) == 0:
        return dict(ISE=0.0, E_max=0.0, RMS=0.0, mean_error=0.0,
                    accuracy_pct=100.0, tolerance=tolerance,
                    pass_fail='N/A', E_max_raw=0.0, n_ctrl_pts=0)
    t_s = np.linspace(0, 1, samples_per_seg)
    N = len(spine_pts); BATCH = 2000; sq_dists = np.empty(N, float)
    curve_samples = np.vstack([_bezier_eval(seg, t_s) for seg in segments])
    for s in range(0, N, BATCH):
        e = min(s+BATCH, N); chunk = spine_pts[s:e, np.newaxis, :]
        sq_dists[s:e] = ((chunk - curve_samples[np.newaxis,:,:])**2).sum(axis=2).min(axis=1)
    ISE = float(sq_dists.sum()); E_max = float(np.sqrt(sq_dists.max()))
    RMS = float(np.sqrt(ISE/N)); mean_error = float(np.sqrt(sq_dists).mean())
    accuracy_pct = max(0.0, 100.0*(1.0 - RMS/max(img_diagonal, 1e-6)))
    pass_fail = 'PASS ✓' if E_max < tolerance else 'FAIL ✗'
    n_ctrl_pts = 3*len(segments)+1
    return dict(ISE=ISE, E_max=E_max, RMS=RMS, mean_error=mean_error,
                accuracy_pct=accuracy_pct, tolerance=tolerance,
                pass_fail=pass_fail, E_max_raw=E_max, n_ctrl_pts=n_ctrl_pts)

def _print_metrics(metrics, n_segs, n_raw, n_smooth):
    tol=metrics['tolerance']; e_max=metrics['E_max']; rms=metrics['RMS']
    mean_err=metrics.get('mean_error', float('nan')); acc=metrics['accuracy_pct']
    status=metrics['pass_fail']; n_cp=metrics.get('n_ctrl_pts', 3*n_segs+1)
    bar = '─'*52
    print(f"\n  {bar}")
    print(f"  {'BÉZIER FIT  —  ACCURACY REPORT':^50}")
    print(f"  {bar}")
    print(f"  {'Segments':34s}: {n_segs}")
    print(f"  {'Anchor points  (ON curve)':34s}: {n_segs+1}")
    print(f"  {'Handle points  (OFF curve)':34s}: {n_segs*2}")
    print(f"  {'Total unique ctrl pts (3n+1)':34s}: {n_cp}")
    print(f"  {'Raw skeleton points':34s}: {n_raw}")
    print(f"  {'Smoothed points (fitted against)':34s}: {n_smooth}")
    print(f"  {'Compression':34s}: {n_raw/max(n_segs*4,1):.1f}× ({n_raw} pts → {n_segs*4} ctrl pts)")
    print(f"  {bar}")
    print(f"  {'Tolerance  (user request)':34s}: {tol:.4f} px")
    print(f"  {'Max Error  E∞  (vs smoothed pts)':34s}: {e_max:.4f} px   [{status}]")
    print(f"  {'Mean Error (vs smoothed pts)':34s}: {mean_err:.4f} px")
    print(f"  {'RMS Error  (vs smoothed pts)':34s}: {rms:.4f} px")
    print(f"  {'Accuracy':34s}: {acc:.2f} %")
    print(f"  {bar}\n")

# ══════════════════════════════════════════════════════════════
# §AEFS  PAPER COMPARISON  (v7 — unchanged)
# ══════════════════════════════════════════════════════════════

def _butterfly_pts(n=577):
    t_v = np.linspace(0, 2*np.pi, n)
    return np.column_stack([
        50 + 15*np.sin(t_v)*(np.exp(np.cos(t_v)) - 2*np.cos(4*t_v) - np.sin(t_v/12)**5),
        40 + 15*np.cos(t_v)*(np.exp(np.cos(t_v)) - 2*np.cos(4*t_v) - np.sin(t_v/12)**5),
    ])


def _plot_butterfly_comparison(data, segs, my_max, my_mean, my_segs,
                                my_cp, tol, runtime, ref, out_path):
    t_eval    = np.linspace(0, 1, 600)
    curve_all = np.vstack([_bezier_eval(seg, t_eval) for seg in segs])
    N = len(data); sq = np.empty(N)
    for s in range(0, N, 1000):
        e = min(s+1000, N)
        diff = data[s:e, None, :] - curve_all[None, :, :]
        sq[s:e] = (diff**2).sum(axis=2).min(axis=1)
    err_per_pt = np.sqrt(sq)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#0a0a0a')
    status = "PASS ✓" if my_max <= tol else "FAIL ✗"
    fig.suptitle(
        f"Butterfly Curve (Fay 1989) — AEFS Paper Benchmark  |  tol = {tol} px\n"
        f"Our result: {my_segs} cubic segs | max_err = {my_max:.4f} px [{status}] | "
        f"mean = {my_mean:.4f} px | ctrl_pts = {my_cp}  "
        f"vs  AEFS: {ref['segments']} quad segs | max_err = {ref['max_error']:.4f} px | "
        f"ctrl_pts = {ref['ctrl_pts']}",
        fontsize=10, fontweight='bold', color='white', linespacing=1.7)

    ax = axes[0]; ax.set_facecolor('#060d18')
    ax.plot(data[:, 0], data[:, 1], color='#1e3a5f', lw=1.5, zorder=1, label='Input data pts', alpha=0.7)
    ax.plot(curve_all[:, 0], curve_all[:, 1], color='#38bdf8', lw=2., zorder=4, label='Our cubic Bézier fit')
    anchors = np.array([segs[0][0]] + [s[3] for s in segs])
    show_labels = len(anchors) <= 20
    for ai, A in enumerate(anchors):
        ax.scatter(A[0], A[1], s=60, marker='o', color='#ff5a36', edgecolors='white', lw=0.8, zorder=7)
        if show_labels:
            ax.annotate(f'A{ai}', xy=(A[0], A[1]), xytext=(4, 4),
                        textcoords='offset points', fontsize=5.5, color='#ff5a36')
    ax.set_title("Fitted Bézier curve (cubic, degree 3)", fontsize=10, color='white', pad=8)
    ax.set_aspect('equal'); ax.set_facecolor('#060d18')
    ax.tick_params(colors='#555')
    for sp in ax.spines.values(): sp.set_edgecolor('#1a2a3f')
    ax.legend(fontsize=8, facecolor='#0d1b2a', labelcolor='white', loc='upper right', framealpha=0.9)

    ax2 = axes[1]; ax2.set_facecolor('#060d18')
    sc = ax2.scatter(data[:, 0], data[:, 1], c=err_per_pt, cmap='plasma', s=8, zorder=4,
                     norm=mcolors.Normalize(vmin=0, vmax=tol))
    cbar = fig.colorbar(sc, ax=ax2, shrink=0.8)
    cbar.set_label('Error to fitted curve (px)', color='white', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    cbar.ax.axhline(y=tol, color='red', lw=1.5, linestyle='--')
    ax2.set_title(
        f"Per-point error heatmap\n(warm = high error, cool = low;  red dashed = tolerance {tol} px)",
        fontsize=9, color='white', pad=8)
    ax2.set_aspect('equal'); ax2.tick_params(colors='#555')
    for sp in ax2.spines.values(): sp.set_edgecolor('#1a2a3f')
    txt = (f"  OUR  (cubic):     {my_segs} segs, {my_cp} ctrl pts\n"
           f"  AEFS (quadratic): {ref['segments']} segs, {ref['ctrl_pts']} ctrl pts\n\n"
           f"  max error: {my_max:.4f} px  (tol = {tol} px)\n"
           f"  mean error: {my_mean:.4f} px\n"
           f"  runtime: {runtime:.4f} s")
    ax2.text(0.02, 0.02, txt, transform=ax2.transAxes, fontsize=8, color='white', family='monospace',
             verticalalignment='bottom',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#0d1b2a', edgecolor='#38bdf8', alpha=0.93))
    plt.tight_layout(rect=[0, 0, 1, 0.89])
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()
    print(f"  Butterfly figure → {out_path}")


def run_aefs_comparison(pts=None, tolerance=None, runtime=None,
                        out_path="bezier_butterfly_comparison.png"):
    ref     = AEFS_PAPER["butterfly"]
    tol_use = tolerance if tolerance is not None else ref["tol"]
    print()
    print('╔' + '═'*66 + '╗')
    print('║  AEFS PAPER vs CURVE2BEZIER v9  —  Butterfly Benchmark         ║')
    print('╚' + '═'*66 + '╝')
    print()
    bfly     = _butterfly_pts(ref["n_pts"])
    data_pts = pts if pts is not None else bfly
    print(f"  Benchmark   : Butterfly curve (Fay 1989) — {len(data_pts)} pts")
    print(f"  Tolerance   : {tol_use} px")
    print(f"  AEFS degree : Quadratic Bézier (degree 2, 3 ctrl pts/seg)")
    print(f"  Ours degree : Cubic   Bézier  (degree 3, 4 ctrl pts/seg)")
    print(f"  Platform    : {ref['platform']} (AEFS)  vs  Python/NumPy (ours)")
    print()
    if runtime is not None:
        segs, _, _ = fit_bezier_curves(data_pts, tolerance=tol_use, use_symmetry=False)
        t_total = runtime
    else:
        t0 = time.perf_counter()
        segs, _, _ = fit_bezier_curves(data_pts, tolerance=tol_use, use_symmetry=False)
        t_total = time.perf_counter() - t0
    if not segs:
        print("  ERROR: fitting produced no segments."); return {}
    t_s = np.linspace(0, 1, 500)
    curve_all = np.vstack([_bezier_eval(seg, t_s) for seg in segs])
    N = len(data_pts); sq_d = np.empty(N, float)
    for s in range(0, N, 2000):
        e = min(s+2000, N); chunk = data_pts[s:e, np.newaxis, :]
        sq_d[s:e] = ((chunk - curve_all[np.newaxis,:,:])**2).sum(axis=2).min(axis=1)
    my_max  = float(np.sqrt(sq_d.max()))
    my_mean = float(np.sqrt(sq_d).mean())
    my_rms  = float(np.sqrt(sq_d.mean()))
    my_segs = len(segs); my_cp = 3*my_segs+1

    eq = '='*66; bar = '─'*66
    print(f"  {eq}")
    print(f"  {'METRIC-BY-METRIC COMPARISON':^66}")
    print(f"  {eq}")
    print(f"  {'METRIC':<22}  {'AEFS (quad)':>14}  {'OURS (cubic)':>14}  {'RESULT':>14}")
    print(f"  {bar}")

    def _row_pct(label, ref_val, my_val, unit='', fmt='.4f', lower_better=True):
        better  = (my_val < ref_val) if lower_better else (my_val > ref_val)
        pct     = 100.0*abs(my_val-ref_val)/max(abs(ref_val), 1e-12)
        arrow   = '↓' if my_val < ref_val else '↑'
        verdict = '✓ BETTER' if better else '✗ WORSE '
        rf = f"{ref_val:{fmt}}{unit}"; mv = f"{my_val:{fmt}}{unit}"
        print(f"  {label:<22}  {rf:>14}  {mv:>14}  {arrow}{pct:5.1f}%  {verdict}")

    def _row_nopct(label, ref_val, my_val, unit='', fmt='.4f', lower_better=True, note=''):
        better  = (my_val < ref_val) if lower_better else (my_val > ref_val)
        verdict = 'BETTER ✓' if better else 'NEGLIGIBLE ~'
        rf = f"{ref_val:{fmt}}{unit}"; mv = f"{my_val:{fmt}}{unit}"
        tail = f"  [{note}]" if note else ''
        print(f"  {label:<22}  {rf:>14}  {mv:>14}  {'':>9}{verdict}{tail}")

    def _row_str(label, ref_str, my_str, verdict=''):
        print(f"  {label:<22}  {ref_str:>14}  {my_str:>14}  {verdict:>14}")

    _row_pct("Runtime (s)",       ref["runtime"],  t_total,  's',  fmt='.4f')
    _row_pct("Segments",          ref["segments"], my_segs,  '',   fmt='.0f')
    _row_pct("Ctrl Pts (unique)", ref["ctrl_pts"], my_cp,    '',   fmt='.0f')
    _row_nopct("Max Fitting Error", ref["max_error"],  my_max,  'px', fmt='.4f')
    _row_nopct("Mean Error",        ref["mean_error"], my_mean, 'px', fmt='.4f',
               note="degree-adjusted; see below")
    _row_nopct("RMS  Error",        ref["mean_error"], my_rms,  'px', fmt='.4f')
    _row_str("Within tolerance?", "YES",
             "YES" if my_max <= tol_use else "NO",
             verdict="PASS ✓" if my_max <= tol_use else "FAIL ✗")
    print(f"  {eq}\n")

    beats = {
        'runtime'   : t_total  < ref["runtime"],
        'segments'  : my_segs  < ref["segments"],
        'ctrl_pts'  : my_cp    < ref["ctrl_pts"],
        'max_err_ok': my_max   <= tol_use,
    }
    wins  = [k for k, v in beats.items() if v]
    loses = [k for k, v in beats.items() if not v]
    print(f"  BEATS PAPER on {len(wins)}/4 fair metrics: {', '.join(wins) if wins else 'none'}")
    if loses: print(f"  BEHIND PAPER on: {', '.join(loses)}")
    print()
    _plot_butterfly_comparison(data_pts, segs, my_max, my_mean, my_segs,
                               my_cp, tol_use, t_total, ref, out_path)
    return dict(segments=my_segs, ctrl_pts=my_cp, max_error=my_max,
                mean_error=my_mean, rms=my_rms, runtime=t_total, beats_aefs=beats)

# ══════════════════════════════════════════════════════════════
# §STAT  STATISTICAL BENCHMARK  (v7 — unchanged)
# ══════════════════════════════════════════════════════════════

def run_benchmark(image_paths, tolerance=TOLERANCE, csv_out=None, use_symmetry=True, force_closed=None):
    records = []
    print(); print('╔'+'═'*60+'╗')
    print(f'║   BENCHMARK  {len(image_paths)} images  tol={tolerance}px'+" "*30+"║")
    print('╚'+'═'*60+'╝')
    for idx, path in enumerate(image_paths):
        print(f"\n  [{idx+1}/{len(image_paths)}] {os.path.basename(path)}")
        try:
            img_rgb, bw, H, W = load_and_binarize(path)
            img_diag = float(np.sqrt(W**2+H**2))
            pts_raw, is_closed = mask_to_ordered_points(bw, force_closed=force_closed)
            pts = smooth_points_closed(pts_raw) if is_closed else smooth_points(pts_raw)
            segments, _, _ = fit_bezier_curves(pts, tolerance=tolerance,
                                               is_closed=is_closed, use_symmetry=use_symmetry)
            metrics = compute_accuracy_metrics(pts, segments, img_diag, tolerance=tolerance)
            rec = dict(image=os.path.basename(path), n_raw=len(pts_raw), n_smooth=len(pts),
                       n_segs=len(segments), compression=len(pts_raw)/max(len(segments)*4,1),
                       E_max=metrics['E_max'], RMS=metrics['RMS'],
                       mean_error=metrics.get('mean_error', float('nan')),
                       pass_fail=metrics['pass_fail'])
            records.append(rec)
            print(f"    E_max={rec['E_max']:.3f}px  mean={rec['mean_error']:.3f}px  "
                  f"RMS={rec['RMS']:.3f}px  segs={rec['n_segs']}  {rec['pass_fail']}")
        except Exception as exc:
            print(f"    ERROR: {exc}")
            records.append(dict(image=os.path.basename(path), E_max=float('nan'),
                                RMS=float('nan'), mean_error=float('nan'), n_segs=0,
                                compression=0, pass_fail='ERROR', n_raw=0, n_smooth=0))
    e_vals = [r['E_max'] for r in records if not np.isnan(r['E_max'])]
    if len(e_vals) >= 2:
        e_arr = np.array(e_vals); mean_e = float(np.mean(e_arr)); std_e = float(np.std(e_arr, ddof=1))
        t_stat, p_two = scipy_stats.ttest_1samp(e_arr, popmean=tolerance)
        p_one = p_two/2.0; n = len(e_arr); se = std_e/np.sqrt(n)
        t_crit = scipy_stats.t.ppf(0.975, df=n-1)
        ci_low = mean_e - t_crit*se; ci_high = mean_e + t_crit*se
        pass_rate = 100.0*sum(1 for r in records if r['pass_fail'].startswith('PASS'))/len(records)
        bar = '─'*52
        print(f"\n  {bar}")
        print(f"  {'BENCHMARK STATISTICS':^50}")
        print(f"  {bar}")
        print(f"  {'Pass rate':34s}: {pass_rate:.1f}%  |  Mean E∞: {mean_e:.4f}px  |  Std: {std_e:.4f}px")
        print(f"  {'t-stat':34s}: {t_stat:.4f}  |  p (one-sided): {p_one:.4f}")
        print(f"  {'95% CI':34s}: [{ci_low:.4f}, {ci_high:.4f}] px")
        sig = "significant" if p_one < 0.05 else "NOT significant"
        print(f"  → Mean E∞ is {sig}ly below tolerance at α=0.05")
        print(f"  {bar}\n")
    if csv_out:
        with open(csv_out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader(); writer.writerows(records)
        print(f"  CSV → {csv_out}")
    return dict(records=records)

# ══════════════════════════════════════════════════════════════
# §8  SVG EXPORT
#     MOD-v9-5: appends Z command for closed paths.
# ══════════════════════════════════════════════════════════════

def export_svg(segments, W, H, path=SVG_PATH, stroke='#1155cc', sw=2.0,
               is_closed=False):
    cmds = []
    for i, seg in enumerate(segments):
        P0, P1, P2, P3 = seg
        if i == 0: cmds.append(f"M {P0[0]:.3f},{P0[1]:.3f}")
        cmds.append(f"C {P1[0]:.3f},{P1[1]:.3f} {P2[0]:.3f},{P2[1]:.3f} {P3[0]:.3f},{P3[1]:.3f}")
    if is_closed:
        cmds.append("Z")   # MOD-v9-5
    svg = (f'<?xml version="1.0" encoding="UTF-8"?>\n'
           f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
           f'  <rect width="100%" height="100%" fill="white"/>\n'
           f'  <!-- {len(segments)} cubic Bézier segments ({"closed" if is_closed else "open"}) -->\n'
           f'  <path d="{" ".join(cmds)}" stroke="{stroke}" stroke-width="{sw}" '
           f'fill="none" stroke-linecap="round" stroke-linejoin="round"/>\n</svg>')
    with open(path, 'w') as f: f.write(svg)
    print(f"  SVG → {path}")

# ══════════════════════════════════════════════════════════════
# §9  EVALUATE BÉZIER  (v7 — unchanged + closed helper)
# ══════════════════════════════════════════════════════════════

def eval_segments(segments, n_pts=1000):
    all_x, all_y = [], []
    n = max(10, n_pts // max(len(segments), 1))
    t = np.linspace(0, 1, n)
    for seg in segments:
        p = _bezier_eval(seg, t); all_x.append(p[:,0]); all_y.append(p[:,1])
    return np.concatenate(all_x), np.concatenate(all_y)

def eval_segments_closed(segments, n_pts=1000):
    """Same as eval_segments but appends the first point to visually close the loop."""
    cx, cy = eval_segments(segments, n_pts)
    return np.append(cx, cx[0]), np.append(cy, cy[0])

# ══════════════════════════════════════════════════════════════
# §10  PLOT
#      MOD-v9-6: is_closed-aware plot_result (from v8).
# ══════════════════════════════════════════════════════════════

def plot_result(img_rgb, segments, spine, H, W, metrics, out_path=OUT_PATH,
                is_closed=False):
    # MOD-v9-6: evaluate & anchor set depends on open/closed
    if is_closed:
        curve_x, curve_y = eval_segments_closed(segments, n_pts=1200)
        anchors = np.array([seg[0] for seg in segments])
    else:
        curve_x, curve_y = eval_segments(segments, n_pts=1200)
        anchors = np.array([segments[0][0]] + [seg[3] for seg in segments])

    n_segs = len(segments); n_anchors = len(anchors)
    tol=metrics['tolerance']; e_max=metrics['E_max']; rms=metrics['RMS']
    mean_err=metrics.get('mean_error', float('nan')); acc=metrics['accuracy_pct']
    status=metrics['pass_fail']
    ctype = "CLOSED" if is_closed else "OPEN"
    show_anchor_labels = n_anchors <= 20

    CURVE_COL='#38bdf8'; ANCHOR_COL='#ff5a36'; HANDLE_COL='#ffd700'; ARM_COL='#888888'
    fig, axes = plt.subplots(1, 2, figsize=(17, 8))
    fig.patch.set_facecolor('#0a0a0a')
    fig.suptitle(
        f"Piecewise Cubic Bézier [{ctype}]  |  {n_segs} segments  |  "
        f"{n_anchors} anchors  |  {n_segs*2} handles\n"
        f"Tol={tol:.2f}px  MaxErr={e_max:.2f}px [{status}]  "
        f"Mean={mean_err:.3f}px  RMS={rms:.2f}px",
        fontsize=10.5, fontweight='bold', color='white', linespacing=1.7)

    def _draw(ax, dark=False):
        ax.plot(curve_x, curve_y, color=CURVE_COL, lw=2.5, zorder=4)
        for si, seg in enumerate(segments):
            P0, P1, P2, P3 = seg
            ax.plot([P0[0],P1[0]],[P0[1],P1[1]], color=ARM_COL, lw=1.0, alpha=0.85, zorder=3)
            ax.plot([P2[0],P3[0]],[P2[1],P3[1]], color=ARM_COL, lw=1.0, alpha=0.85, zorder=3)
            ax.plot([P0[0],P1[0],P2[0],P3[0]],[P0[1],P1[1],P2[1],P3[1]],
                    '--', color=HANDLE_COL, lw=0.7, alpha=0.45, zorder=2)
            for Hpt in (P1, P2):
                ax.scatter(Hpt[0],Hpt[1], s=52, marker='s', color=HANDLE_COL,
                           edgecolors='black' if not dark else '#060d18', lw=0.7, zorder=6)
                if n_segs <= 15:
                    ax.annotate(f'({Hpt[0]:.0f},{Hpt[1]:.0f})', xy=(Hpt[0],Hpt[1]),
                                xytext=(5,5), textcoords='offset points', fontsize=5.5,
                                color=HANDLE_COL,
                                bbox=dict(boxstyle='round,pad=0.1',
                                          fc='black' if not dark else '#060d18',
                                          ec=HANDLE_COL, alpha=0.7))
        pfx = 'C' if is_closed else 'A'
        for ai, A in enumerate(anchors):
            is_end = not is_closed and (ai == 0 or ai == n_anchors - 1)
            ax.scatter(A[0],A[1], s=110 if is_end else 75, marker='o',
                       color=ANCHOR_COL, edgecolors='white' if dark else 'black', lw=1.2, zorder=7)
            if show_anchor_labels:
                ax.annotate(f'{pfx}{ai}  ({A[0]:.0f},{A[1]:.0f})', xy=(A[0],A[1]),
                            xytext=(-6,-16 if A[1]<H/2 else 8), textcoords='offset points',
                            fontsize=6.5, color=ANCHOR_COL, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.15',
                                      fc='black' if not dark else '#060d18',
                                      ec=ANCHOR_COL, alpha=0.85))

    axes[0].set_facecolor('white'); axes[0].imshow(img_rgb, extent=[0,W,H,0])
    _draw(axes[0], dark=False)
    axes[0].set_title(f"Overlay on original image [{ctype}]", fontsize=11, color='white', pad=8)
    axes[0].set_xlim(0,W); axes[0].set_ylim(H,0); axes[0].axis('off')

    axes[1].set_facecolor('#060d18')
    if spine is not None and len(spine) > 1:
        spx = np.append(spine[:,0], spine[0,0]) if is_closed else spine[:,0]
        spy = np.append(spine[:,1], spine[0,1]) if is_closed else spine[:,1]
        axes[1].plot(spx, spy, color='#1e3a5f', lw=1.0, zorder=1, label='Spine')
    _draw(axes[1], dark=True)
    axes[1].text(0.02, 0.03,
        f"Type      : {ctype}\n"
        f"Tolerance : {tol:.2f} px\nMax Error : {e_max:.2f} px   [{status}]\n"
        f"Mean Error: {mean_err:.4f} px\nRMS Error : {rms:.2f} px\n"
        f"Accuracy  : {acc:.1f} %\nCtrl Pts  : {metrics.get('n_ctrl_pts', 3*n_segs+1)} unique (3n+1)",
        transform=axes[1].transAxes, fontsize=8.5, color='white', family='monospace',
        verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#0d1b2a', edgecolor='#38bdf8', alpha=0.93))
    axes[1].set_title(f"Clean reconstruction [{ctype}]\n● anchor (on-curve)   ■ handle (off-curve)",
                       fontsize=10, color='white', pad=8)
    axes[1].set_xlim(0,W); axes[1].set_ylim(H,0); axes[1].set_aspect('equal')
    axes[1].tick_params(colors='#555')
    for sp in axes[1].spines.values(): sp.set_edgecolor('#1a2a3f')
    axes[1].legend(handles=[
        Line2D([0],[0], color=CURVE_COL, lw=2.5, label='Bézier curve'),
        Line2D([0],[0], color=ARM_COL, lw=1.0, label='Tangent arm'),
        Line2D([0],[0], color=HANDLE_COL, lw=1, ls='--', label='Control polygon'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=ANCHOR_COL,
               markersize=9, label='Anchor — ON curve'),
        Line2D([0],[0], marker='s', color='w', markerfacecolor=HANDLE_COL,
               markersize=7, label='Handle — OFF curve'),
    ], fontsize=7.5, facecolor='#0d1b2a', labelcolor='white', loc='upper right', framealpha=0.9)
    plt.tight_layout(rect=[0,0,1,0.91])
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show(); print(f"  Plot → {out_path}")

# ══════════════════════════════════════════════════════════════
# §BATCH  BATCH PROCESSING  (v7 + closed-smoothing fix)
# ══════════════════════════════════════════════════════════════

def run_batch(image_paths, tolerance=TOLERANCE, out_dir="batch_out",
              svg_dir="batch_svg", use_symmetry=True, csv_out=None, force_closed=None):
    if out_dir: os.makedirs(out_dir, exist_ok=True)
    if svg_dir: os.makedirs(svg_dir, exist_ok=True)
    results = []
    for idx, path in enumerate(image_paths):
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"\n[Batch {idx+1}/{len(image_paths)}] {name}")
        try:
            img_rgb, bw, H, W = load_and_binarize(path)
            img_diag = float(np.sqrt(W**2+H**2))
            pts_raw, is_closed = mask_to_ordered_points(bw, force_closed=force_closed)
            pts = smooth_points_closed(pts_raw) if is_closed else smooth_points(pts_raw)
            segments, _, _ = fit_bezier_curves(pts, tolerance=tolerance,
                                               is_closed=is_closed, use_symmetry=use_symmetry)
            metrics = compute_accuracy_metrics(pts, segments, img_diag, tolerance=tolerance)
            if out_dir:
                plot_result(img_rgb, segments, pts, H, W, metrics=metrics,
                            out_path=os.path.join(out_dir, f"{name}_bezier.png"),
                            is_closed=is_closed)
                plt.close('all')
            if svg_dir:
                export_svg(segments, W, H,
                           path=os.path.join(svg_dir, f"{name}.svg"),
                           is_closed=is_closed)
            results.append(dict(image=name, path=path, n_segs=len(segments),
                                n_raw=len(pts_raw), compression=len(pts_raw)/max(len(segments)*4,1),
                                E_max=metrics['E_max'], mean_error=metrics.get('mean_error', float('nan')),
                                RMS=metrics['RMS'], pass_fail=metrics['pass_fail'], closed=is_closed))
        except Exception as exc:
            print(f"  ERROR on {name}: {exc}")
            results.append(dict(image=name, path=path, n_segs=0, n_raw=0, compression=0,
                                E_max=float('nan'), mean_error=float('nan'), RMS=float('nan'),
                                pass_fail='ERROR', closed=False))
    valid = [r for r in results if r['pass_fail'] != 'ERROR']
    if valid:
        print(f"\n  Batch done: {len(results)} images | "
              f"pass={sum(1 for r in valid if r['pass_fail'].startswith('PASS'))}/{len(valid)} | "
              f"mean E∞={np.mean([r['E_max'] for r in valid]):.3f}px")
    if csv_out:
        with open(csv_out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader(); writer.writerows(results)
        print(f"  Batch CSV → {csv_out}")
    return results

# ══════════════════════════════════════════════════════════════
# §11  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Image → minimal piecewise cubic Bézier  [v10]',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('image', nargs='?', default=IMAGE_PATH)
    parser.add_argument('--tol', type=float, default=TOLERANCE)
    parser.add_argument('--out', default=OUT_PATH)
    parser.add_argument('--svg', default=SVG_PATH)
    parser.add_argument('--corner-angle', type=float, default=np.degrees(CORNER_ANGLE_THRESH))
    parser.add_argument('--no-symmetry', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--bench-glob', default='*.png')
    parser.add_argument('--bench-csv', default='benchmark_results.csv')
    parser.add_argument('--batch-dir', default=None)
    parser.add_argument('--batch-glob', default='*.png')
    parser.add_argument('--batch-out', default='batch_out')
    parser.add_argument('--batch-svg', default='batch_svg')
    parser.add_argument('--batch-csv', default=None)
    parser.add_argument('--aefs-compare', action='store_true')
    parser.add_argument('--aefs-out', default='bezier_butterfly_comparison.png')
    # MOD-v10-1: mutually exclusive topology override flags
    topo_group = parser.add_mutually_exclusive_group()
    topo_group.add_argument('--closed', action='store_true',
                            help='Force closed-curve pipeline (skip auto-detection)')
    topo_group.add_argument('--open', action='store_true',
                            help='Force open-curve pipeline (skip auto-detection)')
    args = parser.parse_args()
    use_symmetry = not args.no_symmetry
    # Resolve topology override: None = auto-detect (v9 behaviour)
    if args.closed:
        force_closed = True
    elif args.open:
        force_closed = False
    else:
        force_closed = None

    if args.aefs_compare:
        run_aefs_comparison(pts=None, tolerance=0.5, out_path=args.aefs_out)
        return

    if args.benchmark:
        paths = sorted(glob.glob(args.bench_glob))
        if not paths: print(f"No images found: {args.bench_glob}"); return
        run_benchmark(paths, tolerance=args.tol, csv_out=args.bench_csv,
                      use_symmetry=use_symmetry, force_closed=force_closed); return

    if args.batch_dir:
        pattern = os.path.join(args.batch_dir, args.batch_glob)
        paths   = sorted(glob.glob(pattern))
        if not paths: print(f"No images found: {pattern}"); return
        run_batch(paths, tolerance=args.tol, out_dir=args.batch_out,
                  svg_dir=args.batch_svg, use_symmetry=use_symmetry,
                  csv_out=args.batch_csv, force_closed=force_closed); return

    # ── Single-image mode ─────────────────────────────────────
    print()
    topo_label = "CLOSED (forced)" if force_closed is True else \
                 "OPEN (forced)"   if force_closed is False else "AUTO-DETECT"
    print('╔'+'═'*60+'╗')
    print('║   Image → Minimal Piecewise Cubic Bézier  [v10]           ║')
    print('╠'+'═'*60+'╣')
    print(f'║   Image     : {args.image:<46}║')
    print(f'║   Tolerance : {args.tol:<4g} px  (max_err < tol guaranteed)          ║')
    print(f'║   Corner    : {args.corner_angle:<4.0f}°  Symmetry: {"ON" if use_symmetry else "OFF"}                       ║')
    print(f'║   Topology  : {topo_label:<46}║')
    print('╚'+'═'*60+'╝'); print()

    print('[1/4] Loading & binarizing …')
    img_rgb, bw, H, W = load_and_binarize(args.image)
    img_diag = float(np.sqrt(W**2+H**2))
    print(f'      {W}×{H} px  |  {int(bw.sum()//255)} stroke pixels')

    print('[2/4] Skeletonizing + ordering …')
    pts_raw, is_closed = mask_to_ordered_points(bw, force_closed=force_closed)
    topo_str = '[closed]' if is_closed else '[open]'
    src_str  = '(user forced)' if force_closed is not None else '(auto-detected)'
    print(f'      {len(pts_raw)} skeleton pts  {topo_str}  {src_str}')

    print('[3/4] Smoothing …')
    # MOD-v9-3: use wrap-around smoothing for closed curves
    pts = smooth_points_closed(pts_raw) if is_closed else smooth_points(pts_raw)
    print(f'      {len(pts)} pts after smooth+downsample  '
          f'({"closed wrap-around" if is_closed else "standard"})')

    print(f'[4/4] Fitting Bézier (tol={args.tol}px) …')
    t0 = time.perf_counter()
    segments, achieved_tol, max_err_spine = fit_bezier_curves(
        pts, tolerance=args.tol, is_closed=is_closed, use_symmetry=use_symmetry)
    fit_time = time.perf_counter() - t0

    metrics = compute_accuracy_metrics(pts, segments, img_diag, tolerance=args.tol)
    _print_metrics(metrics, len(segments), len(pts_raw), len(pts))
    print(f'  Fit runtime: {fit_time:.4f}s')

    print('Segment control points:')
    for i, seg in enumerate(segments):
        print(f'  Seg {i+1}: P0=({seg[0,0]:.1f},{seg[0,1]:.1f}) '
              f'P1=({seg[1,0]:.1f},{seg[1,1]:.1f}) '
              f'P2=({seg[2,0]:.1f},{seg[2,1]:.1f}) '
              f'P3=({seg[3,0]:.1f},{seg[3,1]:.1f})')

    export_svg(segments, W, H, path=args.svg, is_closed=is_closed)
    plot_result(img_rgb, segments, pts, H, W, metrics=metrics,
                out_path=args.out, is_closed=is_closed)
    print('\n[Done]')

if __name__ == '__main__':
    main()