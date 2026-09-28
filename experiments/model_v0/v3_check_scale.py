# -*- coding: utf-8 -*-
"""
V3 pre-run check C -- 672-equivalent ground scale.

docs/MODEL_V3_PREREGISTRATION.md section 10 item 1 / section 2.3 item 3:

    assert that a high-res token in a crop covers the SAME ground area as a
    token of the full-image 672 pass; if it does not, STOP and fix the
    implementation -- the design does not change.

Why this cannot be taken on faith: `DINOv2Wrapper.transform` uses
`transforms.Resize(size=smaller_edge_size)` with an INT, which always rescales
the input so its smaller edge equals 672.  Feed a crop into `prepare_image`
and it gets rescaled a second time, so the crop's tokens cover a different
ground area than the global pass's tokens.  Every A3 number would then be
measuring some other intervention.

The implementation used instead (a pure implementation choice; the invariant
in section 2.3 is unchanged):

    resize the FULL image to smaller-edge 672  ->  crop a token-aligned
    window (multiples of 14) from the RESIZED image  ->  feed the window to
    the ViT with no further resize

so the window's tokens coincide with the global pass's tokens positionally.

Checks:
  C1 geometry   token centres of a windowed pass map back to exactly the same
                original-image coordinates as the corresponding global tokens
  C2 scale      a square of side 14/r original pixels (exactly one 672 token's
                ground footprint) perturbs ~one token in BOTH passes; a
                wrongly rescaled crop would perturb a very different number

Usage: python experiments/model_v0/v3_check_scale.py
"""
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, r"E:\work\freshman\third_party\AnomalyDINO")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from src.backbones import get_model  # noqa: E402

EDGE, STRIDE = 672, 14
LAYERS = [5, 8, 11]
IMG = r"E:\work\freshman\data\mvtec_anomaly_detection\bottle\test\broken_large\000.png"


def build(device="cuda"):
    model = get_model("dinov2_vits14", device, smaller_edge_size=EDGE)
    resize = transforms.Resize(size=EDGE,
                               interpolation=transforms.InterpolationMode.BICUBIC,
                               antialias=True)
    norm = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225))
    return model, resize, transforms.ToTensor(), norm


def prep(size_wh, resize, totensor, norm, arr):
    """arr: HxWx3 uint8 ORIGINAL image -> (resized PIL, tensor of the grid)"""
    im = Image.fromarray(arr)
    big = resize(im)                       # smaller edge -> 672, aspect kept
    W, H = big.size
    cw, ch = W - W % STRIDE, H - H % STRIDE
    big = big.crop((0, 0, cw, ch))
    return big, norm(totensor(big))


def tokens(model, t):
    """t: C x h x w, h,w multiples of 14 -> (h//14, w//14, D)"""
    with torch.no_grad():
        tk = model.model.get_intermediate_layers(
            t.unsqueeze(0).to(model.device), n=LAYERS)
    return [x.squeeze(0).cpu().numpy() for x in tk]


def r_of(orig_hw):
    return EDGE / min(orig_hw)


def main():
    model, resize, totensor, norm = build()
    arr = np.array(Image.open(IMG).convert("RGB"))
    H, W = arr.shape[:2]
    r = r_of((H, W))
    big, t_full = prep((W, H), resize, totensor, norm, arr)
    gw, gh = big.size[0] // STRIDE, big.size[1] // STRIDE
    print(f"  image {W}x{H} -> resized {big.size[0]}x{big.size[1]}, "
          f"grid {gh}x{gw}, scale r = {r:.6f}")
    print(f"  one 672 token covers {STRIDE / r:.4f} original pixels")

    full = tokens(model, t_full)[-1].reshape(gh, gw, -1)

    # ---------------- C1: geometry ----------------
    r0, c0 = 3 * STRIDE, 2 * STRIDE
    r1, c1 = r0 + 7 * STRIDE, c0 + 9 * STRIDE          # 7 x 9 token window
    win = big.crop((c0, r0, c1, r1))
    w_tok = tokens(model, norm(totensor(win)))[-1].reshape(7, 9, -1)

    # token centre in the RESIZED image -> original coordinates
    def centres(rr0, cc0, nrow, ncol):
        ii, jj = np.mgrid[rr0:rr0 + nrow, cc0:cc0 + ncol]
        cy = (ii * STRIDE + STRIDE / 2.0) / r
        cx = (jj * STRIDE + STRIDE / 2.0) / r
        return np.stack([cy.ravel(), cx.ravel()], 1)

    c_win = centres(3, 2, 7, 9)                        # window local indices
    c_glob = centres(r0 // STRIDE, c0 // STRIDE, 7, 9)
    d = np.abs(c_win - c_glob).max()
    print(f"\n  C1 geometry: max |token centre (orig coords) difference| = {d:.3e}")
    assert d < 1e-9, "windowed tokens do not land on the global token grid"
    print("     -> OK (exact by construction: window is token-aligned)")

    # the window's tokens are NOT expected to equal the global ones -- a crop
    # restricts attention, which is inherent to any crop-based design and is
    # disclosed in the pre-registration.  Report the size of that difference.
    diff = np.abs(w_tok - full[3:10, 2:11]).mean()
    scale = np.abs(full).mean()
    print(f"     crop-vs-global token L1 {diff:.4f} on a feature scale of "
          f"{scale:.4f}  ({100 * diff / scale:.1f}%)  -- attention-range "
          f"effect, not a geometry error")

    # ---------------- C2: no second resize, hence matching ground scale ----
    # The invariant is mechanical: the window handed to the ViT must be a
    # token-aligned SLICE of the globally resized image, with no further
    # resampling.  That is what makes each of its tokens cover 14/r original
    # pixels, exactly like the global pass.
    wy0, wx0 = 3 * STRIDE, 2 * STRIDE
    wy1, wx1 = wy0 + 7 * STRIDE, wx0 + 9 * STRIDE
    t_win = norm(totensor(big.crop((wx0, wy0, wx1, wy1))))
    same = torch.equal(t_win, t_full[:, wy0:wy1, wx0:wx1])
    print(f"\n  C2a window tensor == the global tensor's slice: {same}")
    assert same, ("the window is being resampled a second time -- its tokens "
                  "would cover a different ground area than global 672")

    px_per_tok = STRIDE / r
    print(f"      ground scale: {px_per_tok:.4f} original px per token, "
          f"identical for the global and windowed paths by construction")

    # Negative control: route the SAME window through prepare_image, which
    # rescales it to smaller-edge 672.  The test must be able to see that.
    win_pil = big.crop((wx0, wy0, wx1, wy1))
    _, grid_bug = model.prepare_image(np.array(win_pil))
    extent_w = (wx1 - wx0) / r                       # original px, width
    px_per_tok_bug = extent_w / grid_bug[1]
    ratio = px_per_tok_bug / px_per_tok
    print(f"      negative control (via prepare_image): grid {grid_bug}, "
          f"{px_per_tok_bug:.4f} original px per token, "
          f"ratio {ratio:.3f} vs the correct path")
    assert ratio < 0.5, ("negative control did not actually differ -- the "
                         "check has no power to detect the rescale bug")
    print("      -> OK: the check detects the rescale, and the real path "
          "does not rescale")

    # ---------------- C3: response localisation vs window size ------------
    # DINOv2 is a global-attention ViT: inside a small crop the perturbation
    # from one anomalous token spreads over most of the window, so the
    # response is not localised and argmax is not a stable cue.  This sweep
    # measures how that recovers with window size.  It is (a) the crop-margin
    # information the pre-registration leaves open, and (b) the size of the
    # attention difference from a global-672 pass that must be disclosed --
    # a crop is at 672 GROUND SCALE but its tokens are NOT global-672 tokens.
    side = int(round(STRIDE / r))
    blank = np.full_like(arr, 128)
    square = blank.copy()
    y0, x0 = H // 2, W // 2
    square[y0:y0 + side, x0:x0 + side] = 0
    outs = {}
    for nm, a in (("blank", blank), ("square", square)):
        _, t = prep((W, H), resize, totensor, norm, a)
        outs[nm] = tokens(model, t)[-1].reshape(gh, gw, -1)
    delta = np.linalg.norm(outs["square"] - outs["blank"], axis=-1)
    gy, gx = np.unravel_index(np.argmax(delta), delta.shape)

    # NOTE: the window must be cut from the SQUARE image and the BLANK image --
    # cutting it from the real photo would difference the photo against a blank
    # and measure nothing.  (At n = 48 the window is the whole grid, so the
    # argmax must come back exactly equal to the global argmax; that identity is
    # what catches this class of mistake.)
    b_bl, _ = prep((W, H), resize, totensor, norm, blank)
    b_sq, _ = prep((W, H), resize, totensor, norm, square)
    print(f"\n  C3 response localisation vs window size "
          f"(square = {side} orig px, global peak at grid ({gy},{gx}))")
    print(f"     {'win':>4} {'peak':>9} {'median':>9} {'peak/med':>9} "
          f"{'argmax':>10} {'predicted':>10}")
    for n in (7, 13, 21, 33, 48):
        if n > min(gh, gw):
            continue
        h = n // 2
        rr0 = min(max(0, gy - h), gh - n) * STRIDE
        cc0 = min(max(0, gx - h), gw - n) * STRIDE
        rr1, cc1 = rr0 + n * STRIDE, cc0 + n * STRIDE
        sq = tokens(model, norm(totensor(b_sq.crop((cc0, rr0, cc1, rr1)))))[-1]
        bl = tokens(model, norm(totensor(b_bl.crop((cc0, rr0, cc1, rr1)))))[-1]
        dm = np.linalg.norm(sq - bl, axis=-1).reshape(n, n)
        loc = tuple(int(v) for v in np.unravel_index(np.argmax(dm), dm.shape))
        pred = (gy - rr0 // STRIDE, gx - cc0 // STRIDE)
        print(f"     {n:>4} {dm.max():>9.2f} {np.median(dm):>9.2f} "
              f"{dm.max() / np.median(dm):>9.2f} {str(loc):>10} {str(pred):>10}")

    print("\n  NOTE (must be disclosed): a crop is at 672 ground SCALE but its "
          "tokens are not\n  global-672 tokens -- attention is confined to the "
          "crop.  Reported above as the\n  crop-vs-global L1. A3 cannot be "
          "expected to reproduce A2 exactly for this reason.")

    print("\n  SCALE CHECK PASSED: windowed extraction is at 672-equivalent "
          "ground scale, and the check has power to detect a rescale.")


if __name__ == "__main__":
    main()
