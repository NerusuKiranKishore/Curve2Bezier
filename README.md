# Curve2Bezier

> **Adaptive Vectorization via Piecewise Cubic Bézier Curves:
> A Topology-Aware Curve Approximation Framework**
>
> Nerusu Kiran Kishore, Prayaga Mohan Sashank, Piyush Joshi
> Indian Institute of Information Technology, Sri City
> *Computers & Graphics (Elsevier)* — under review

---

## Overview

`curve2bezier` is the official source code repository for the above paper.
It implements a topology-aware raster vectorisation pipeline that converts
bitmap images of planar strokes into compact piecewise **cubic Bézier curves**,
outperforming the quadratic AEFS baseline by **28.9% in segment count** on
the standard Butterfly benchmark while satisfying the prescribed tolerance.

---

## Key Contributions (from the paper)

- **Compact cubic segmentation** — binary-search greedy extension combined
  with exhaustive segment merging produces fewer segments than standard
  greedy subdivision alone. This combination has not previously been applied
  to cubic-curve raster vectorisation.

- **Topology-aware pipeline** — a two-gate closure test automatically
  classifies input strokes as open or closed directly from raster input,
  enabling wrap-around smoothing and G¹ seam enforcement for closed curves.

- **G¹ continuity at seam** — an arm-averaging pass aligns tangent
  directions at the closed-loop junction, eliminating visible kinks.

- **Symmetry acceleration** — optional axis-symmetry pre-pass halves
  fitting cost on symmetric shapes with guaranteed exact output symmetry.

- **100% pass rate** across all five test cases spanning diverse
  curve geometries.

---

## Benchmark Results

### Butterfly Curve (T1) — 577 pts, δ = 0.5 px

| Method | Segments | Ctrl Pts | Max Error | Within Tolerance |
|--------|----------|----------|-----------|-----------------|
| Smith (1982) | 116 | 349 | 0.6052 px | ✗ |
| Mao–Zhao (2003) | 43 | 131 | 0.1086 px | ✓ |
| Sarfraz (2008) | 18 | 57 | 8.0153 px | ✗ |
| Ueda (2020) | 38 | 115 | 3.1449 px | ✗ |
| Grove (2011) | 36 | 109 | 1.4934 px | ✗ |
| Dung (2017) | 34 | 103 | 1.5246 px | ✗ |
| AEFS — quadratic (2023) | 38 | 77 | 0.2091 px | ✓ |
| **Ours — cubic (2024)** | **27** | **82** | **0.4941 px** | **✓** |

**28.9% fewer segments than AEFS** at comparable fitting error,
both within the δ = 0.5 px tolerance.

### All Five Test Cases

| Case | Description | δ (px) | Segments | Max Error | Pass |
|------|-------------|--------|----------|-----------|------|
| T1 | Butterfly benchmark | 0.5 | 27 | 0.4941 px | ✓ |
| T2 | Inflection curve | 1.0 | 2 | 0.3120 px | ✓ |
| T3 | Open stroke — simple | 4.0 | 7 | 3.92 px | ✓ |
| T4 | Open stroke — complex | 4.0 | 23 | 3.98 px | ✓ |
| T5 | Closed letterform "G" | 4.0 | 24 | 3.97 px | ✓ |

---

## Pipeline

```
Stage 1 → Multi-Hypothesis Binarisation
          (background subtraction + adaptive Gaussian + Otsu)

Stage 2 → Topology-Aware Skeletonisation
          (Zhang–Suen thinning → two-gate open/closed classification)

Stage 3 → Smoothing
          (wrap-around Gaussian for closed curves,
           endpoint-clamped Gaussian for open curves)

Stage 4 → Compact Cubic Fitting
          (corner detection → binary-search greedy extension
           → exhaustive merging)

Stage 5 → Post-Processing & SVG Export
          (G¹ seam enforcement for closed curves, SVG path with Z command)
```

---

## Repository Structure

```
Curve2Bezier/
├── Curve2Bezier_v9.py       # Full pipeline source code
├── input/                   # Test input images
│   ├── G.png                # Closed letterform (T5)
│   ├── S.jpeg               # S-curve test
│   ├── circle.jpeg          # Closed circle test
│   ├── img1.jpeg            # Freehand open stroke (T3/T4)
│   ├── img12.jpeg
│   ├── img15.jpeg
│   └── s_curve.png          # S-curve open stroke
└── output/                  # Result figures produced by the pipeline
    ├── bezier_butterfly_comparison.png   # T1 benchmark figure
    ├── G.png                             # T5 closed letterform result
    ├── S.png                             # S-curve result
    ├── circle.png                        # Circle result
    ├── img1_output.png                   # Freehand stroke result
    ├── s_curve.png                       # S-curve result
    ├── amoeba.png
    └── wave.png
```

---

## Installation

```bash
pip install opencv-python numpy matplotlib pillow scipy scikit-image
```

---

## Usage

```bash
# Single image — auto-detect topology
python Curve2Bezier_v9.py image.png

# Force closed-curve pipeline
python Curve2Bezier_v9.py image.png --closed

# Force open-curve pipeline
python Curve2Bezier_v9.py image.png --open

# Custom tolerance (default: 8.0 px)
python Curve2Bezier_v9.py image.png --tol 2.0

# Export SVG
python Curve2Bezier_v9.py image.png --svg output.svg

# Reproduce paper benchmark (Butterfly vs AEFS)
python Curve2Bezier_v9.py --aefs-compare

# Batch mode — process entire folder
python Curve2Bezier_v9.py --batch-dir ./input

# Disable symmetry detection
python Curve2Bezier_v9.py image.png --no-symmetry

# Custom corner angle threshold
python Curve2Bezier_v9.py image.png --corner-angle 35
```

---

## Output

Each run produces:
- `bezier_output.png` — fitted curve overlaid on original image with
  anchor/handle annotations and a per-point error heatmap
- `bezier_output.svg` — clean SVG path export with Z command for
  closed curves

---

## Reproducibility

To reproduce the Butterfly benchmark figure from the paper:

```bash
python Curve2Bezier_v9.py --aefs-compare
```

This generates `bezier_butterfly_comparison.png` with the
metric-by-metric comparison against AEFS.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{kishore2024curve2bezier,
  title   = {Adaptive Vectorization via Piecewise Cubic B{\'{e}}zier Curves:
             A Topology-Aware Curve Approximation Framework},
  author  = {Nerusu Kiran Kishore and Prayaga Mohan Sashank and Piyush Joshi},
  journal = {Computers \& Graphics},
  year    = {2024},
  note    = {Under review},
  url     = {https://github.com/NerusuKiranKishore/Curve2Bezier}
}
```

---

## License

This repository is released for academic and research purposes only.

© 2024 Nerusu Kiran Kishore, Prayaga Mohan Sashank, Piyush Joshi
Indian Institute of Information Technology, Sri City, India.
