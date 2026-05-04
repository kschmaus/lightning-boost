# Installation

## Requirements

- Python >= 3.11
- LightGBM >= 4.0
- NumPy >= 1.24
- SciPy >= 1.10
- scikit-learn >= 1.3

## Install with pip

```bash
pip install ngboost-lightning
```

## Install with uv

```bash
uv add ngboost-lightning
```

## Optional Dependencies

**matplotlib** is required for the plotting functions in `ngboost_lightning.evaluation` (`plot_pit_histogram`, `plot_calibration_curve`). It is not installed automatically.

```bash
pip install matplotlib
```

## Development Install

To install from source with development dependencies:

```bash
git clone https://github.com/kschmaus/ngboost-lightning.git
cd ngboost-lightning
uv sync --group dev
```

To also install documentation dependencies:

```bash
uv sync --group dev --group docs
```
