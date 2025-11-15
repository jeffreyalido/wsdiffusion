# Whitened Score Diffusion Models (WS‑DM)


## Overview

Whitened Score Diffusion (WS‑DM) is a generalization of score‑based diffusion models that learns the *Whitened Score* function instead of the conventional score, thereby eliminating the need to invert large covariance matrices. WS‑DM unifies score‑based diffusion and flow‑matching for *arbitrary* Gaussian forward processes, enabling stable training under anisotropic and structured noise. The framework provides flexible spectral inductive biases and strong Bayesian priors for challenging imaging inverse problems.

**Paper:** *Whitened Score Diffusion Models* — available on [arXiv:2505.10311](https://arxiv.org/abs/2505.10311).

## Usage

### Installation

1. **Clone the repository:**
```bash
   git clone https://github.com/yourusername/wsdiffusion.git
   cd wsdiffusion
```

2. **Install uv:**
   
   If you don't have uv installed, you can install it using one of the following methods:
   
   **On macOS and Linux:**
```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
```
   
   **On Windows:**
```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
   
   **Alternative (using pip):**
```bash
   pip install uv
```

3. **Install dependencies:**
```bash
   uv sync
```
   
   This will create a virtual environment and install all required packages specified in the project.

### Training

The provided code in this repo is very minimal and raw for ease of hacking and experimentation. To train a WS-DM model, run:
```bash
uv run train.py
```

If you use this work in your research, please cite the accompanying paper:

```bibtex
@misc{alido2025whitenedscorediffusionstructured,
      title={Whitened Score Diffusion: A Structured Prior for Imaging Inverse Problems}, 
      author={Jeffrey Alido and Tongyu Li and Yu Sun and Lei Tian},
      year={2025},
      eprint={2505.10311},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2505.10311}, 
}
```

We will update the citation once a final DOI is available.

## License

The project will be released under the **MIT License**.

## Contact

For questions regarding WS‑DM or related research, please open an issue or contact the maintainer:

> Jeffrey Alido — *[jalido@bu.edu](mailto:jalido@bu.edu)*

