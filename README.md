# Whitened Score Diffusion Models (WS‑DM)

**Status: Code release coming soon — stay tuned!**

---

## Overview

Whitened Score Diffusion (WS‑DM) is a generalization of score‑based diffusion models that learns the *Whitened Score* function instead of the conventional score, thereby eliminating the need to invert large covariance matrices. WS‑DM unifies score‑based diffusion and flow‑matching for *arbitrary* Gaussian forward processes, enabling stable training under anisotropic and structured noise. The framework provides flexible spectral inductive biases and strong Bayesian priors for challenging imaging inverse problems.

**Paper:** *Whitened Score Diffusion Models* — available on [arXiv:2505.10311](https://arxiv.org/abs/2505.10311).



## Citing WS‑DM

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

---

*This repository accompanies the paper “Whitened Score Diffusion Models” (WS‑DM). Code review is currently in progress; check back for updates and feel free to star the repo to receive notifications.*
