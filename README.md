# Rethinking Robustness of Model Attributions

This repository is the codebase for the AAAI 2024 paper [Rethinking Robustness of Model Attributions]

## Overview
The paper shows the two main causes for fragile attributions: first, the existing metrics of robustness (e.g., top-k intersection) *over-penalize* even reasonable local shifts in attribution, thereby making random perturbations to appear as a strong attack, and second, the attribution can be concentrated in a small region even when there are multiple important parts in an image. To rectify this, we propose simple ways to strengthen existing metrics and attribution methods that incorporate *locality* of pixels in robustness metrics and *diversity* of pixel locations in attributions.

## Dependencies
Codebases used in the paper as is or modified accordingly.

* [Interpretation Fragility] (https://github.com/amiratag/InterpretationFragility)
* [RAR] (https://github.com/jfc43/robust-attribution-regularization/)
* [Captum] (https://github.com/pytorch/captum)

## Code documentation.
* Folder \<dataset\> are based on [RAR] code base. We have incorporated the LENS and top-k-div with LENS as part of the code. The metrics are applied after the attacks return the corresponding explanation maps of the unperturbed and perturbed images.

Please use the instructions given in Chen-et-al-README.md for obtaining Natural, Adversarial (PGD) and Attributional (IG-SUM-NORM) models for MNIST and Flower datasets. For the proposed LENS-metric based evaluation and attack construction, check ./\<dataset\>/README.md.

* Folder *scripts* contains scripts to post process stored explanation maps of unperturbed and perturbed images to obtain LENS and top-k-div with LENS values for ImageNet dataset. Please use the instructions given in ./scripts/README.md to know more about individual files. 

## Citation

If the code related to our work is useful for your work, kindly cite this work as given below:

```[bibtex]
@inproceedings{kamath2024rethinkingrobustnessmodelattributions,
  title={Rethinking Robustness of Model Attributions}, 
  author={Sandesh Kamath and Sankalp Mittal and Amit Deshpande and Vineeth N Balasubramanian},
  booktitle={Association for the Advancement of Artificial Intelligence (AAAI)},
  howpublished={arXiv preprint arXiv:2312.10534},
  url={https://ojs.aaai.org/index.php/AAAI/article/view/28047}
}

```
