## Correspondence Learning via Linearly-invariant Embedding (PyTorch)

This repository is a PyTorch implementation of [Correspondence Learning via Linearly-invariant Embedding](https://arxiv.org/abs/2010.13136).

This is *not* the code used to produce the paper results, which can be found [Here](https://github.com/riccardomarin/Diff-FMaps).
This implementation has been made to make handier the use of the method (and also to replicate it).

### Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
Installing pytorch may require different procedure, depending by your computer settings.

### Data & Pretrained models
You can download data and the pretrained models using the scripts:
```
python .\data\download_data.py
python .\models\pretrained\download_pretrained.py
```

### Training

To train the basis and descriptors models, run these commands:

```train
python .\code\train_basis.py
python .\code\train_desc.py
```

### Evaluation

To evaluate the model on FAUST w\noise, run:

```eval
python .\code\test_faust.py
```

### Results

These are the results of the two implementations:

| Model name         | Ours            | Ours+Opt       |
| ------------------ |---------------- | -------------- |
| TF 1.5             |     6.0e-2      |      2.9e-2    |
| PyTorch            |     6.8e-2      |      3.1e-2    |

The small discrepancies have several reasons:
1) basis and descriptors networks are trained 400 epochs in PyTorch implementation; several thousands in TF 1.5
2) while the two implementations are similar, there are some differences in the training process and hyperparameters due to library differences.
3) the training requires pseudo-inverse computation; these can produce different results depending by the library

### License
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

If you use this code, please cite our paper.

```
@article{marin2020correspondence,
  title={Correspondence learning via linearly-invariant embedding},
  author={Marin, Riccardo and Rakotosaona, Marie-Julie and Melzi, Simone and Ovsjanikov, Maks},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/). 
For any commercial uses or derivatives, please contact us.
