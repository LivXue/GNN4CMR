## Dual Adversarial Label-aware Graph Neural Networks

This repository contains the author's implementation in PyTorch for the paper "Dual Adversarial Label-aware Graph Neural Networks for Cross-modal Retrieval".


## Overview

> The structures of DALGNN  

<div align="center">
    <img src="docs/images/framework.jpg" width="100%" height ="100%" alt="framework.jpg" />
</div>
<p align = 'center'>
    <small>The overall structure of DALGNN.</small>
</p>


## Dependencies

- Python (>=3.7)

- PyTorch (>=1.2.0)

- Scipy (>=1.3.2)

## Datasets
You can download the features of the datasets from:
 - MIRFlickr, 
 - NUS-WIDE(top-21 concepts)
 
## Implementation

Here we provide the implementation of DALGNN, along with datasets. The repository is organised as follows:

 - `data/` contains the necessary dataset files for NUS-WIDE and MIRFlickr;
 - `models.py` contains the implementation of the `DALGNN`;
 
 Finally, `main.py` puts all of the above together and can be used to execute a full training run on MIRFlcikr or NUS-WIDE.

## Process
 - Place the datasets in `data/`
 - Change the `dataset` in `main.py` to `mirflickr` or `NUS-WIDE-TC21`.
 - Train a model:
 ```bash
 python main.py
```
 - Modify the parameter `EVAL = True` in `main.py` for evaluation:
  ```bash
 python main.py
```