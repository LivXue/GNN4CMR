## Dual Adversarial Graph Neural Networks for Multi-label Cross-modal Retrieval

This repository contains the author's implementation in PyTorch for the AAAI-21 paper "Dual Adversarial Graph Neural Networks for Multi-label Cross-modal Retrieval".


## Dependencies

- Python (>=3.8)

- PyTorch (>=1.7.1)

- Scipy (>=1.5.2)

## Datasets
You can download the features of the datasets from:
 - MIRFlickr, 
 - NUS-WIDE(top-21 concepts),
 - MS-COCO
 
## Implementation

Here we provide the implementation of our proposed models, along with datasets. The repository is organised as follows:

 - `data/` contains the necessary dataset files for NUS-WIDE, MIRFlickr and MS-COCO;
 - `models.py` contains the implementation of the `P-GNN(DAGNN)` and `I-GNN`;
 
 Finally, `main.py` puts all of the above together and can be used to execute a full training run on MIRFlcikr or NUS-WIDE or MS-COCO.

## Process
 - Place the datasets in `data/`
 - Set the experiment parameters in `main.py`.
 - Train a model:
 ```bash
 python main.py
```
 - Modify the parameter `EVAL = True` in `main.py` for evaluation:
  ```bash
 python main.py
```

## Citation
If you find our work or the code useful, please consider cite our paper using:
```bash
@article{Qian_Xue_Zhang_Fang_Xu_2021, 
  title={Dual Adversarial Graph Neural Networks for Multi-label Cross-modal Retrieval}, 
  volume={35}, 
  number={3}, 
  journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
  author={Qian, Shengsheng and Xue, Dizhan and Zhang, Huaiwen and Fang, Quan and Xu, Changsheng}, 
  year={2021}, 
  pages={2440-2448} 
}
```