## ConMatch - Official Pytorch Implementation
<p align="middle"><img width="70%" src="figs/overall.png" /></p>

> **ConMatch: Semi-Supervised Learning with Confidence-Guided Consistency Regularization**<br>
> Jiwon Kim, Youngjo Min, Daehwan Kim, Gyuseong Lee, Junyoung Seo, Kwangrok Ryoo, Seungryong Kim
> In ECCV 2022.<br>

> **Abstract:** *We present a novel semi-supervised learning framework that intelligently leverages the consistency regularization between the model's predictions from two strongly-augmented views of an image, weighted by a confidence of pseudo-label, dubbed ConMatch. While the latest semi-supervised learning methods use weakly- and strongly-augmented views of an image to define a directional consistency loss, how to define such direction for the consistency regularization between two strongly-augmented views remains unexplored. To account for this, we present novel confidence measures for pseudo-labels from strongly-augmented views by means of weakly-augmented view as an anchor in non-parametric and parametric approaches. Especially, in parametric approach, we present, for the first time, to learn the confidence of pseudo-label within the networks, which is learned with backbone model in an end-to-end manner. In addition, we also present a stage-wise training to boost the convergence of training. When incorporated in existing semi-supervised learners, ConMatch consistently boosts the performance. We conduct experiments to demonstrate the effectiveness of our ConMatch over the latest methods and provide extensive ablation studies. Source code is available at https://github.com/JiwonCocoder/ConMatch.*

## Installation

### Clone this repository

```bash
git clone https://github.com/JiwonCocoder/ConMatch.git
cd ConMatch/
```

### Install the dependencies

```bash
conda env create -f environment.yml
```
When using 3090 GPUs,
```bash
conda env create -f environment3090.yml
```

### Run the experiments

1. Modify the config file in `config/conmatch/*.yaml`
2. Run `python conmatch.py --c config/conmatch/*.yaml`

## Citation
If you find this work useful for your research, please cite our paper:
```
@inproceedings{kim2022conmatch,
  title={ConMatch: Semi-Supervised Learning with Confidence-Guided Consistency Regularization},
  author={Kim, Jiwon and Min, Youngjo and Kim, Daehwan and Lee, Gyuseong and Seo, Junyoung and Ryoo, Kwangrok and Kim, Seungryong},
  booktitle={European Conference on Computer Vision},
  year={2022}
}
```

## Related Projects

Our model code starts from [TorchSSL](https://github.com/TorchSSL/TorchSSL).
