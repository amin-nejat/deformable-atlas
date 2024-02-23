# Learning Probabilistic Piecewise Rigid Atlases of Model Organisms
![Learning Probabilistic Piecewise Rigid Atlases of Model Organisms](https://github.com/amin-nejat/deformable-atlas/assets/5959554/960a5935-6df9-4423-b71a-fd2bf97ef412)

Atlases are crucial to imaging statistics as they enable the standardization of inter-subject and inter-population analyses. While existing atlas estimation methods based on fluid/elastic/diffusion registration yield high-quality results for the human brain, these deformation models do not extend to a variety of other challenging areas of neuroscience such as the anatomy of C. elegans worms and fruit flies. To this end, this work presents a general probabilistic deep network-based framework for atlas estimation and registration which can flexibly incorporate various deformation models and levels of keypoint supervision that can be applied to a wide class of model organisms.

See **[our paper](https://link.springer.com/chapter/10.1007/978-3-031-34048-2_26)** for further details:


```
@inproceedings{nejatbakhsh2023learning,
  title={Learning Probabilistic Piecewise Rigid Atlases of Model Organisms via Generative Deep Networks},
  author={Nejatbakhsh, Amin and Dey, Neel and Venkatachalam, Vivek and Yemini, Eviatar and Paninski, Liam and Varol, Erdem},
  booktitle={International Conference on Information Processing in Medical Imaging},
  pages={332--343},
  year={2023},
  organization={Springer}
}
```
**Note:** This research code remains a work-in-progress to some extent. It could use more documentation and examples. Please use at your own risk and reach out to us (anejatbakhsh@flatironinstitute.org) if you have questions. If you are using this code package, please cite our paper.

## A short and preliminary guide

### Installation Instructions

1. Download and install [**anaconda**](https://docs.anaconda.com/anaconda/install/index.html)
2. Create a **virtual environment** using anaconda and activate it

```
conda create -n datlas python=3.8
conda activate datlas
```

3. Install [**Pytorch**](https://pytorch.org/) package

4. Install other requirements (pyro, matplotlib, scipy, sklearn, cv2, dipy, ray)

5. Run either using the demo file or the run script via the following commands

```
python run.py -c configs/test.yaml -o ../results/
```

Since the code is preliminary, you will be able to use `git pull` to get updates as we release them.
