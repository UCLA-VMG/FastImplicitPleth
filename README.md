# FastImplicitPleth
Repository for fast implicit representations for Plethysmography signals.

This code has been tested on Ubuntu 20.04

# Requirements (Most Relevant)
- [CUDA Toolkit 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive)
- [torch](https://pytorch.org/get-started/locally/)
- [tinycudann](https://github.com/NVlabs/tiny-cuda-nn)

# Code References
- [Siren model](https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb#scrollTo=gupA19Fc2kvw)
- [Deep Physiological Sensing toolbox](https://github.com/ubicomplab/rPPG-Toolbox)



# Implicit Neural Models to Extract Heart Rate from Video

> __Implicit Neural Models to Extract Heart Rate from Video__  
> [Pradyumna Chari](https://pradyumnachari.github.io/), [Anirudh Bindiganavale Harish](https://anirudh0707.github.io/), [Adnan Armouti](https://adnan-armouti.github.io/), [Alexander Vilesov](https://asvilesov.github.io/), [Sanjit Sarda](https://sanit1.github.io/), [Laleh Jalilian](https://www.uclahealth.org/laleh-jalilian/), [Achuta Kadambi](https://www.ee.ucla.edu/achuta-kadambi/)<br/>

For details on the citation format, kindly refer to the [Citation](https://github.com/UCLA-VMG/FastImplicitPleth#citation) section below.

<hr /> 

## Dataset and Pre-prep

The FastImplicitPleth dataset can be downloaded by filling this [Google Form](https://forms.gle/mUg2WozmtUh2MBzRA).

If you choose to collect your own data, use a face cropping software (MTCNN in our case) to crop the face and save each frame as an image within the trial/volunteer's folder to the following pre-processing instructions to obtain a similar dataset to the FastImplicitPleth dataset.

Hierarchy of the FastImplicitPleth dataset - RGB Files
```
|
|--- rgb_files
|        |
|        |--- volunteer id 1 trial 1 (v_1_1)
|        |         |
|        |         |--- frame 0 (rgbd_rgb_0.png)
|        |         |--- frame 1 (rgbd_rgb_1.png)
|        |         |
|        |         |
|        |         |
|        |         |--- last frame (rgbd_rgb_899.png)
|        |         |--- ground truth PPG (rgbd_ppg.npy)
|        | 
|        | 
|        |--- volunteer id 1 trial 2 (v_1_2)
|        | 
|        | 
|        | 
|        |--- volunteer id 2 trial 1 (v_2_1)
|        |
|        |
|        |
|
|
|--- fitzpatrick labels file (fitzpatrick_labels.pkl)
|--- folds pickle file (demo_fold.pkl)
```

<hr/>

## NNDL Execution
Before running the following commands, ensure that the configurations are flags are correctly set to run with your environment set up.

In particular, pay particular attention to `configs/dataset/ch_appearance_{set}.json` -> `checkpoints.dir`, and `configs/dataset/residual_{set}.json` -> `checkpoints.dir`; `appearance_model`.


1. Run `python auto_dataset_appearance.py`
2. Run `python auto_dataset_residual.py`
3. Run `inference.ipynb`



## Citation

```
@inproceedings{chari2024implicit,
  title={Implicit Neural Models to Extract Heart Rate from Video},
  author={Chari, Pradyumna and Harish, Anirudh Bindiganavale and Armouti, Adnan and Vilesov, Alexander and Sarda, Sanjit and Jalilian, Laleh and Kadambi, Achuta},
  booktitle={European conference on computer vision},
  year={2024},
  organization={Springer}
}
```