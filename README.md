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

<!-- ## NNDL Execution

Please make sure to navigate into the _nndl_ folder prior to running the following scripts.

**(1) RGB / RF**

Run the following command to train the rf and the rgb models.
```
>> python {rf or rgb}/train.py --train-shuffle --verbose
```

Run the following command to test the rf and the rgb models.
```
>> python {rf or rgb}/test.py --verbose
```

**(2) Fusion Data Generation**

Run the following command to generate the pickle file with the data for the fusion model.
```
>> python data/fusion_gen.py --verbose
```

**(3) Fusion**

Run the following command to train the fusion model.
```
>> python fusion/train.py --shuffle --verbose
```

Run the following command to test the fusion model.
```
>> python fusion/test.py --verbose
```

**(4) Command Line Args**

For more info about the command line arguments, please run the following:
```
>> python {folder}/file.py --help
```

<hr/>

## References

1) Zheng, Tianyue, et al. "MoRe-Fi: Motion-robust and Fine-grained Respiration Monitoring via Deep-Learning UWB Radar." Proceedings of the 19th ACM Conference on Embedded Networked Sensor Systems. 2021.

2) Yu, Zitong, Xiaobai Li, and Guoying Zhao. "Remote photoplethysmograph signal measurement from facial videos using spatio-temporal networks." arXiv preprint arXiv:1905.02419 (2019).

<hr />

## Citation

```
@article{vilesov2022blending,
  title={Blending camera and 77 GHz radar sensing for equitable, robust plethysmography},
  author={Vilesov, Alexander and Chari, Pradyumna and Armouti, Adnan and Harish, Anirudh Bindiganavale and Kulkarni, Kimaya and Deoghare, Ananya and Jalilian, Laleh and Kadambi, Achuta},
  journal={ACM Transactions on Graphics (TOG)},
  volume={41},
  number={4},
  pages={1--14},
  year={2022},
  publisher={ACM New York, NY, USA}
}
``` -->