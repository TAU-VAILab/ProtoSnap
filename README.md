# ProtoSnap

**[project page]() | [paper]()**

[Rachel Mikulinsky](https://www.linkedin.com/in/rachel-mikulinsky-3a099411b/)¹, 
[Morris Alper](https://morrisalp.github.io/)¹, 
[Shai Gordin],
[Enrique Jimenez],
[Yoram Cohen],
[Hadar Averbuch-Elor](https://www.elor.sites.tau.ac.il/)¹

*¹[Tel Aviv University](https://english.tau.ac.il/),

This is the official implementation of ProtoSnap, a method for aligning a cuneiform prototype and a corresponding sign image.

![](repo_images/teaser.png?raw=true)

Given a target image of a cuneiform sign, and a correspoiding prototype with annotated skeleton, we align the skeletong with the target image.
To this aim, we use diffusion features, extracted from a fine-tuned stable diffusion model.
<br>
We used this method to train ControlNet, to generate new a diverse cuneiform signs, based only on a prototype. Weights for the ControlNet are available here.

## Installation

```bash
pip install -r requirements.txt
```

## Run

### Run on a single image

To run on a single sign image:
```bash
python main.py <prompt> --target_image_path <path_to_image_dir>
```

Arguments:
* ```prompt``` The name of the sign (such as A, AN, MA, etc.), used as prompt to the SD model
* ```--target_image_path``` The directory path where the targe image is located. The image name should be ```<prompt>.png```. By defualt - ```target_images```
* ```--font_dir``` The directory with available prototypes. By default - ```prototypes/Santakku```, corresponding to Old Babylonian era. The font Assurbanipal for the Neo-Assyrian era avaliable as well in this repo
* ```--con_dir``` The directory with annotated skeletons. By default - ```skeletons/Santakku```, skeletons for Assurbanipal font available as well.
* ```--output_folder``` None by default. If not None, the results will be saved under ```output/<output_folder>```, else directly under ```output```

### Run on the test set

To run the system on a list of images:

```bash
python run_test.py --samples_df_path <samples_csv>
```
Arguments:
* ```--samples_df_path``` A metadata csv for the requested samples. By default ```test_set/metadata.csv```
* ```--font_dir```, ```--con_dir``` and ```--output_folder``` same as for a single image

## Citation
If you find this project useful, you may cite us as follows:
...
