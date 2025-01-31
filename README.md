# ProtoSnap

**[project page]() | [paper]()**

[Rachel Mikulinsky*](https://www.linkedin.com/in/rachel-mikulinsky-3a099411b/)<sup>1</sup>, 
[Morris Alper*](https://morrisalp.github.io/)<sup>1</sup>, 
[Shai Gordin](https://cris.ariel.ac.il/en/persons/shai-gordin-2)<sup>2</sup>,
[Enrique Jimenez](https://lmu-munich.academia.edu/EnriqueJiménez)<sup>3</sup>,
[Yoram Cohen](https://english.tau.ac.il/profile/ycohen1)<sup>1</sup>,
[Hadar Averbuch-Elor](https://www.hadarelor.com/)<sup>1,4</sup>

*<sup>1</sup>[Tel Aviv University](https://english.tau.ac.il/), <sup>2</sup>[Ariel University](https://cris.ariel.ac.il/en/), <sup>3</sup>[LMU](https://www.lmu.de/en/), <sup>4</sup>[Cornell University](https://www.cornell.edu/)
*Equal Contribution

This is the official implementation of ProtoSnap, a method for aligning a cuneiform prototype and a corresponding sign image. ICLR 2025

![](repo_images/teaser.png?raw=true)

Given a target image of a cuneiform sign, and a correspoiding prototype with annotated skeleton, we align the skeletong with the target image.
To this aim, we use diffusion features, extracted from a fine-tuned stable diffusion model.
<br>
We used this method to train ControlNet, to generate new a diverse cuneiform signs, based only on a prototype. Weights for the ControlNet are available here.

## Installation

```bash
pip install -r requirements.txt
```
To download the weights:
```bash
gdown 'https://drive.google.com/uc?export=download&id=1x2RlD4jk3O7QFZ6z4ApkSe4RWNnJq_K_'
unzip weights.zip -d weights
rm weights.zip
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

### Generate images with ControlNet

To generate images using our fine-tunes ControlNet:
```bash
python gen_images_with_cn.py <sign_name> --num_of_samples <num_of_samples>
```
The script generats controls, by using available skeletons, and applying small agumentations on each stroke, to create diversity.
Then each control is used to generate an image, using ControlNet.

Arguments:
* ```sign_name``` The name of the sign to generate (such as A, AN, MA, etc.)
* ```--num_of_samples``` Number of samples to generate. 50 by default
* ```--output_path``` The results will be saved under ```<output_path>/<sign_name>/images```. The controls used for generation will be saved under ```<output_path>/<sign_name>/controls```]

## Acknowledgments
* The method and the test set were devolped using the [cunieform OCR dataset](https://github.com/ElectronicBabylonianLiterature/cuneiform-ocr-data). The photographs of tablets are from the [British Museum Digital Collections](https://www.britishmuseum.org/collection).
* This implementation uses code form the official repository of [DIFT](https://github.com/Tsingularity/dift)

## Citation
If you find this project useful, you may cite us as follows:
...
