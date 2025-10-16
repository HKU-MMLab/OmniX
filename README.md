<h1 align="center">OmniX: From Unified Panoramic Generation and Perception to Graphics-Ready 3D Scenes</h1>

<div align="center">

[![Project Page](https://img.shields.io/badge/🏠-Project%20Page-green.svg)](https://yukun-huang.github.io/OmniX/)
[![Paper](https://img.shields.io/badge/📑-Paper-red.svg)](https://arxiv.org/abs/xxxx.xxxxx)
[![Model](https://img.shields.io/badge/🤗-Model-yellow.svg)](https://huggingface.co/KevinHuang/OmniX)
[![Data](https://img.shields.io/badge/🤗-Data-green.svg)](https://huggingface.co/datasets/KevinHuang/PanoX)
[![Video](https://img.shields.io/badge/🎬-Video-blue.svg)]()

</div>

<p align="left">
<img src="assets/teaser.png" width="100%">
<br>
We introduce <b>OmniX</b>, a family of flow matching generative models that achieves <b>unified panorama perception, generation, and completion</b>. Using OmniX as a world generator, we can create graphics-ready 3D scenes ready for physically based rendering, relighting, and simualtion.
</p>

<!-- ## 📢 News
- [2025-10-16] Released. -->

## 📝 TODO

- [x] Release inference code for panorama generation and perception.
- [x] Release inference code for graphics-ready 3D scene generation (beta).
- [ ] Release inference code for panorama completion.
- [ ] Release PanoX dataset.
- [ ] Release training code.

## ⚙️ Installation
Please follow the instructions below to get the code and install dependencies.

Clone the repo:
```bash
git clone https://github.com/HKU-MMLab/OmniX.git
cd OmniX
```

Create a conda environment (optional):
```
conda create -n omnix python=3.11
conda activate omnix
```

Install dependencies:
```
pip install -r requirements.txt
```

## 🚀 Inference

### Panorama Generation
```bash
# Generation from Text
python run_pano_generation.py --prompt "Photorealistic modern living room" --output_dir "outputs/generation_from_text"

# Generation from Image and Text
python run_pano_generation.py --image "assets/examples/image.png" --prompt "Photorealistic modern living room" --output_dir "outputs/generation_from_image_and_text"
```

### Panorama Perception
```bash
# Perception (Depth, Normal, Albedo, Roughness, Metallic, Semantic) from Panorama
python run_pano_perception.py --panorama "assets/examples/panorama.png" --output_dir "outputs/perception_from_panorama"
```

### Panorama Generation and Perception
```bash
# Generation and Perception from Text
python run_pano_all.py --prompt "Photorealistic modern living room" --output_dir "outputs/generation_and_perception_from_text"

# Generation and Perception from Image and Text
python run_pano_all.py --image "assets/examples/image.png" --prompt "Photorealistic modern living room" --output_dir "outputs/generation_and_perception_from_image_and_text"
```

### Graphics-Ready Scene Generation (Beta)
```bash
# Generation from Text
python run_scene_generation.py --prompt "Photorealistic modern living room" --output_dir "outputs/construction_from_text"

# Generation from Image and Text
python run_scene_generation.py --image "assets/examples/image.png" --prompt "Photorealistic modern living room" --output_dir "outputs/construction_from_image_and_text"

# Generation from Panorama
python run_scene_generation.py --panorama "assets/examples/panorama.png" --output_dir "outputs/construction_from_panorama"

# Generation from Panorama (Fast)
python run_scene_generation.py --panorama "assets/examples/panorama.png" --output_dir "outputs/construction_from_image_and_text" --sr 4 --rgb_as_albedo --disable_normal --use_default_pbr --fill_invalid_depth
```

## 👏 Acknowledgement
This repository is based on many amazing research works and open-source projects: [PanFusion](https://github.com/chengzhag/PanFusion), [WorldGen](https://github.com/ZiYang-xie/WorldGen), [diffusers](https://github.com/huggingface/diffusers), [equilib](https://github.com/haruishi43/equilib), etc. Thanks all the authors for their selfless contributions to the community!

## 😉 Citation
If you find this repository helpful for your work, please consider citing it as follows:
```bib
@article{huang2025omnix,
    title={OmniX: From Unified Panoramic Generation and Perception to Graphics-Ready 3D Scenes},
    author={},
    journal={arXiv preprint arXiv:xxxx.xxxxx},
    year={2025}
}
```
