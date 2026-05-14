# Semantic Segmentation for Self-Driving Cars with U-Net and FCN

[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Space-yellow?logo=huggingface)](https://huggingface.co/spaces/piopanjaitan/Segmentation_Unet_and_F8CN)
[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-orange?logo=pytorch)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-Demo-orange)](https://www.gradio.app/)

Semantic segmentation project for urban road-scene understanding using **U-Net** and a comparison model based on **FCN-ResNet50**.  
This project covers training, evaluation, Optuna tuning, video inference, Gradio testing, and Hugging Face deployment.

## Demo

- Hugging Face Space: [Segmentation_Unet_and_F8CN](https://huggingface.co/spaces/piopanjaitan/Segmentation_Unet_and_F8CN)

## Main Files

- Main notebook: [` Self-Driving Car - Semantic Segmentation.ipynb`](<%20Self-Driving%20Car%20-%20Semantic%20Segmentation.ipynb>)
- FCN training notebook: [`training_self_driving_car_v2.ipynb`](training_self_driving_car_v2.ipynb)
- FCN Optuna notebook: [`self_driving_car_v2_optuna.ipynb`](self_driving_car_v2_optuna.ipynb)
- Model checkpoint: [`best_self_driving_model.pth`](best_self_driving_model.pth)

## Classes

The dataset contains 12 classes:

`Sky`, `Building`, `Pole`, `Road`, `Pavement`, `Tree`, `SignSymbol`, `Fence`, `Car`, `Pedestrian`, `Bicyclist`, `Unlabelled`

## Results

### U-Net

- Best validation loss: `1.0252`
- Best validation mIoU: `0.5251`
- Test loss: `0.9841`
- Pixel Accuracy: `0.8614`
- Macro F1: `0.6658`
- Mean IoU: `0.5550`

### FCN-ResNet50

- Best validation loss: `0.2805`
- Best validation Dice: `0.6680`
- Best validation IoU: `0.8646`
- Pixel Accuracy: `0.9137`
- Macro Precision: `0.7214`
- Macro Recall: `0.7251`
- Macro F1: `0.7157`

### Optuna

**U-Net**
- Best optimizer: `Adam`
- Learning rate: `0.0010370844668954541`
- Weight decay: `0.0048696409415209`

**FCN-ResNet50**
- Best optimizer: `Adam`
- Learning rate: `8.17949947521167e-05`
- Weight decay: `2.9204338471814107e-05`

## Visual Results

### U-Net Video Preview

![U-Net Preview](assets/unet_preview.gif)

### FCN-ResNet50 Video Preview

![FCN Preview](assets/fcn_preview.gif)

## Videos

- U-Net render output: [`nD_6_result_unet.mp4`](nD_6_result_unet.mp4)
- U-Net browser-friendly output: [`assets/nD_6_result_unet_web.mp4`](assets/nD_6_result_unet_web.mp4)
- FCN-ResNet50 render output: [`nD_6_result_f8cn.mp4`](nD_6_result_f8cn.mp4)
- Gradio render output: [`nD_1_gradio_result.mp4`](nD_1_gradio_result.mp4)
- U-Net preview GIF: [`assets/unet_preview.gif`](assets/unet_preview.gif)
- FCN preview GIF: [`assets/fcn_preview.gif`](assets/fcn_preview.gif)

## Installation

```bash
git clone https://github.com/piopanjaitan/Semantic_Segmentation_for_Self-Driving_Cars_with_UNet_and_F8CN.git
cd Semantic_Segmentation_for_Self-Driving_Cars_with_UNet_and_F8CN
pip install -r requirements.txt
```

## How to Run

1. Open the main notebook:
   - [` Self-Driving Car - Semantic Segmentation.ipynb`](<%20Self-Driving%20Car%20-%20Semantic%20Segmentation.ipynb>)
2. Run dataset validation, EDA, model, training, evaluation, and video cells in order.
3. Use the Gradio section in the notebook or the Hugging Face Space for demo testing.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── LICENSE
├── best_self_driving_model.pth
├── nD_1_gradio_result.mp4
├── nD_6_result_f8cn.mp4
├── nD_6_result_unet.mp4
├── self_driving_car_v2_optuna.ipynb
├── training_self_driving_car_v2.ipynb
├──  Self-Driving Car - Semantic Segmentation.ipynb
└── assets/
    ├── fcn_preview.gif
    ├── nD_6_result_unet_web.mp4
    └── unet_preview.gif
```

## Notes

- U-Net notebook includes video rendering, playback, and Gradio integration.
- FCN comparison in this repository uses `torchvision.models.segmentation.fcn_resnet50`.
- Browser playback can still depend on codec support in the runtime environment.
