# Road-Scene Semantic Segmentation for Self-Driving Cars

[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Space-yellow?logo=huggingface)](https://huggingface.co/spaces/piopanjaitan/Segmentation_Unet_and_F8CN)
[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5c3ee8?logo=opencv)](https://opencv.org/)
[![Gradio](https://img.shields.io/badge/Gradio-Demo-orange)](https://www.gradio.app/)
[![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-green)](https://optuna.org/)

Portfolio project for **semantic segmentation in self-driving / urban road scenes** using **U-Net** as the primary model, with comparison against **FCN-ResNet50**, Optuna-based tuning, video inference, and deployment-oriented Gradio/Hugging Face integration.

> Dense scene understanding for urban driving videos, built with PyTorch, evaluated with segmentation metrics, and packaged for interactive demos.

The project focuses on:
- semantic segmentation on a 12-class road-scene dataset
- end-to-end notebook workflow from dataset validation to video inference
- hybrid loss with **Weighted Cross Entropy + Dice**
- **Optuna** hyperparameter tuning
- quantitative evaluation with `Pixel Accuracy`, `F1`, `Dice`, and `mIoU`
- semantic video rendering and Gradio-based testing
- deployment preparation for **Hugging Face**

### Online Demo

- [Hugging Face Space: Segmentation_Unet_and_F8CN](https://huggingface.co/spaces/piopanjaitan/Segmentation_Unet_and_F8CN)

### Portfolio Snapshot

- task: **multi-class semantic segmentation**
- domain: **self-driving / urban road scene understanding**
- primary model: **U-Net**
- comparison model: **FCN-ResNet50**
- output formats: **metrics, plots, semantic masks, rendered video, Gradio demo**

### Portfolio Highlights

- built a full semantic segmentation pipeline from dataset validation to video demo
- trained and evaluated **U-Net** for 12-class road-scene understanding
- compared U-Net against **FCN-ResNet50**
- added **Optuna** for hyperparameter search
- rendered semantic overlays on video and exposed testing through **Gradio**
- prepared the project for **Hugging Face Spaces**

### Key Results

- **U-Net**
  - Pixel Accuracy: `0.8614`
  - Macro F1: `0.6658`
  - Mean IoU: `0.5550`
- **FCN-ResNet50**
  - Pixel Accuracy: `0.9137`
  - Macro F1: `0.7157`
  - Best validation IoU: `0.8646`

### What This Project Demonstrates

- semantic segmentation modeling in PyTorch
- data cleaning and label validation for dense prediction tasks
- metric-driven model evaluation beyond simple accuracy
- experiment tracking and hyperparameter tuning
- video inference and demo-oriented visualization
- deployment thinking for interactive model demos

### Quick Links

- [Main Notebook](#1-project-summary)
- [Dataset](#2-dataset)
- [U-Net Model](#4-model-u-net)
- [Optuna Tuning](#6-hyperparameter-tuning-with-optuna)
- [Evaluation Results](#7-evaluation-results-for-u-net)
- [Visual Results](#8-visual-results)
- [Video Inference](#9-video-inference-and-rendering)
- [U-Net vs FCN-ResNet50](#10-comparative-study-u-net-vs-fcn-resnet50)
- [Hugging Face Deployment](#11-hugging-face-deployment)

## Table of Contents

1. [Project Summary](#1-project-summary)
2. [Dataset](#2-dataset)
3. [Exploratory Data Analysis](#3-exploratory-data-analysis)
4. [Model: U-Net](#4-model-u-net)
5. [Training Strategy](#5-training-strategy)
6. [Hyperparameter Tuning with Optuna](#6-hyperparameter-tuning-with-optuna)
7. [Evaluation Results for U-Net](#7-evaluation-results-for-u-net)
8. [Visual Results](#8-visual-results)
9. [Video Inference and Rendering](#9-video-inference-and-rendering)
10. [Comparative Study: U-Net vs FCN-ResNet50](#10-comparative-study-u-net-vs-fcn-resnet50)
11. [Hugging Face Deployment](#11-hugging-face-deployment)
12. [Repository Structure](#12-repository-structure)
13. [How to Run](#13-how-to-run)
14. [Strengths and Limitations](#14-strengths-and-limitations)
15. [Future Work](#15-future-work)
16. [Tech Stack](#16-tech-stack)
17. [Author Note](#17-author-note)

---

## 1. Project Summary

This project builds a complete semantic segmentation workflow for self-driving / urban scene understanding.  
Instead of predicting a single class per image, the system predicts a semantic class for **every pixel**, allowing it to distinguish between:
- road
- pavement
- car
- pedestrian
- bicyclist
- building
- sky
- pole
- fence
- sign symbol
- tree
- unlabelled regions

In practice, this means the model can convert a raw street image or video frame into a structured scene map that is much easier to analyze for autonomous driving and urban perception tasks.

The main experimental notebook used in this repository is:

- [`from_serverAI/ Self-Driving Car - Semantic Segmentation.ipynb`](</mnt/pioneer/Pioneer/Learning/Indonesia AI/Computer Vision/Advanced Computer Vision/Proj3/from_serverAI/ Self-Driving Car - Semantic Segmentation.ipynb>)

Additional notebooks:
- [` Self-Driving Car - Semantic Segmentation.ipynb`](</mnt/pioneer/Pioneer/Learning/Indonesia AI/Computer Vision/Advanced Computer Vision/Proj3/ Self-Driving Car - Semantic Segmentation.ipynb>)  
  Simpler / baseline notebook
- [`from_serverAI/ Self-Driving Car - Semantic Segmentation_bc_others.ipynb`](</mnt/pioneer/Pioneer/Learning/Indonesia AI/Computer Vision/Advanced Computer Vision/Proj3/from_serverAI/ Self-Driving Car - Semantic Segmentation_bc_others.ipynb>)  
  Alternative experiment notebook

---

## 2. Dataset

The project uses a local road-scene semantic segmentation dataset with **12 classes**.

| ID | Class |
|---:|---|
| 0 | Sky |
| 1 | Building |
| 2 | Pole |
| 3 | Road |
| 4 | Pavement |
| 5 | Tree |
| 6 | SignSymbol |
| 7 | Fence |
| 8 | Car |
| 9 | Pedestrian |
| 10 | Bicyclist |
| 11 | Unlabelled |

### Dataset Statistics

From the main notebook:
- paired train samples: `367`
- paired test samples: `101`
- train images without labels: `31`
- final train split: `294`
- final validation split: `73`
- final test split: `101`

### Why Dataset Validation Matters

Before training, the notebook validates:
- image-mask pairing
- class index consistency
- train / validation / test split quality

This step is important because semantic segmentation models are highly sensitive to:
- missing labels
- corrupted masks
- inconsistent class indices

---

## 3. Exploratory Data Analysis

The EDA section checks:
- observed class IDs
- mask label distribution
- example image and example segmentation mask
- class legend consistency

The notebook now uses a **consistent palette** across:
- EDA masks
- class legend
- evaluation visuals
- video rendering

This avoids confusion when interpreting the meaning of each semantic class.

---

## 4. Model: U-Net

The primary model in this project is **U-Net**.

### Why U-Net?

U-Net is well suited for semantic segmentation because it combines:
- an **encoder** to capture global context
- a **decoder** to reconstruct spatial details
- **skip connections** to preserve fine-grained information

This makes U-Net especially useful for:
- road and pavement boundaries
- object-level mask structure
- dense urban scenes where spatial precision matters

### Representative Main Configuration

From the `from_serverAI` notebook:
- input resolution: `256 x 512`
- base channels: `32`
- physical batch size: `32`
- effective batch size: `32`
- epochs: `100`
- AMP: enabled

---

## 5. Training Strategy

The project uses:
- **Weighted Cross Entropy**
- **Dice Loss**
- combined as **Cross Entropy + Dice**

### Motivation

`Cross Entropy`:
- learns per-pixel class prediction
- works well for multi-class classification at pixel level

`Dice Loss`:
- improves mask overlap quality
- helps segmentation consistency

`Class Weights`:
- reduce the dominance of large classes such as `Road` and `Sky`
- emphasize smaller / harder classes such as `Pole`, `Pedestrian`, and `Bicyclist`

### Example Class Weights

Representative weights from the notebook:
- `Pole = 1.4586`
- `SignSymbol = 1.3724`
- `Fence = 1.3995`
- `Pedestrian = 1.7796`
- `Bicyclist = 2.6342`

---

## 6. Hyperparameter Tuning with Optuna

The advanced notebook includes a complete **Optuna** pipeline:
- tuning
- summary table
- visualization
- CSV export
- PNG export
- report asset collection

### Tuned Parameters

The notebook tunes:
- optimizer
- learning rate
- weight decay
- momentum where needed

### Best U-Net Optuna Result

Best recorded Optuna result from the notebook:
- best optimizer: `Adam`
- best learning rate: `0.0010370844668954541`
- best weight decay: `0.0048696409415209`
- best Optuna validation loss: `2.367030700047811`

### Optuna Visualizations

Examples stored in the repository:

| Optimization History | Parameter Importance |
|---|---|
| ![Optuna History](docs/assets/images/optuna_optimization_history.png) | ![Optuna Importance](docs/assets/images/optuna_parameter_importance.png) |

| LR vs Validation Loss | Weight Decay vs Validation Loss |
|---|---|
| ![Optuna LR](docs/assets/images/optuna_lr_vs_val_loss.png) | ![Optuna WD](docs/assets/images/optuna_weight_decay_vs_val_loss.png) |

Note:
- the Optuna result is based on short tuning runs
- it is used to guide the main training setup, not replace the full training result

---

## 7. Evaluation Results for U-Net

The project evaluates the model using:
- Pixel Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Macro Dice
- Mean IoU
- Boundary Precision / Recall / F1
- Confusion Matrix

### Best Validation Result

- best validation loss: `1.0252`
- best validation mIoU: `0.5251`

### Test Result

- test loss: `0.9841`
- pixel accuracy: `0.8614`
- macro precision: `0.6609`
- macro recall: `0.7071`
- macro F1: `0.6658`
- macro Dice: `0.6658`
- mean IoU: `0.5550`
- boundary F1: `0.6340`

### Per-Class IoU

- Sky: `0.9399`
- Building: `0.7102`
- Pole: `0.0611`
- Road: `0.9389`
- Pavement: `0.7794`
- Tree: `0.8349`
- SignSymbol: `0.2647`
- Fence: `0.5253`
- Car: `0.5860`
- Pedestrian: `0.3325`
- Bicyclist: `0.5123`
- Unlabelled: `0.1744`

### Interpretation

The model is strongest on:
- road
- sky
- tree
- pavement
- building

The model is weaker on:
- pole
- sign symbol
- pedestrian
- unlabelled

This is expected because:
- large classes dominate the image area
- thin and small objects are harder to preserve during downsampling

---

## 8. Visual Results

### Video Contact Sheet

Rendered output example:

![Video Contact Sheet](docs/assets/images/segmentation_contact_sheet.jpg)

Alternative render style:

![Felim Style Contact Sheet](docs/assets/images/segmentation_contact_sheet_felim_style.jpg)

These images show how the model performs visually on real video frames, especially for:
- road layout
- pavement segmentation
- building and sky regions
- object-level classes such as car and bicyclist

---

## 9. Video Inference and Rendering

The notebook supports video inference with:
- per-frame segmentation
- semantic overlay rendering
- frame export
- notebook playback cell
- Gradio-based interactive testing

### Representative Video Benchmark

From the notebook:
- processed frames: `451`
- source FPS: `29.7391`
- pipeline FPS: around `24-25`
- model-only FPS: around `470+`

### Available Sample Files

- input video: [`nD_6.mp4`](</mnt/pioneer/Pioneer/Learning/Indonesia AI/Computer Vision/Advanced Computer Vision/Proj3/nD_6.mp4>)
- rendered video: [`from_serverAI/nD_6_result.mp4`](</mnt/pioneer/Pioneer/Learning/Indonesia AI/Computer Vision/Advanced Computer Vision/Proj3/from_serverAI/nD_6_result.mp4>)
- Gradio render sample: [`from_serverAI/nD_1_gradio_result.mp4`](</mnt/pioneer/Pioneer/Learning/Indonesia AI/Computer Vision/Advanced Computer Vision/Proj3/from_serverAI/nD_1_gradio_result.mp4>)

### GitHub Supporting Media

For easier browsing on GitHub, supporting media is also collected under `docs/assets`:

#### Images
- [`docs/assets/images/segmentation_contact_sheet.jpg`](docs/assets/images/segmentation_contact_sheet.jpg)
- [`docs/assets/images/segmentation_contact_sheet_felim_style.jpg`](docs/assets/images/segmentation_contact_sheet_felim_style.jpg)
- [`docs/assets/images/optuna_optimization_history.png`](docs/assets/images/optuna_optimization_history.png)
- [`docs/assets/images/optuna_parameter_importance.png`](docs/assets/images/optuna_parameter_importance.png)
- [`docs/assets/images/optuna_lr_vs_val_loss.png`](docs/assets/images/optuna_lr_vs_val_loss.png)
- [`docs/assets/images/optuna_weight_decay_vs_val_loss.png`](docs/assets/images/optuna_weight_decay_vs_val_loss.png)

#### Videos
- [`docs/assets/videos/input_demo.mp4`](docs/assets/videos/input_demo.mp4)
- [`docs/assets/videos/unet_render_output.mp4`](docs/assets/videos/unet_render_output.mp4)
- [`docs/assets/videos/gradio_render_output.mp4`](docs/assets/videos/gradio_render_output.mp4)

These files can be used directly as presentation support material, report attachments, or GitHub repository assets.

### Current Practical Note

The notebook contains explicit logic for browser playback, but video playback may still depend on:
- `ffmpeg` availability
- H.264 encoder availability
- notebook runtime path consistency

So the segmentation model can be correct even when browser playback still needs environment-side adjustment.

---

## 10. Comparative Study: U-Net vs FCN-ResNet50

This repository also includes a comparison-ready segmentation baseline based on **`torchvision.models.segmentation.fcn_resnet50`**.

### FCN Baseline Notebooks

Local notebooks:
- [`from_serverAI/training_self_driving_car_v2.ipynb`](</mnt/pioneer/Pioneer/Learning/Indonesia AI/Computer Vision/Advanced Computer Vision/Proj3/from_serverAI/training_self_driving_car_v2.ipynb>)
- [`from_serverAI/self_driving_car_v2_optuna.ipynb`](</mnt/pioneer/Pioneer/Learning/Indonesia AI/Computer Vision/Advanced Computer Vision/Proj3/from_serverAI/self_driving_car_v2_optuna.ipynb>)

Original Colab references:
- main training notebook: https://colab.research.google.com/drive/1RcwV9-whDs7nKAeGtI86mui35Cny_4Ok#scrollTo=KUYFcY5TiZ2D
- Optuna notebook: https://colab.research.google.com/drive/1pYaMrY7_g6IxZGv-70o1WH7mlDb0OKc0#scrollTo=5NsONGeZlYyG

### Architectural Comparison

| Aspect | U-Net | FCN-ResNet50 |
|---|---|---|
| Core idea | encoder-decoder with dense skip connections | fully convolutional segmentation with coarse-to-fine fusion |
| Spatial detail recovery | strong | moderate |
| Boundary quality | usually better | usually coarser |
| Small-object preservation | relatively better | often weaker |
| Interpretability for segmentation tasks | high | high |
| Historical importance | widely used baseline and practical model | classic milestone segmentation architecture |

### Experimental Comparison

The U-Net metrics in this repository are documented in the main `from_serverAI` notebook.  
The FCN baseline metrics are documented in the local notebooks listed above.

### FCN-ResNet50 Training Summary

From [`from_serverAI/training_self_driving_car_v2.ipynb`](</mnt/pioneer/Pioneer/Learning/Indonesia AI/Computer Vision/Advanced Computer Vision/Proj3/from_serverAI/training_self_driving_car_v2.ipynb>):
- architecture: `torchvision.models.segmentation.fcn_resnet50`
- classifier head adapted to `12` classes
- loss: `CrossEntropyLoss`
- training stopped early at epoch `12`
- best validation loss observed: `0.2805`
- best validation Dice observed: `0.6680`
- best validation IoU observed: `0.8646`

### FCN-ResNet50 Evaluation Summary

From the same notebook:
- pixel accuracy: `0.9137`
- macro precision: `0.7214`
- macro recall: `0.7251`
- macro F1: `0.7157`

Per-class precision / recall / F1 are also available in the notebook, for example:
- Road F1: `0.9782`
- Sky F1: `0.9590`
- Building F1: `0.9182`
- Pole F1: `0.0002`
- Pedestrian F1: `0.5858`
- Bicyclist F1: `0.7961`

### FCN-ResNet50 Optuna Summary

From [`from_serverAI/self_driving_car_v2_optuna.ipynb`](</mnt/pioneer/Pioneer/Learning/Indonesia AI/Computer Vision/Advanced Computer Vision/Proj3/from_serverAI/self_driving_car_v2_optuna.ipynb>):
- best optimizer: `Adam`
- best learning rate: `8.17949947521167e-05`
- best weight decay: `2.9204338471814107e-05`
- best momentum: `0.9111852894722379`
- best beta1: `0.9125544474586837`
- best beta2: `0.992629301836817`
- best Optuna objective value: `0.6484393338744456`

### U-Net vs FCN-ResNet50 Reference Table

| Metric | U-Net | FCN-ResNet50 |
|---|---:|---:|
| Best Validation Loss | `1.0252` | `0.2805` |
| Best Validation mIoU / IoU | `0.5251` | `0.8646` |
| Best Validation Dice | not explicitly reported as best standalone final summary | `0.6680` |
| Test Loss | `0.9841` | not explicitly summarized in final table |
| Pixel Accuracy | `0.8614` | `0.9137` |
| Macro Precision | `0.6609` | `0.7214` |
| Macro Recall | `0.7071` | `0.7251` |
| Macro F1 | `0.6658` | `0.7157` |
| Macro Dice | `0.6658` | not explicitly reported in final summary table |
| Mean IoU | `0.5550` | not explicitly reported in final summary table |

### Optuna Comparison

| Metric | U-Net | FCN-ResNet50 |
|---|---:|---:|
| Best Optimizer | `Adam` | `Adam` |
| Learning Rate | `0.0010370844668954541` | `8.17949947521167e-05` |
| Weight Decay | `0.0048696409415209` | `2.9204338471814107e-05` |
| Best Optuna Objective | `2.367030700047811` | `0.6484393338744456` |

### Academic Positioning

This allows the project to be framed as:
- an implementation study of **U-Net for road-scene semantic segmentation**
- a methodological comparison against **FCN-ResNet50**
- an evaluation study across image metrics, video rendering, and deployment readiness

---

## 11. Hugging Face Deployment

This project is suitable for deployment to **Hugging Face Spaces**, especially with **Gradio**.

### Live Demo

- [Open the Hugging Face Space](https://huggingface.co/spaces/piopanjaitan/Segmentation_Unet_and_F8CN)

### Recommended Deployment Target

- **Hugging Face Spaces**
- framework: **Gradio**
- purpose:
  - upload a video
  - run semantic segmentation using the trained model
  - return rendered semantic video

### Suggested Deployment Assets

To deploy this project to Hugging Face, prepare:
- `app.py` for Gradio app entry point
- model checkpoint such as `best_self_driving_model.pth`
- inference-only utilities from the notebook
- `requirements.txt`
- optional example video

### Suggested Hugging Face Space Features

- upload video
- select render confidence
- select overlay alpha
- return semantic segmentation video
- optionally preview representative frame outputs

### Deployment Note

The current notebook already contains Gradio-based logic.  
The main remaining step for Hugging Face deployment is to:
- move notebook inference code into a clean `app.py`
- ensure browser-compatible output video generation inside the Space runtime

---

## 12. Repository Structure

```text
Proj3/
├── README.md
├──  Self-Driving Car - Semantic Segmentation.ipynb
├── best_self_driving_model.pth
├── nD_6.mp4
├── nD_6_result.mp4
├── cityt /
├── from_serverAI/
│   ├──  Self-Driving Car - Semantic Segmentation.ipynb
│   ├──  Self-Driving Car - Semantic Segmentation_bc_others.ipynb
│   ├── best_self_driving_model.pth
│   ├── nD_6_result.mp4
│   ├── nD_1_gradio_result.mp4
│   ├── contact_sheet_latest_analysis.jpg
│   ├── contact_sheet_felim_style_result.jpg
│   ├── optuna_exports/
│   ├── gradio_renders/
│   └── rendered_frames_felim_style/
└── versi_login_cityscapes/
    └──  Self-Driving Car - Semantic Segmentation.ipynb
```

---

## 13. How to Run

### Main notebook

Use:
- [`from_serverAI/ Self-Driving Car - Semantic Segmentation.ipynb`](</mnt/pioneer/Pioneer/Learning/Indonesia AI/Computer Vision/Advanced Computer Vision/Proj3/from_serverAI/ Self-Driving Car - Semantic Segmentation.ipynb>)

### Suggested execution order

1. environment setup
2. config
3. dataset validation
4. EDA
5. model definition
6. training setup
7. Optuna tuning
8. training loop
9. evaluation
10. video rendering
11. playback
12. Gradio testing

---

## 14. Strengths and Limitations

### Strengths

- complete notebook-based segmentation pipeline
- dataset validation included
- hybrid loss for class imbalance
- Optuna support
- detailed evaluation metrics
- video inference support
- Gradio-based testing
- ready foundation for Hugging Face deployment

### Limitations

- thin classes like `Pole` remain difficult
- video visual quality depends on post-processing tuning
- browser playback still depends on runtime codec support
- FCN8 metrics are still referenced externally and not yet fully mirrored into the repo

---

## 15. Future Work

- import and document full FCN8 results locally
- add direct side-by-side U-Net vs FCN8 tables
- finalize Hugging Face Space deployment
- improve browser-safe video export
- improve thin-object segmentation quality

---

## 16. Tech Stack

- Python
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Matplotlib
- Optuna
- Gradio
- Jupyter Notebook
- Hugging Face Spaces

---

## 17. Author Note

This repository is structured as an academic / experimental project for semantic segmentation, with emphasis on:
- reproducible notebook experiments
- visual and quantitative evaluation
- comparison readiness with FCN8
- deployment preparation for real interactive demos
# Semantic_Segmentation_for_Self-Driving_Cars_with_UNet_and_F8CN
# Semantic_Segmentation_for_Self-Driving_Cars_with_UNet_and_F8CN
