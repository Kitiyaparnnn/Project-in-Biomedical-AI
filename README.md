![falcon](https://github.com/user-attachments/assets/b7b18293-4f48-48c6-86e2-e35a1264ecbf)

# FALCON: FoveA LoCalizatiON in En-face OCT Imaging via Explainable B-scan Classification and Transformer-Based Segmentation
This repository is the official implementation of FALCON: FoveA LoCalizatiON in En-face OCT Imaging via Explainable B-scan Classification and Transformer-Based Segmentation. And also a part of 42-687 Projects in Biomedical AI (Spring 2025) at Carnegie Mellon University.

Members: Arav Jain, Kitiyaporn Takham, Micah Baldonado, Shreyas Sanghvi




## Training

### Fovea Classification
folder: fovea_classification_codes

The VGG16 was fine-tuning with fovea/non-fovea B-scan slices in both healthy and unhealthy eyes.

### Fovea Explainability 
folder: fovea_explainability_codes

The fovea explainability algorithms include:
- Vanilla Seliency
- Occlusion
- GradCAM
- GradCAM++
- ScoreCAM

### En-face Segmentation 
folder: enface_segmentation_codes

The model consists of ViT-Based MAE (encoder) and UNet++ Decoder (decoder).

## Evaluation
### Fovea Classification
folder: fovea_classification_codes

The model was evaluated on a sample volumetric
- potential fovea images (15 slices)
- potential non-fovea images (117 slices)

## Pre-trained Models
You can download pretrained models here:
- [Fovea Classification](https://drive.google.com/file/d/1a1xVnmCWiHugP7fa0k8KmDSWtyDWPTb7/view?usp=drive_link) trained on VGG16 using learning rate = 0.0001, Adam optimizer, and 10 epochs

## Results

Our fovea classification model achieves the following performance on fovea_classification_codes folder:
| Model name         | Potential Fovea | Potential Non-fovea |
| ------------------ |---------------- | --------------      |
| best_VGG_model_1   |     80%         |      96.58%         |
