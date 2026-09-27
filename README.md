# context-aware-text-to-3d-scene-generation
AI-based pipeline for generating spatially coherent 3D indoor environments from natural language using FLUX, SAM, SAM3D and MaterialGAN.
# Context-Aware Text-to-3D Scene Generation for Indoor Environments

An AI-based pipeline for generating spatially coherent 3D indoor environments from natural language descriptions.

## 📌 Overview

This project presents a multi-stage pipeline that converts a text description of an indoor environment into a structured 3D scene.

The pipeline combines text-to-image generation, object segmentation, 3D object reconstruction, pose estimation, material generation, and final scene composition.

## 🔄 Pipeline

Text Prompt
↓
FLUX Text-to-Image
↓
SAM Object Segmentation
↓
SAM 3D Object Reconstruction
↓
6D Pose Estimation
↓
Material Generation using MaterialGAN
↓
Final 3D Scene
↓
Unity Visualization

## 🧩 Project Pipeline

### 1. Text-to-Image Generation

A natural-language description is converted into a 2D indoor scene image using FLUX.

### 2. Object Segmentation

Objects in the generated scene are identified and segmented using SAM.

### 3. 3D Object Reconstruction

The segmented objects are converted into 3D representations using SAM 3D.

### 4. Pose Estimation

Object position and orientation are estimated to help place the reconstructed objects correctly in the scene.

### 5. Material Generation

Material information such as appearance, surface properties and textures is generated using MaterialGAN.

### 6. Final Scene Composition

The generated 3D objects, estimated poses and materials are combined to create the final indoor 3D scene.

## 📁 Repository Structure

```text
context-aware-text-to-3d-scene-generation/
│
├── README.md
│
├── Results/
│   ├── 01.Text_to_Image/
│   ├── 02.Object_Segmentation/
│   ├── 03.Sam_Output/
│   ├── 04.Pose_Estimation/
│   ├── 05.Material_Generation/
│   └── 06.Final_Results/
│
└── src/
