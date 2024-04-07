# Grain Boundary Classification for Bicrystalline Materials using Molecular Dynamic Simulation Images

This repository hosts the implementation of a one-shot learning approach using Siamese networks for classifying grain boundaries in bicrystalline materials. The project leverages the unique capabilities of Siamese neural networks to effectively predict grain boundary types from molecular dynamics (MD) simulation images with minimal example data.

## How It Works

The application workflow is straightforward:

- **Upload an Image**: Users can upload an image of a bicrystalline material's grain boundary obtained from MD simulations.
- **Model Prediction**: Our pre-trained Siamese network analyzes the image to determine the type of grain boundary.
- **Results**: The app displays the predicted grain boundary type, the confidence of the prediction, and relevant energy values associated with the boundary.

## Why One-Shot Learning?

One-shot learning is particularly beneficial in fields where data is scarce or expensive to obtain:

- **Efficiency**: It requires significantly fewer training examples compared to traditional machine learning models.
- **Accuracy**: Despite the limited data, it can achieve high accuracy by focusing on learning distinctive features that differentiate between classes.
- **Speed**: Reduces the time and resources required for training complex models.

## Future Directions

### Application in Polycrystalline Materials

We are actively working to expand the capabilities of this application to include grain-boundary recognition in polycrystalline materials. Future updates will enable the tool not only to classify but also to identify and locate various types of grain boundaries across different polycrystalline materials. This development aims to provide a comprehensive tool for detailed analysis of GB structures, supporting advancements in materials science research and applications.

## Getting Started

To get started with this project, clone the repository and follow the setup instructions:

```bash
git clone https://github.com/VedantZope/Automatic-Sigma-Grain-Boundary-Detection.git
cd Automatic-Sigma-Grain-Boundary-Detection
python GB_identification.py
streamlit run Deployment/app.py
