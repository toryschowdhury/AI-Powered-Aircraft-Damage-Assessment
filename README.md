
# Automated Aircraft Damage Assessment using Deep Learning

This project presents a multi-modal AI solution for automating the detection and reporting of aircraft damage. It combines a Convolutional Neural Network (CNN) for damage classification with a Vision-Language model for generating descriptive captions, aiming to improve the efficiency and accuracy of aircraft maintenance inspections.

## Project Overview

Traditional aircraft inspections are manual, time-consuming, and can be subjective. This project addresses these challenges by developing a two-part automated system:

1.  **Damage Classification**: A pre-trained VGG16 model is fine-tuned to classify images of aircraft parts into "dent" or "crack" categories. This provides a rapid and objective method for initial damage identification.

2.  **Automated Reporting/Captioning**: A pre-trained BLIP (Bootstrapping Language-Image Pre-training) model is used to automatically generate natural language captions and summaries for the damage shown in an image. This capability is a step towards automated report generation for maintenance logs.

The project demonstrates the power of combining different neural network architectures to create a comprehensive solution and showcases the integration of PyTorch-based models (from Hugging Face) within a TensorFlow/Keras workflow.

## Dataset

The model was trained on the [Aircraft Damage Dataset from Roboflow](httpss://universe.roboflow.com/youssef-donia-fhktl/aircraft-damage-detection-1j9qk), which contains labeled images of dents and cracks on aircraft surfaces.

The specific version used in this project can be downloaded [here](httpss://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZjXM4RKxlBK9__ZjHBLl5A/aircraft-damage-dataset-v1.tar).

## Key Features

-   **Image Classification**: Achieved ~84% accuracy in classifying dents vs. cracks on the test dataset.
-   **Image Captioning**: Generates relevant, human-readable descriptions of damage from images.
-   **Transfer Learning**: Leverages the power of pre-trained VGG16 and BLIP models, significantly reducing training time and data requirements.
-   **Framework Integration**: Successfully integrates a PyTorch-based Hugging Face model into a TensorFlow/Keras pipeline using a custom layer.

## Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. You will also need to install the libraries listed in the `requirements.txt` file.

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/aircraft-damage-assessment.git](https://github.com/your-username/aircraft-damage-assessment.git)
    cd aircraft-damage-assessment
    ```

2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  Download the dataset and place it in the project's root directory.
2.  Open the `Classification and Captioning Aircraft Damage Using Pretrained Models.ipynb` notebook in Jupyter or another compatible environment.
3.  Run the cells in the notebook to train the classification model and see examples of caption generation.

## Results

-   **Classification Model**:
    -   Training Accuracy: ~84.5%
    -   Validation Accuracy: ~72.9%
    -   Test Accuracy: ~84.4%
-   The model shows good learning but also signs of overfitting, which could be mitigated with more data or advanced augmentation techniques.

-   **Captioning Model**:
    -   The BLIP model generated contextually relevant captions and summaries for test images, successfully describing the visual content in natural language.

## Future Work

-   Expand the number of damage categories (e.g., scratches, corrosion).
-   Implement object detection (e.g., YOLO, Faster R-CNN) to first locate the damage on a larger surface before classifying it.
-   Optimize the models for deployment on edge devices for real-time inspection.
