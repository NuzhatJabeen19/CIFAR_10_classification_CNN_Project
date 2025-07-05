# CIFAR-10 Image Classification with CNNs and Transfer Learning

This project demonstrates image classification on the CIFAR-10 dataset using both custom Convolutional Neural Networks (CNNs) and transfer learning with ResNet50. The notebook explores data preprocessing, model training, evaluation, and comparison of different deep learning approaches.

## Project Structure

```
CIFAR_10_classification_project.ipynb
model_cp/
    exp1.model.keras
    exp2.model.keras
    exp3.model.keras
    exp4.model.keras
Models/
    cifar10_model_3_architecture.json
    cifar10_model_3_custom.h5
    cifar10_model_3_history.json
    cifar10_model_3.h5
    cifar10_model_4_architecture.json
    cifar10_model_4_custom.h5
    cifar10_model_4_history.json
    cifar10_model_4.h5
```

## Features

- **Data Loading & Visualization:** Loads CIFAR-10, visualizes samples and label distribution.
- **Preprocessing:** Normalizes images and prepares categorical labels.
- **Model 1:** Transfer learning with ResNet50 (frozen layers).
- **Model 2:** Transfer learning with ResNet50, data augmentation, and input resizing.
- **Model 3:** Fine-tuned ResNet50 (all layers trainable).
- **Model 4:** Custom CNN built from scratch.
- **Training & Evaluation:** Includes training/validation plots, accuracy/loss metrics, and confusion matrices.
- **Model Saving:** Saves trained models, architectures, and training histories for reproducibility.

## How to Run

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/cifar10-classification.git
    cd cifar10-classification
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```
    Or manually install:
    ```sh
    pip install tensorflow matplotlib pandas seaborn scikit-learn
    ```

3. **Run the notebook:**
    Open `CIFAR_10_classification_project.ipynb` in Jupyter Notebook or VS Code and run all cells.

## Results

| Model Description                                                        | Train Accuracy | Validation Accuracy | Test Accuracy |
|--------------------------------------------------------------------------|:--------------:|:------------------:|:-------------:|
| **ResNet50 Model: With Transfer Learning**                               |     0.46       |        0.43        |     0.42      |
| **ResNet50 Model: With Data Augmentation + Input Resizing to 224x224**   |     0.16       |        0.21        |     0.20      |
| **ResNet50 Model: Transfer Learning with all layers unfrozen**           |     0.85       |        0.74        |     0.72      |
| **CNN Model from Scratch**                                               |     0.88       |        0.74        |     0.72      |

## Insights

- **Model 1 (ResNet50 Transfer Learning):**  
  Achieves moderate accuracy, showing the benefit of transfer learning, but is limited by not fine-tuning deeper layers.

- **Model 2 (ResNet50 + Data Augmentation + Resizing):**  
  Performs poorly, suggesting that data augmentation and resizing alone are not sufficient and may require further tuning or more training epochs.

- **Model 3 (ResNet50, all layers unfrozen):**  
  Fine-tuning the entire ResNet50 model leads to a significant boost in performance, indicating the importance of allowing the model to adapt to the new dataset.

- **Model 4 (Custom CNN from Scratch):**  
  Achieves the highest training accuracy and matches the fine-tuned ResNet50 on validation and test accuracy, demonstrating that a well-designed CNN can perform as well as transfer learning when trained properly.

## Model Checkpoints & Saved Models

- Model checkpoints are saved in the `model_cp/` directory.
- Final trained models, architectures, and histories are saved in the `Models/` directory for easy loading and inference.

## License



---

**Author:** Nuzhat Jabeen Amna
**Contact:** nuzhat.jabeen.amna19@gmail.com

