# Few-shot-classification-Project

# Image Classification with Pre-trained Models

This repository demonstrates image classification using pre-trained models (ResNet50 and EfficientNetB0) with TensorFlow and Keras. The code showcases the steps for data preprocessing, model configuration, training, and evaluation.

## ResNet50 Model

### Data Preprocessing
- Utilizes `ImageDataGenerator` for data augmentation and normalization.
- Loads images from a specified directory using `flow_from_directory`.

### Model Creation
- Loads ResNet50 with pre-trained weights.
- Adds custom classifier layers on top of the base architecture.
- Compiles the model using Adam optimizer and categorical cross-entropy loss.

### Model Training
- Trains the model for a specified number of epochs using `data_generator`.
- Evaluates model performance on test data.

## EfficientNetB0 Model

### Data Preprocessing
- Similar data preprocessing steps as ResNet50.

### Model Creation
- Loads EfficientNetB0 with pre-trained weights.
- Adds custom classifier layers.
- Compiles the model with Adam optimizer and categorical cross-entropy loss.

### Model Training
- Trains the model using `data_generator` for a set number of epochs.
- Evaluates model performance on test data.

## Key Points
- Both models utilize transfer learning with pre-trained weights.
- Similarities exist in data preprocessing, model setup, compilation, training, and evaluation.
- Configurable parameters such as output classes or epochs can be adjusted for experimentation.

The provided code serves as an example for training image classification models using established pre-trained architectures. Adaptations or enhancements can be made to suit specific datasets or use cases.
