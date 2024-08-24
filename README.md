# CIFAR-10 Image Classification with Convolutional Neural Network

This repository contains a Python script to perform image classification on the CIFAR-10 dataset using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The script allows for training a model from scratch or loading a pre-trained model, classifying images, and visualizing the results.

## Table of Contents

- [CIFAR-10 Image Classification with Convolutional Neural Network](#cifar-10-image-classification-with-convolutional-neural-network)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
  - [Model Architecture](#model-architecture)
  - [Training and Evaluation](#training-and-evaluation)
  - [Prediction and Visualization](#prediction-and-visualization)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)

## Installation

Before running the script, you need to install the required Python libraries. You can install them using `pip`:

```bash
pip install tensorflow matplotlib
```

## Data Preparation

The script uses the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is loaded and preprocessed by normalizing the pixel values to be between 0 and 1.

```python
from tensorflow.keras import datasets

# Load and preprocess data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
```

## Model Architecture

The CNN model consists of the following layers:

- **Conv2D**: Three convolutional layers with ReLU activation.
- **MaxPooling2D**: Pooling layers to reduce dimensionality.
- **Flatten**: Flattens the input.
- **Dense**: Two dense layers, including the output layer with 10 units (one for each class).

```python
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10),
])
```

## Training and Evaluation

If a pre-trained model is found on disk, it is loaded; otherwise, the model is trained from scratch for 10 epochs.

```python
import os
import tensorflow as tf

model_path = "cifar10_cnn_model.h5"
if os.path.exists(model_path):
    # Load the model if it exists
    model = tf.keras.models.load_model(model_path)
    print("Model loaded from disk.")
else:
    # Compile and train the model if it doesn't exist
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(
        train_images,
        train_labels,
        epochs=10,
        validation_data=(test_images, test_labels),
    )

    # Save the model
    model.save(model_path)
    print("Model saved to disk.")
```

## Prediction and Visualization

The script includes a function `classify_image` to predict the class of a given image and a function `show_image_with_prediction` to display the image alongside its predicted and true labels.

```python
import matplotlib.pyplot as plt

def classify_image(image):
    img_array = tf.expand_dims(image, 0)  # Create a batch
    predictions = model.predict(img_array)
    predicted_class = tf.argmax(predictions[0]).numpy()
    return class_names[predicted_class]

def show_image_with_prediction(image, true_label):
    predicted_label = classify_image(image)
    plt.figure()
    plt.imshow(image)
    plt.title(f"Predicted: {predicted_label}, True: {true_label}")
    plt.axis("off")
    plt.show()

# Example usage
class_pred = classify_image(test_images[2])
show_image_with_prediction(test_images[2], class_pred)
```

## Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/cifar10-image-classification.git
   cd cifar10-image-classification
   ```

2. **Install the required libraries:**

   ```bash
   pip install tensorflow matplotlib
   ```

3. **Run the script:**

   ```bash
   python cifar10_cnn.py
   ```

4. **Use the `classify_image` function** to classify a new image.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
