# Neural Evolution for Handwritten Digit Recognition using Differential Evolution (DE)

This project explores the performance of optimization techniques, focusing on the effect of over-selection for large populations. Specifically, the methods of Differential Evolution (DE) and neural networks using scikit-learn's MLPClassifier are applied to a digits dataset similar to MNIST.

## Project Objectives

* Evaluate the performance of Differential Evolution for optimizing neural networks.
* Compare results with a traditional gradient descent-based MLPClassifier.
* Analyze the learning curves, classification reports, and visualizations of predictions.

## Dataset

The project uses the digits dataset from scikit-learn, which contains images of handwritten digits (0-9) with the following features:

* **Features**: 64 (8x8 grayscale pixel values).
* **Labels**: 10 classes (digits 0 to 9).
* **Size**: 1797 samples.

The dataset is normalized for better performance.

## Key Steps

1. **Loading and Preprocessing**

   * Normalized the dataset to bring pixel values between 0 and 1.
   * One-hot encoded the labels for compatibility with neural network training.

2. **Differential Evolution Optimization**

   * Custom implementation of DE for neural network parameter optimization.
   * Visualized progress using custom loss curves.

3. **Gradient Descent Optimization**

   * Trained an MLPClassifier with hidden layers.
   * Plotted its loss curve and compared it with DE.

4. **Evaluation and Visualization**

   * Plotted confusion matrices and ROC curves for multiclass classification.
   * Compared metrics like accuracy and runtime for both methods.
   * Visualized misclassified examples and learning progress.

## Results

* Comparative plots demonstrate the effectiveness of DE vs. gradient descent.
* Metrics such as test accuracy, training time, and classification performance are reported.
* DE showed advantages in certain scenarios, particularly for large populations, but required more computational resources.

## Requirements

The project uses the following Python libraries:

* `numpy`
* `matplotlib`
* `scikit-learn`
* `random`
* Other standard libraries for data manipulation and visualization.

## How to Run

1. Clone the repository.
2. Ensure all required libraries are installed.
3. Run the notebook step-by-step to reproduce the results.

