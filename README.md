# Deep Learning Architectures & Implementations

This repository contains a collection of deep learning projects and experiments implemented from scratch and using high-level frameworks. The focus is on understanding core architectures in Computer Vision, NLP, and Sequence Models, ranging from fundamental CNNs/RNNs to modern Transformers and Generative models.

## 🧠 Computer Vision

### Autonomous Driving - Car Detection

- **Architecture**: YOLO (You Only Look Once)
- **Key Concepts**: Object detection, Bounding Box Regression, Non-max Suppression, Intersection over Union (IoU), Anchor Boxes.
- **Implementation**: Real-time object detection pipeline capable of identifying vehicles in video streams.

### Face Recognition & Verification

- **Architecture**: Inception Network / FaceNet
- **Key Concepts**: One-Shot Learning, Triplet Loss, Siamese Networks, Encoding Generation.
- **Implementation**: A facial recognition system that learns 128-dimensional encodings to verify identities with high accuracy even with minimal training examples.

### Semantic Image Segmentation

- **Architecture**: U-Net
- **Key Concepts**: Transpose Convolutions, Skip Connections, Pixel-level Classification.
- **Implementation**: Applied to self-driving car datasets to segment drivable areas and obstacles, maintaining spatial resolution through the encoder-decoder structure.

### Deep Residual Networks

- **Architecture**: ResNet-50
- **Key Concepts**: Residual Blocks, Skip Connections, Vanishing Gradient mitigation.
- **Implementation**: Built a 50-layer ResNet to solve the degradation problem in deep networks, trained for image classification.

### Neural Style Transfer

- **Architecture**: VGG-19 (Pre-trained)
- **Key Concepts**: Content vs. Style Cost Functions, Transfer Learning, Gram Matrices.
- **Implementation**: An optimization-based approach to generate artistic images by combining the content of a photograph with the artistic style of a painting.

### Transfer Learning with MobileNetV2

- **Architecture**: MobileNetV2
- **Key Concepts**: Transfer Learning, Fine-tuning, Data Augmentation.
- **Implementation**: Adapted a pre-trained MobileNetV2 model to classify custom image datasets by freezing base layers and training a new classification head.

---

## 🗣️ Natural Language Processing (NLP)

### Transformer Network

- **Architecture**: Transformer (Attention is All You Need)
- **Key Concepts**: Self-Attention, Multi-Head Attention, Positional Encoding, Scaled Dot-Product Attention.
- **Implementation**: Built a full Transformer architecture from scratch to handle sequence data without recurrence, enabling parallelization.

### Transformer Applications

- **Question Answering**: Implemented an extractive QA model using the bAbI dataset to identify answers within text context.
- **Named-Entity Recognition (NER)**: Developed a system to parse resumes and extract structured entities (Skills, Email, etc.) using Transformer-based token classification.

### Neural Machine Translation

- **Architecture**: Seq2Seq with Attention
- **Key Concepts**: Encoder-Decoder, Attention Mechanism, Bi-directional LSTMs.
- **Implementation**: A translation model capable of converting human-readable date formats into standardized machine-readable formats.

### Word Vector Debiasing

- **Key Concepts**: GloVe Embeddings, Cosine Similarity, Subspace Projection, Bias Neutralization.
- **Implementation**: Analysis and mitigation of gender bias in word embeddings by projecting words onto non-biased subspaces.

### Emojify!

- **Architecture**: LSTM + Word Embeddings (GloVe)
- **Key Concepts**: Word Vector Representations, Sentiment Classification, Many-to-One RNN.
- **Implementation**: A sentiment analysis model that classifies sentences and associates them with appropriate emojis, handling distinct phrasing and context.

---

## 🎵 Audio & Sequence Models

### Character-Level Language Model (Dinosaurus)

- **Architecture**: RNN / LSTM
- **Key Concepts**: Character-level text generation, Sampling, Gradient Clipping.
- **Implementation**: Trained a language model on a dataset of dinosaur names to generate novel, plausible names character by character.

### Trigger Word Detection

- **Architecture**: 1D CNN + GRU/LSTM
- **Key Concepts**: Speech Recognition, Spectrogram Processing, Sliding Windows.
- **Implementation**: A wake-word detection system (similar to "Alexa" or "Siri") that activates upon detecting a specific keyword in an audio stream.

### Jazz Solo Generation

- **Architecture**: LSTM
- **Key Concepts**: Sequence Generation, Music Processing, Temporal Dependencies.
- **Implementation**: A generative LSTM model trained on a corpus of jazz music to improvise and generate novel jazz solos.

---

## 📚 Deep Learning Fundamentals

### Neural Network Basics

- **Vectorization**: Introduction to vectorization using NumPy to replace explicit for-loops, significantly speeding up large-scale data processing.
- **Shallow & Deep Networks**: Implementation of forward and backward propagation from scratch (NumPy) for L-layer neural networks.
- **Deep Neural Network Application**: Built an L-layer deep neural network to classify images (Cat vs Non-Cat), demonstrating performance improvement over shallow models.
- **Optimization Algorithms**: Implementation and comparison of SGD, Momentum, RMSProp, and Adam optimizers.
- **TensorFlow Introduction**: Foundational implementations of neural networks using the TensorFlow framework.

### Hyperparameter Tuning & Regularization

- **Regularization**: Implemented L2 regularization and Dropout to prevent overfitting in a 2D football player position dataset.
- **Gradient Checking**: Verified backpropagation correctness using gradient checking in the context of a fraud detection model.
- **Initialization**: Analyzed the impact of different weight initialization methods (Zeros, Random, He) on model convergence.

### Convolutional Neural Networks

- **Step-by-Step Implementation**: Built convolution and pooling layers (forward and backward prop) from scratch using NumPy to understand the mechanics of spatial feature extraction.
- **TensorFlow Application**: Implemented a CNN to classify images (Happy House dataset) using the TensorFlow Keras API.

### Recurrent Neural Networks

- **Step-by-Step Implementation**: Built RNN and LSTM cells (forward and backward prop) from scratch to understand temporal dependency processing and gating mechanisms.

### Logistic Regression as a Neural Network

- **Basics**: Implemented a logistic regression classifier to recognize cats, serving as an introduction to neural network structure (forward/backward propagation, cost function).

---

## Getting Started

### Prerequisites

- Python 3.7+
- pip
- Jupyter Notebook or JupyterLab

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a virtual environment (optional but recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

Start the Jupyter Notebook server:

```bash
jupyter notebook
```

Navigate to the directory of the project you want to explore and open the `.ipynb` file.

---

## Tech Stack

- **Frameworks**: TensorFlow, Keras
- **Libraries**: NumPy, Pandas, Matplotlib, SciPy
- **Tools**: Jupyter Notebooks
