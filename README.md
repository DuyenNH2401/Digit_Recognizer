# Digit Recognizer With Pytorch CNN

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://duyennh-digit-recognizer.streamlit.app/)

Built with **PyTorch** and **Streamlit**.

## Key Features

### 1. Advanced Inputs & Preprocessing
-   **Dual Input Modes**: Support for both **Interactive Canvas** (drawing) and **Image Uploads**.
-   **Smart Preprocessing**: Implementation of a **"Center of Mass"** alignment algorithm. This ensures the model works reliably on real-world inputs (off-center drawings, photos) by mimicking the original MNIST data distribution.
-   **Auto-Inversion**: Automatically detects and inverts "Black text on White paper" images to match the model's training data.

### 2. Explainable AI (XAI)
-   **Grad-CAM Heatmaps**: Real-time visualization of the model's "Attention". See exactly which strokes contributed to the prediction (Warm colors = High Focus).
-   **Feature Maps**: Visual inspection of the internal **Convolutional Filters**, showing how the AI detects edges and curves.
-   **Confidence Breakdown**: Full probability distribution chart, not just the top prediction.

### 3. MLOps & Robustness
-   **Noise Injection**: A "Stress Test" slider to add Gaussian noise to inputs, demonstrating the model's stability (or lack thereof) under degradation.
-   **Active Learning Loop (Feedback)**: A "Human-in-the-loop" system. Users can correct wrong predictions, which saves the data to a local dataset for future model retraining.

## Project Structure

```bash
Project/
├── app.py              # Main Streamlit Application
├── src/                # Modular Source Code
│   ├── model.py        # PyTorch CNN Architecture
│   └── utils.py        # Core Logic (Preprocessing, GradCAM, Feedback)
├── models/             # Weighted Files
│   └── model.pth       # Custom Trained MNIST Model
├── notebooks/          # Exploratory Data Analysis & Training
│   └── training.ipynb  # Original Training Notebook
├── requirements.txt    # Project Dependencies
└── .streamlit/         # UI Configuration
    └── config.toml     # Custom Theme Settings
```

## How to Run Locally

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/DuyenNH2401/Digit_Recognizer.git
    cd Digit_Recognizer
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**:
    ```bash
    streamlit run app.py
    ```

## Model Architecture

A custom CNN designed for efficiency and accuracy:
-   **Input**: 1x28x28 Grayscale
-   **Conv Block 1**: 32 Filters (3x3) + ReLU + MaxPool
-   **Conv Block 2**: 64 Filters (3x3) + ReLU + MaxPool
-   **Conv Block 3**: 128 Filters (3x3) + ReLU
-   **Classifier**: Flatten -> Dense (128) -> Dropout (0.5) -> Output (10)

## 👤 Author

**Duyen Nguyen**





