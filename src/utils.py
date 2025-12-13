import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import center_of_mass
import math
import os
import csv
from datetime import datetime

def load_image_from_bytes(file_bytes):
    """
    Decodes an image from bytes (from st.file_uploader or camera).
    Returns an RGB numpy array.
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(file_bytes.getvalue(), np.uint8)
    msg = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(msg, cv2.COLOR_BGR2RGB)
    return img_rgb

def get_best_shift(img):
    """
    Calculates the shift needed to center the image based on its center of mass.
    """
    cy, cx = center_of_mass(img)
    rows, cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)
    return shiftx, shifty

def shift(img, sx, sy):
    """
    Shifts the image by sx and sy.
    """
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted

def process_image(img_data):
    """
    Processes the raw image data from the canvas for the model.
    Steps:
    1. Grayscale conversion
    2. Resize to intermediate size
    3. Inversion (if needed)
    4. Centering by center of mass
    5. Normalization
    
    Args:
        img_data: numpy array from streamlit_drawable_canvas (RGBA)
        
    Returns:
        torch.Tensor: The processed image tensor ready for the model (1, 1, 28, 28)
        numpy.ndarray: The 28x28 image for visualization
    """
    # 1. Convert to grayscale
    if img_data.shape[-1] == 4:
        img_gray = cv2.cvtColor(img_data, cv2.COLOR_RGBA2GRAY)
    else:
        img_gray = img_data

    # 2. Resize to 20x20 while maintaining aspect ratio (inner box of 28x28)
    # MNIST digits are centered in a 20x20 box within the 28x28 image
    # But first, we need to crop the bounding box of the drawing to avoid scaling whitespace
    coords = cv2.findNonZero(img_gray)
    if coords is None: # Blank canvas
        return torch.zeros(1, 1, 28, 28), np.zeros((28, 28))
    
    x, y, w, h = cv2.boundingRect(coords)
    rect = img_gray[y:y+h, x:x+w]
    
    # Resize keeping aspect ratio to fit in 20x20
    rows, cols = rect.shape
    factor = 20.0 / max(rows, cols)
    rows = int(rows * factor)
    cols = int(cols * factor)
    rect_resized = cv2.resize(rect, (cols, rows), interpolation=cv2.INTER_AREA)
    
    # Pad to 28x28
    padded = np.zeros((28, 28), dtype=np.uint8)
    
    # Place result in center of 28x28
    pad_top = (28 - rows) // 2
    pad_left = (28 - cols) // 2
    padded[pad_top:pad_top+rows, pad_left:pad_left+cols] = rect_resized
    
    # 3. Center of Mass Adjustment
    # This is the "Smart Preprocessing"
    shiftx, shifty = get_best_shift(padded)
    shifted = shift(padded, shiftx, shifty)
    padded = shifted

    # 4. Normalize to [0, 1]
    final_img = padded / 255.0
    
    # 5. Convert to tensor
    tensor = torch.tensor(final_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # (1, 1, 28, 28)
    
    return tensor, final_img

def add_noise(tensor, noise_factor=0.0):
    """
    Adds Gaussian noise to the image tensor.
    Args:
        tensor: (1, 1, 28, 28) float tensor [0, 1]
        noise_factor: float 0.0 to 1.0 (strength)
    Returns:
        Noisy tensor
    """
    if noise_factor == 0:
        return tensor
    
    noise = torch.randn_like(tensor) * noise_factor
    noisy_tensor = tensor + noise
    # Clip to keep it valid image
    noisy_tensor = torch.clamp(noisy_tensor, 0., 1.)
    return noisy_tensor

def save_feedback(image_array, user_label, model_label):
    """
    Saves the image and labels to a CSV feedback loop.
    Implements a simple Active Learning pipeline.
    """
    file_exists = os.path.isfile('feedback.csv')
    
    # Save image to a folder
    os.makedirs('feedback_images', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_filename = f"feedback_images/img_{timestamp}_true{user_label}_pred{model_label}.png"
    
    # image_array is 0-1 float, convert to 0-255 uint8
    cv2.imwrite(img_filename, (image_array * 255).astype(np.uint8))
    
    with open('feedback.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'image_path', 'user_label', 'model_label'])
        
        writer.writerow([timestamp, img_filename, user_label, model_label])
        
    return True

# --- EXPLAINABILITY UTILS ---

def get_feature_maps(model, layer_name, input_tensor):
    """
    Hooks into the model to extract feature maps from a specific layer.
    """
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    # Find the layer by name
    for name, layer in model.named_modules():
        if name == layer_name:
            layer.register_forward_hook(get_activation(layer_name))
            break
            
    # Run inference
    with torch.no_grad():
        model(input_tensor)
        
    return activation.get(layer_name, None)

def get_gradcam(model, input_tensor, target_class=None):
    """
    Generates Grad-CAM heatmap for the last convolutional layer.
    """
    # We need gradients, so enable grad for input (though we care about layer grads)
    # But usually input doesn't need grad, the weights do. 
    # Actually for Grad-CAM we need gradients of Output w.r.t Feature Map.
    
    # 1. Hook for gradients and activations
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
        
    def forward_hook(module, input, output):
        activations.append(output)
        
    # Hook the last conv layer (conv3 in our model)
    handle_b = model.conv3.register_full_backward_hook(backward_hook)
    handle_f = model.conv3.register_forward_hook(forward_hook)
    
    # 2. Forward pass
    # Ensure tensor requires grad? No, model parameters do.
    # But we want to backprop from output.
    
    # Zero grads
    model.zero_grad()
    output = model(input_tensor)
    
    if target_class is None:
        target_class = torch.argmax(output)
        
    # 3. Backward pass
    # Target score
    score = output[0, target_class]
    score.backward()
    
    # 4. Generate Heatmap
    # Get gradients relative to feature map
    # grads: (1, 128, 4, 4) - depending on layer output size
    # acts: (1, 128, 4, 4)
    
    grads = gradients[0].cpu().data.numpy()[0]
    fmap = activations[0].cpu().data.numpy()[0]
    
    # Global Average Pooling of gradients
    weights = np.mean(grads, axis=(1, 2)) # (128,)
    
    # Weighted sum of feature maps
    cam = np.zeros(fmap.shape[1:], dtype=np.float32) # (4, 4)
    for i, w in enumerate(weights):
        cam += w * fmap[i]
        
    # ReLU
    cam = np.maximum(cam, 0)
    
    # Resize to input size (28x28)
    cam = cv2.resize(cam, (28, 28))
    
    # Normalize 0-1
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)
    
    # Clean up hooks
    handle_b.remove()
    handle_f.remove()
    
    return cam

