import numpy as np
import tensorflow as tf
from PIL import Image

def load_and_prep_image(filename, img_shape=224):
    """
    Reads an image, converts to RGB, resizes it, and returns the raw array.
    EfficientNet model architectures natively handle their own internal rescaling.
    """
    img = Image.open(filename).convert('RGB')
    img = img.resize((img_shape, img_shape))
    img_array = np.array(img)  # Keep raw 0-255 pixel values intact
    
    return np.expand_dims(img_array, axis=0)