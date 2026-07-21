import streamlit as st
import os
import sys
from PIL import Image

# Get the absolute paths for both the app folder and the project root folder
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Add the project root directory to the Python path so it can find the 'utils' folder
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Since app.py and predictor.py live in the same folder, import it directly!
import predictor

st.set_page_config(page_title="Food-101 Classifier", page_icon="🍔")
st.title("🍔 Food-101 Classifier UI")

# Points directly to the model binary sitting out in your root folder
MODEL_PATH = os.path.join(project_root, "best_finetuned_model_gpu.keras")

@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        # Call the class from our direct import
        return predictor.FoodPredictor(path)
    return None

food_predictor = load_model(MODEL_PATH)

if food_predictor is None:
    st.error(f"Could not find `best_finetuned_model_gpu.keras` at {MODEL_PATH}")
else:
    uploaded_file = st.file_uploader("Upload a dish photo...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Target Image", use_container_width=True)
        
        # Save briefly to run the predictor pipeline
        temp_path = "temp_test_image.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        with st.spinner("Classifying dish..."):
            try:
                label, confidence = food_predictor.predict_image(temp_path)
                st.success(f"### Result: {label} ({confidence:.2f}%)")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
            
        if os.path.exists(temp_path):
            os.remove(temp_path)