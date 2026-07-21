import streamlit as st
import os
import sys
import pandas as pd
from PIL import Image
from datetime import datetime
import gc  # Added for memory cleanup
import requests

# ---------------------------------------------------------
# PATH SETUP (Fixes ModuleNotFoundError for utils & predictor)
# ---------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))

# Add both root directory and app directory to sys.path
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import predictor
from utils.nutrition_data import get_nutrition_info, FOOD_NUTRITION

# Page Configuration
st.set_page_config(
    page_title="NutriVision AI | Smart Food & Macro Tracker", 
    page_icon="🥗", 
    layout="wide"
)

# Custom CSS for Professional Styling & Centering
st.markdown("""
    <style>
        .main-header {
            text-align: center;
            padding-bottom: 1.5rem;
        }
        .upload-section {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 2rem;
        }
        div[data-testid="stFileUploader"] {
            width: 70%;
            margin: 0 auto;
        }
        .metric-card {
            background-color: #1e222d;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            border: 1px solid #2e3440;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------------
if "meal_journal" not in st.session_state:
    st.session_state.meal_journal = []

# ---------------------------------------------------------
# SIDEBAR: HEALTH GOAL ADVISOR
# ---------------------------------------------------------
st.sidebar.title("🎯 Health & Fitness Goals")
daily_budget = st.sidebar.number_input("Daily Calorie Budget (kcal):", min_value=1000, max_value=5000, value=2000, step=50)

# Calculate totals from current session journal
total_logged_calories = sum(item["Calories (kcal)"] for item in st.session_state.meal_journal)
total_logged_protein = sum(item["Protein (g)"] for item in st.session_state.meal_journal)
total_logged_carbs = sum(item["Carbs (g)"] for item in st.session_state.meal_journal)
total_logged_fats = sum(item["Fats (g)"] for item in st.session_state.meal_journal)

remaining_calories = daily_budget - total_logged_calories

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Daily Budget Progress")
st.sidebar.metric("Calories Consumed", f"{total_logged_calories} kcal", delta=f"{remaining_calories} kcal remaining")
st.sidebar.progress(min(1.0, total_logged_calories / daily_budget))
st.sidebar.caption(f"**{ (total_logged_calories / daily_budget) * 100:.1f}%** of daily budget used.")

st.sidebar.markdown("---")
st.sidebar.write(f"💪 **Protein:** {total_logged_protein} g")
st.sidebar.write(f"🌾 **Carbs:** {total_logged_carbs} g")
st.sidebar.write(f"🥑 **Fats:** {total_logged_fats} g")

# ---------------------------------------------------------
# CENTERED HERO HEADER
# ---------------------------------------------------------
st.markdown("<div class='main-header'><h1>🥗 NutriVision AI</h1><p style='font-size:1.1rem; color:#888;'>Upload your meal photo to instantly classify dishes, scale portions, and track daily macro intake.</p></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# MODEL DOWNLOAD & LOADING (Fixes Git LFS & missing path errors)
# ---------------------------------------------------------
MODEL_FILENAME = "best_finetuned_model_gpu.keras"
MODEL_PATH = os.path.join(ROOT_DIR, MODEL_FILENAME)
MODEL_URL = "https://media.githubusercontent.com/media/tripathipravardhan/PRODIGY_ML_05/main/best_finetuned_model_gpu.keras"

def ensure_model_downloaded(target_path):
    # Check if the file is missing or if it's just a tiny LFS pointer file (< 1 MB)
    if not os.path.exists(target_path) or os.path.getsize(target_path) < 1_000_000:
        with st.spinner("⏬ Fetching ML model weights from GitHub (33 MB)..."):
            response = requests.get(MODEL_URL, stream=True)
            if response.status_code == 200:
                with open(target_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                st.error("Failed to download model file from GitHub LFS.")

ensure_model_downloaded(MODEL_PATH)

@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return predictor.FoodPredictor(path)
    return None

food_predictor = load_model(MODEL_PATH)

if food_predictor is None:
    st.error(f"Could not load model file at: `{MODEL_PATH}`")
else:
    # ---------------------------------------------------------
    # CENTERED FILE UPLOADER
    # ---------------------------------------------------------
    center_col1, center_col2, center_col3 = st.columns([1, 2, 1])
    with center_col2:
        uploaded_file = st.file_uploader("📸 Drop or upload a food photo here...", type=["jpg", "jpeg", "png"])

    st.markdown("---")

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1.2], gap="large")
        
        image = Image.open(uploaded_file)
        with col1:
            st.image(image, caption="Analyzed Dish", use_container_width=True)
            
        with col2:
            with st.spinner("Analyzing image features & mapping nutritional metrics..."):
                try:
                    # Reset file pointer and pass raw uploaded_file to predictor safely
                    uploaded_file.seek(0)
                    results = food_predictor.predict_image(uploaded_file)
                    
                    top_match = results[0]
                    confidence = top_match["confidence"]
                    
                    # Low Confidence Warning & Manual Override
                    if confidence < 40.0:
                        st.warning(f"⚠️ Low confidence prediction ({confidence:.1f}%). Confirm or select your dish manually:")
                        all_classes_sorted = sorted([k.replace("_", " ").title() for k in FOOD_NUTRITION.keys()])
                        selected_food = st.selectbox("Correct Dish:", all_classes_sorted, index=all_classes_sorted.index(top_match['label']))
                        active_label = selected_food
                        active_nutrition = get_nutrition_info(selected_food)
                    else:
                        st.success(f"### Identified Dish: **{top_match['label']}**")
                        active_label = top_match['label']
                        active_nutrition = top_match["nutrition"]
                    
                    # Portion Scaler
                    servings = st.slider("🍽️ **Select Portion / Serving Size:**", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
                    
                    calc_calories = int(active_nutrition["calories"] * servings)
                    calc_protein = int(active_nutrition["protein"] * servings)
                    calc_carbs = int(active_nutrition["carbs"] * servings)
                    calc_fat = int(active_nutrition["fat"] * servings)
                    
                    # Goal Advisor Message
                    pct_of_day = (calc_calories / daily_budget) * 100
                    st.info(f"💡 This portion covers **{pct_of_day:.1f}%** of your target daily budget ({daily_budget} kcal).")

                    st.markdown("### 📊 Calculated Nutrition Facts")
                    
                    n_col1, n_col2, n_col3, n_col4 = st.columns(4)
                    n_col1.metric("🔥 Calories", f"{calc_calories} kcal")
                    n_col2.metric("💪 Protein", f"{calc_protein} g")
                    n_col3.metric("🌾 Carbs", f"{calc_carbs} g")
                    n_col4.metric("🥑 Fats", f"{calc_fat} g")
                    
                    # Macro Breakdown Chart
                    macro_data = pd.DataFrame({
                        "Nutrient": ["Protein", "Carbs", "Fats"],
                        "Grams": [calc_protein, calc_carbs, calc_fat]
                    }).set_index("Nutrient")
                    st.bar_chart(macro_data, height=160)
                    
                    # Log Meal Action
                    if st.button("➕ Log Meal to Daily Journal", type="primary", use_container_width=True):
                        st.session_state.meal_journal.append({
                            "Timestamp": datetime.now().strftime("%I:%M %p"),
                            "Dish": active_label,
                            "Servings": servings,
                            "Calories (kcal)": calc_calories,
                            "Protein (g)": calc_protein,
                            "Carbs (g)": calc_carbs,
                            "Fats (g)": calc_fat
                        })
                        st.toast(f"Successfully logged {active_label} ({calc_calories} kcal)!", icon="✅")
                        st.rerun()

                    st.markdown("---")
                    with st.expander("🔍 **View Model Confidence Breakdown**"):
                        for item in results:
                            st.write(f"**{item['label']}** — `{item['confidence']:.2f}%`")
                            st.progress(int(item['confidence']) / 100)
                            
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                finally:
                    # Clear RAM memory after execution
                    gc.collect()

# ---------------------------------------------------------
# DAILY MEAL LOG & EXPORT SECTION
# ---------------------------------------------------------
st.markdown("---")
st.header("📖 Today's Meal Journal")

if len(st.session_state.meal_journal) > 0:
    df_journal = pd.DataFrame(st.session_state.meal_journal)
    st.dataframe(df_journal, use_container_width=True)
    
    col_dl1, col_dl2 = st.columns([1, 4])
    with col_dl1:
        csv_data = df_journal.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Journal (CSV)",
            data=csv_data,
            file_name=f"meal_journal_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_dl2:
        if st.button("🗑️ Clear Journal"):
            st.session_state.meal_journal = []
            st.rerun()
else:
    st.info("No meals logged yet today. Upload a photo above and click 'Log Meal to Daily Journal'.")
