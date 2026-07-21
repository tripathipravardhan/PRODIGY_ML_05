import numpy as np
import tensorflow as tf
from utils.preprocess import load_and_prep_image
from utils.nutrition_data import get_nutrition_info

CLASS_NAMES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheese_plate", "cheesecake", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
    "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
    "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
    "grilled_salmon", "guacamole", "gyro", "hamburger", "hot_and_sour_soup",
    "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
    "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
    "mussels", "nachos", "omelette", "onion_rings", "oysters",
    "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
    "pho", "pizza", "pork_chop", "poutine", "prime_rib",
    "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
    "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
    "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
    "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles"
]

class FoodPredictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def predict_image(self, image_path):
        # 1. Preprocess raw tensor
        img_array = load_and_prep_image(image_path)
        
        # 2. Predict probabilities across all classes
        preds = self.model.predict(img_array)[0]
        
        # Get top 3 indices sorted by confidence
        top_3_indices = np.argsort(preds)[::-1][:3]
        
        top_3_results = []
        for idx in top_3_indices:
            raw_label = CLASS_NAMES[idx]
            formatted_label = raw_label.replace("_", " ").title()
            confidence = float(preds[idx]) * 100
            nutrition = get_nutrition_info(raw_label)
            top_3_results.append({
                "raw_label": raw_label,
                "label": formatted_label,
                "confidence": confidence,
                "nutrition": nutrition
            })
            
        return top_3_results