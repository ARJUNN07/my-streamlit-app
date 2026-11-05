"""Shared helpers for NutriLens AI."""
import json
import os
import numpy as np
import pandas as pd

# The 6 nutrition values the MLP / VAE work with (grams, sodium in mg)
FEATURES = ["protein", "carbs", "fat", "fibre", "sugar", "sodium"]

# Approximate nutrition per typical serving for the 10 food classes the CNN knows.
# (Approximate values - good enough for a demo. You can edit them.)
NUTRITION = {
    "chicken_curry":  {"protein": 25, "carbs": 12, "fat": 18, "fibre": 3, "sugar": 5,  "sodium": 700},
    "chicken_wings":  {"protein": 30, "carbs": 3,  "fat": 25, "fibre": 0, "sugar": 1,  "sodium": 900},
    "fried_rice":     {"protein": 8,  "carbs": 45, "fat": 10, "fibre": 2, "sugar": 3,  "sodium": 800},
    "grilled_salmon": {"protein": 34, "carbs": 0,  "fat": 13, "fibre": 0, "sugar": 0,  "sodium": 120},
    "hamburger":      {"protein": 25, "carbs": 35, "fat": 20, "fibre": 2, "sugar": 7,  "sodium": 800},
    "ice_cream":      {"protein": 4,  "carbs": 28, "fat": 14, "fibre": 1, "sugar": 24, "sodium": 80},
    "pizza":          {"protein": 12, "carbs": 36, "fat": 10, "fibre": 2, "sugar": 4,  "sodium": 640},
    "ramen":          {"protein": 10, "carbs": 55, "fat": 14, "fibre": 2, "sugar": 3,  "sodium": 1500},
    "steak":          {"protein": 40, "carbs": 0,  "fat": 20, "fibre": 0, "sugar": 0,  "sodium": 150},
    "sushi":          {"protein": 9,  "carbs": 38, "fat": 1,  "fibre": 1, "sugar": 6,  "sodium": 400},
}


def make_nutrition_data(n=5000, seed=42):
    """Create a nutrition dataset. Calories follow the standard Atwater formula
    (4 kcal/g protein, 4 kcal/g carbs, 9 kcal/g fat) plus a little noise,
    so the MLP has something real to learn."""
    rng = np.random.default_rng(seed)
    protein = rng.uniform(0, 60, n)
    fat = rng.uniform(0, 60, n)
    carbs = rng.uniform(0, 100, n)
    fibre = carbs * rng.uniform(0, 0.2, n)
    sugar = carbs * rng.uniform(0, 0.6, n)
    sodium = rng.uniform(0, 1500, n)
    calories = 4 * protein + 4 * carbs + 9 * fat + rng.normal(0, 10, n)
    df = pd.DataFrame({"protein": protein, "carbs": carbs, "fat": fat,
                       "fibre": fibre, "sugar": sugar, "sodium": sodium,
                       "calories": np.clip(calories, 0, None)})
    return df


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None
