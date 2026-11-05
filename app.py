"""NutriLens AI - the website (Streamlit).  Run:  streamlit run app.py"""
import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from utils import FEATURES, NUTRITION, load_json

st.set_page_config(page_title="NutriLens AI", page_icon="🥗", layout="wide")


# ---------- helpers ----------
def need(path, script):
    """Stop the page politely if a trained model file is missing."""
    if not os.path.exists(path):
        st.warning(f"Model file `{path}` not found. Run `python {script}` first.")
        st.stop()


@st.cache_resource
def load_model(path):
    import tensorflow as tf
    return tf.keras.models.load_model(path)


@st.cache_resource
def load_scaler(path):
    import joblib
    return joblib.load(path)


def predict_calories(values):
    """values: dict with the 6 nutrition features -> predicted calories (MLP)."""
    need("models/mlp_calories.keras", "train_mlp.py")
    model, scaler = load_model("models/mlp_calories.keras"), load_scaler("models/mlp_scaler.pkl")
    x = scaler.transform(np.array([[values[f] for f in FEATURES]], dtype="float32"))
    return float(model.predict(x, verbose=0)[0][0]) * 1000.0


def pretty(name):
    return name.replace("_", " ").title()


def health_score(n):
    """Simple 0-100 score: rewards protein/fibre, penalises sugar, sodium, fat."""
    s = 60 + n["protein"] * 0.8 + n["fibre"] * 3 - n["sugar"] * 0.8 - n["sodium"] * 0.01 - n["fat"] * 0.5
    return int(max(0, min(100, s)))


def make_report(food, n, calories, goal):
    """MODULE 5 (generative AI): uses an LLM if ANTHROPIC_API_KEY is set,
    otherwise falls back to a built-in template so the page always works."""
    prompt = (f"Write a short, friendly nutrition report (under 150 words) for a serving of {pretty(food)}: "
              f"about {calories:.0f} kcal, protein {n['protein']} g, carbs {n['carbs']} g, fat {n['fat']} g, "
              f"fibre {n['fibre']} g, sugar {n['sugar']} g, sodium {n['sodium']} mg. "
              f"The person's goal is: {goal}. Include a verdict, one healthier tip and one side-dish suggestion.")
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            msg = anthropic.Anthropic().messages.create(
                model="claude-sonnet-5-5", max_tokens=400,
                messages=[{"role": "user", "content": prompt}])
            return msg.content[0].text, "LLM (Claude API)"
        except Exception as e:  # fall through to template
            st.info(f"LLM unavailable ({e}). Using template report.")
    score = health_score(n)
    verdict = "a healthy choice" if score >= 65 else "an okay choice in moderation" if score >= 40 else "best as an occasional treat"
    tips = []
    if n["sodium"] > 700: tips.append("sodium is high, so go easy on added salt and sauces")
    if n["sugar"] > 15: tips.append("sugar is high, so share the portion or pair it with fruit")
    if n["fat"] > 18: tips.append("fat is on the higher side, so balance it with a light, veggie-rich next meal")
    if n["protein"] >= 25: tips.append("it is a strong protein source, which supports muscle and keeps you full")
    if n["fibre"] < 2: tips.append("fibre is low, so add a salad or steamed vegetables")
    goal_line = {"Weight loss": "Keep the portion small and add vegetables to stay full on fewer calories.",
                 "Muscle gain": "Pair it with a protein-rich side such as eggs, curd or lentils.",
                 "Balanced diet": "Combine it with vegetables and a whole grain to round out the meal."}[goal]
    text = (f"**{pretty(food)}** - about **{calories:.0f} kcal** per serving. Health score: **{score}/100**, "
            f"{verdict}.\n\n" + ("; ".join(tips).capitalize() + ".\n\n" if tips else "") + f"**For your goal ({goal}):** {goal_line}")
    return text, "Template (no API key set)"


# ---------- pages ----------
def page_home():
    st.title("🥗 NutriLens AI")
    st.subheader("Snap your food. Know your nutrition. Get your recipe.")
    st.write("An AI-powered food and nutrition assistant built with deep learning. "
             "Upload a photo of a meal and NutriLens identifies it, estimates its nutrition, "
             "reads what people say about it and writes you a personalised report.")
    st.markdown("### How the course modules power each feature")
    st.table(pd.DataFrame({
        "Module": ["Module 1", "Module 2", "Module 3", "Module 4", "Module 5"],
        "Theory topics": ["ANN, MLP, Activations, Optimizers",
                          "Backpropagation, Regularization, Initialization",
                          "CNN, Transfer Learning, YOLO",
                          "RNN, LSTM, GRU, BiLSTM",
                          "VAE, Generative AI"],
        "NutriLens feature": ["Calorie prediction (MLP)",
                              "Dropout/L2 + He init + optimizer comparison",
                              "Food recognition (MobileNetV2) + YOLO detection",
                              "Food review sentiment (BiLSTM)",
                              "Synthetic nutrition data (VAE) + AI report"],
        "Page": ["Calorie Predictor", "Model Performance", "Food Scanner / YOLO",
                 "Review Sentiment", "Synthetic Generator / AI Report"]}))
    st.info("Use the sidebar on the left to open each feature.")


def page_scanner():
    st.header("📸 Food Scanner  (Module 3 - CNN + Transfer Learning)")
    need("models/food_cnn.keras", "train_cnn.py")
    file = st.file_uploader("Upload a food photo", type=["jpg", "jpeg", "png"])
    if not file:
        st.caption("The model knows: " + ", ".join(pretty(c) for c in NUTRITION))
        return
    img = Image.open(file).convert("RGB")
    c1, c2 = st.columns(2)
    c1.image(img, caption="Your photo", use_container_width=True)
    model = load_model("models/food_cnn.keras")
    class_names = load_json("models/class_names.json")
    arr = np.expand_dims(np.array(img.resize((224, 224)), dtype="float32"), 0)
    probs = model.predict(arr, verbose=0)[0]
    top = np.argsort(probs)[::-1][:3]
    best = class_names[int(top[0])]
    st.session_state["last_food"] = best
    with c2:
        st.success(f"Prediction: **{pretty(best)}**  ({probs[top[0]] * 100:.1f}% confident)")
        st.write("Top 3 guesses")
        st.bar_chart(pd.DataFrame({"confidence": [float(probs[i]) for i in top]},
                                  index=[pretty(class_names[int(i)]) for i in top]))
        n = NUTRITION[best]
        kcal = predict_calories(n)
        st.metric("Estimated calories (MLP)", f"{kcal:.0f} kcal")
        st.metric("Health score", f"{health_score(n)}/100")
        st.dataframe(pd.DataFrame([n]).rename(index={0: "per serving"}), use_container_width=True)
    st.caption("Nutrition values are approximate per typical serving.")


def page_calories():
    st.header("🧮 Calorie Predictor  (Module 1 + 2 - MLP)")
    st.write("Move the sliders and the neural network predicts the calories.")
    cols = st.columns(3)
    vals = {}
    ranges = {"protein": (0, 60, 20), "carbs": (0, 100, 40), "fat": (0, 60, 15),
              "fibre": (0, 20, 3), "sugar": (0, 60, 8), "sodium": (0, 1500, 500)}
    for i, f in enumerate(FEATURES):
        lo, hi, d = ranges[f]
        unit = "mg" if f == "sodium" else "g"
        vals[f] = cols[i % 3].slider(f"{f.title()} ({unit})", lo, hi, d)
    if st.button("Predict calories", type="primary"):
        pred = predict_calories(vals)
        formula = 4 * vals["protein"] + 4 * vals["carbs"] + 9 * vals["fat"]
        a, b = st.columns(2)
        a.metric("MLP prediction", f"{pred:.0f} kcal")
        b.metric("Textbook formula (4P + 4C + 9F)", f"{formula:.0f} kcal")
    st.caption("The network learned this relationship from data - it was never given the formula.")


def page_sentiment():
    st.header("💬 Review Sentiment  (Module 4 - Bidirectional LSTM)")
    need("models/sentiment_bilstm.keras", "train_lstm.py")
    examples = {"Positive example": "Absolutely delicious! Fresh ingredients, great flavour and perfect portion size.",
                "Negative example": "Terrible. The food was cold, bland and way too salty. Never ordering again."}
    choice = st.radio("Try an example or write your own", ["Write my own"] + list(examples), horizontal=True)
    default = examples.get(choice, "")
    text = st.text_area("Food review", value=default, height=120)
    if st.button("Analyse sentiment", type="primary") and text.strip():
        model = load_model("models/sentiment_bilstm.keras")
        import tensorflow as tf
        p = float(model.predict(tf.constant([[text]]), verbose=0)[0][0])
        label = "😊 Positive" if p >= 0.5 else "😞 Negative"
        st.metric("Sentiment", label, f"{max(p, 1 - p) * 100:.1f}% confident")
        st.progress(p, text=f"Positivity score: {p:.2f}")


def page_generator():
    st.header("🧪 Synthetic Nutrition Generator  (Module 5 - VAE)")
    st.write("A Variational Autoencoder learned the patterns in nutrition data and can now invent new, realistic food profiles.")
    need("models/vae_decoder.keras", "train_vae.py")
    n = st.slider("How many synthetic foods?", 5, 200, 20)
    if st.button("Generate", type="primary"):
        decoder = load_model("models/vae_decoder.keras")
        scaler = load_scaler("models/vae_scaler.pkl")
        z = np.random.normal(size=(n, decoder.input_shape[1]))
        data = np.clip(scaler.inverse_transform(decoder.predict(z, verbose=0)), 0, None)
        df = pd.DataFrame(data, columns=FEATURES).round(1)
        df["approx_calories"] = (4 * df["protein"] + 4 * df["carbs"] + 9 * df["fat"]).round(0)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False), "synthetic_nutrition.csv", "text/csv")
    m = load_json("assets/vae_metrics.json")
    if m:
        with st.expander("How realistic is it? (real vs generated averages)"):
            st.dataframe(pd.DataFrame({"real": m["real_mean"], "generated": m["generated_mean"]}))


def page_report():
    st.header("📝 AI Nutrition Report  (Module 5 - Generative AI)")
    foods = list(NUTRITION)
    default = foods.index(st.session_state["last_food"]) if st.session_state.get("last_food") in foods else 0
    food = st.selectbox("Food", foods, index=default, format_func=pretty)
    goal = st.selectbox("Your goal", ["Balanced diet", "Weight loss", "Muscle gain"])
    if st.button("Generate report", type="primary"):
        n = NUTRITION[food]
        kcal = predict_calories(n)
        text, source = make_report(food, n, kcal, goal)
        st.markdown(text)
        st.caption(f"Generated by: {source}")


def page_yolo():
    st.header("🔍 Food Detection  (Module 3 - YOLO)")
    st.write("YOLO finds and boxes several food items in one picture (it knows apple, banana, orange, broccoli, carrot, sandwich, hot dog, pizza, donut, cake).")
    try:
        from ultralytics import YOLO
    except ImportError:
        st.warning("Optional page: run `pip install ultralytics` to enable it.")
        return
    file = st.file_uploader("Upload a photo with food items", type=["jpg", "jpeg", "png"], key="yolo")
    if file:
        img = Image.open(file).convert("RGB")
        model = YOLO("yolov8n.pt")  # downloads the small pretrained model on first use
        result = model(np.array(img))[0]
        st.image(result.plot()[:, :, ::-1], caption="Detections", use_container_width=True)
        names = [result.names[int(c)] for c in result.boxes.cls]
        st.write("Detected:", ", ".join(sorted(set(names))) if names else "nothing recognised")


def page_performance():
    st.header("📊 Model Performance")
    for title, path, img in [("CNN (Module 3)", "assets/cnn_metrics.json", "assets/cnn_accuracy.png"),
                             ("BiLSTM (Module 4)", "assets/lstm_metrics.json", "assets/lstm_accuracy.png")]:
        st.subheader(title)
        m = load_json(path)
        if m:
            st.metric("Test accuracy", f"{m['test_accuracy_percent']}%")
        if os.path.exists(img):
            st.image(img, width=550)
    st.subheader("MLP (Module 1 + 2)")
    m = load_json("assets/mlp_metrics.json")
    if m:
        a, b = st.columns(2)
        a.write("**Optimizer comparison** (test MAE, kcal - lower is better)")
        a.table(pd.Series(m["optimizers"], name="MAE"))
        b.write("**Regularization** (test MAE, kcal)")
        b.table(pd.Series(m["regularization"], name="MAE"))
    for img in ["assets/mlp_optimizers.png", "assets/mlp_regularization.png"]:
        if os.path.exists(img):
            st.image(img, width=550)
    v = load_json("assets/vae_metrics.json")
    if v:
        st.subheader("VAE (Module 5)")
        st.write(f"Final loss: {v['final_loss']}  |  latent dimensions: {v['latent_dim']}")


PAGES = {"🏠 Home": page_home, "📸 Food Scanner": page_scanner, "🧮 Calorie Predictor": page_calories,
         "💬 Review Sentiment": page_sentiment, "🧪 Synthetic Generator": page_generator,
         "📝 AI Nutrition Report": page_report, "🔍 YOLO Detection": page_yolo,
         "📊 Model Performance": page_performance}
st.sidebar.title("🥗 NutriLens AI")
choice = st.sidebar.radio("Go to", list(PAGES))
PAGES[choice]()
st.sidebar.caption("Deep Learning project - Modules 1 to 5")
