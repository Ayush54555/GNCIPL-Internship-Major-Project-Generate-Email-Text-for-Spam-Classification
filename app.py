import streamlit as st
import pickle
import numpy as np

CUSTOM_THRESHOLD = 0.75

# --- 1. CONFIGURATION ---
MODEL_PATH = "Model/best_model_svc.pkl"
VECTORIZER_PATH = "Model/tfidf_vectorizer.pkl"

# --- 2. LOAD MODEL AND VECTORIZER (CACHE) ---
@st.cache_resource
def load_assets(model_path, vectorizer_path):
    """Loads the pickled model and vectorizer."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)

        return model, vectorizer
    except FileNotFoundError:
        st.error("Error: Model or Vectorizer file not found. Please ensure 'best_model_svc.pkl' and 'tfidf_vectorizer.pkl' are in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        st.stop()

model, vectorizer = load_assets(MODEL_PATH, VECTORIZER_PATH)


# --- 3. PREDICTION FUNCTION ---
def predict_email(text: str):
    """Predicts if an email is spam or ham."""
    if not text.strip():
        return None, 0.0
    text_vectorized = vectorizer.transform([text])
    # The model prediction is not directly used, as we rely on the decision score and custom threshold.
    # prediction_label = model.predict(text_vectorized)[0] 
    decision_score = model.decision_function(text_vectorized)[0]

    return "prediction", decision_score # Returning a placeholder for label as it's not used.


# --- 4. STREAMLIT APP INTERFACE ---
st.set_page_config(page_title="Spam Classifier Deployment", layout="centered")
st.title("📧 Email Spam Classifier")
st.markdown("### Deployment of the Final SVC Model")

email_input = st.text_area(
    "Paste the email content below for classification:",
    height=250,
    placeholder="Example: Congratulations! You've won a FREE prize! Click NOW to claim."
)

if st.button("Classify Email"):
    if not email_input.strip():
        st.warning("Please enter some email text to classify.")
    else:
        # Get prediction and score
        _, score = predict_email(email_input)

        # 1. Convert decision score to a confidence probability using the sigmoid function
        confidence = 1 / (1 + np.exp(-score))

        # 2. Use the custom threshold for the final classification decision
        if confidence >= CUSTOM_THRESHOLD:
            label = "SPAM"
            emoji = "🚫"
            final_prediction_text = f"Classified as SPAM (Score >= {CUSTOM_THRESHOLD})"
        else:
            label = "HAM (Not Spam)"
            emoji = "✅"
            final_prediction_text = f"Classified as HAM (Score < {CUSTOM_THRESHOLD})"

        # --- Display Results ---
        st.markdown("---")
        st.subheader(f"{emoji} {final_prediction_text}")

        st.markdown(
            f'**Classification:** {label}',
            unsafe_allow_html=True
        )

        st.info(
            f"**Model Confidence Score:** {confidence:.2f} (Decision Threshold is set at {CUSTOM_THRESHOLD})"
        )
