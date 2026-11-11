import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="My AI App")
st.title("SENTIMENT AND WORD GUESSING APP")


# Load each model only once, then keep it in memory (this saves time and memory)
@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )


@st.cache_resource
def load_fill_mask_model():
    return pipeline("fill-mask", model="distilbert-base-uncased")


# The special symbol that hides a word (it is [MASK] for this model)
MASK = "[MASK]"

tab1, tab2 = st.tabs(["Sentiment", "Missing word"])

# Tab 1: is the sentence positive or negative?
with tab1:
    st.write("Type a sentence. The model will say if it is POSITIVE or NEGATIVE.")
    text = st.text_area("Your sentence", "I love learning about artificial intelligence.")

    if st.button("Check sentiment"):
        if not text.strip():
            st.warning("Please type a sentence first.")
        else:
            with st.spinner("Thinking..."):
                result = load_sentiment_model()(text, truncation=True)[0]
            message = f"{result['label']} (score: {round(result['score'], 3)})"
            if result["label"] == "POSITIVE":
                st.success(message)
            else:
                st.error(message)

# Tab 2: guess the hidden word
with tab2:
    st.write(f"Type a sentence and use {MASK} where a word is missing.")
    sentence = st.text_area("Your sentence", f"We are about {MASK} go home.")

    if st.button("Guess the word"):
        if sentence.count(MASK) != 1:
            st.warning(f"Please use {MASK} exactly one time in your sentence.")
        else:
            with st.spinner("Thinking..."):
                guesses = load_fill_mask_model()(sentence, top_k=5)
            st.write("Top 5 guesses:")
            for guess in guesses:
                st.write(f"**{guess['token_str']}**  (score: {round(guess['score'], 3)})")
                st.progress(min(float(guess["score"]), 1.0))
