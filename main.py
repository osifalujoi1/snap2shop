import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image
from serpapi import GoogleSearch


def search_marketplaces(query):
    search_query = f"{query} site:amazon.com OR site:ebay.com OR site:aliexpress.com OR site:google.com/shopping"
    search_url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"
    return search_url

def load_model():
    model = MobileNetV2(weights="imagenet")
    return model

def preprocess_image(image):
    img = np.array(image) # Convert image into an array of numbers
    img = cv2.resize(img, (224, 224)) # Resize the image
    img = preprocess_input(img) # Preprocess the input for it to actually be ready
    img = np.expand_dims(img, axis=0) # Take single image and convert to a list of images [img]
    return img

def classify_image(model, image):
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        return decoded_predictions
    except Exception as e:
        st.error(f"Error classifying image: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Snap2Shop", page_icon="🖼️", layout="centered")
    col1, col2 = st.columns([1, 15])
    with col2:
        st.title(":blue[Snap2Shop] AI Image Classifier", width="stretch")

    st.markdown("<h5 style='text-align: center; color: #2C3E4C;'>Upload a photo, and let AI tell you where to shop for it.</h5>", unsafe_allow_html=True)

    @st.cache_resource
    def load_cached_model():
        return load_model()

    model = load_cached_model()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        # Store file in session_state for persistence
        st.session_state.uploaded_file = uploaded_file

        if "predictions" not in st.session_state:
            st.session_state.predictions = None
        if "top_label" not in st.session_state:
            st.session_state.top_label = None

        # Button to classify
        if st.button("Classify Image"):
            image = Image.open(uploaded_file)
            with st.spinner("Analyzing Image..."):
                predictions = classify_image(model, image)
                if predictions:
                    st.session_state.predictions = predictions
                    st.session_state.top_label = predictions[0][1]

    # Show predictions if they exist
    if st.session_state.get("predictions"):
        st.subheader("Predictions")
        for _, label, score in st.session_state.predictions:
            st.write(f"**{label}**: {score:.2%}")

    params = {
        "q": st.session_state.get("top_label"),
        "location": "United States",
        "hl": "en",
        "gl": "us",
        "google_domain": "google.com",
        "api_key": "d7dbf4df4ffd24faf99502612756533d46bed21ac95c5988759c33dda04cb003"
    }

    search = GoogleSearch(params)
    html_results = search.get_html()
    dict_results = search.get_dict()
    products = dict_results.get('immersive_products', [])


    # Show "Find Where to Buy" if classification happened
    if st.session_state.get("top_label"):
        st.subheader("Find Where to Buy")
        search_url = search_marketplaces(st.session_state.top_label)

        if st.button("Search Online Marketplaces"):
            st.markdown(
                f"[Click here to search for '{st.session_state.top_label}']({search_url})",
                unsafe_allow_html=True
            )
            # st.markdown(f"Google search results: {dict_results}")
            col1, col2, col3 = st.columns(3)
            columns = [col1, col2, col3]
            for i, product in enumerate(products):
                with columns[i%3]:
                    st.image(product.get('thumbnail'), width=150)
                    st.markdown(f"**{product.get('title')}**")
                    st.markdown(f"Source: {product.get('source')}")
                    st.markdown(f"Price: ${product.get('extracted_price')}")
                    st.badge(f"Rating: {product.get('rating')}")
                    st.markdown("---")





if __name__ == "__main__":
    main()
