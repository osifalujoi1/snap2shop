# Snap2Shop — AI Image Classifier & Shopping Assistant

Snap2Shop is a lightweight AI-powered tool that classifies images and helps users find where to buy the item online, using real-time Google Shopping results.

---

## Demo

![Snap2Shop Demo](Snap2ShopDemo.gif) 

---

## Features

- **Image Classification** with TensorFlow's MobileNetV2
- **Online Product Search** via SerpAPI and Google Shopping
- **Responsive Web UI** with Streamlit
- **Visual Display** of product image, source, price, and ratings

---

## Tech Stack

- **Frontend:** Streamlit
- **ML Model:** MobileNetV2 (pretrained on ImageNet)
- **Data Fetching:** SerpAPI (Google Shopping)
- **Libraries:** TensorFlow, OpenCV, PIL, NumPy, Streamlit

---

## How It Works

1. Upload an image.
2. The app uses MobileNetV2 to classify the object in the image.
3. The top predicted label is used as a query to fetch product listings from online marketplaces.
4. Products are displayed with images, price, source, rating, and direct links.

---

## Installation

```bash
git clone https://github.com/osifalujoi1/snap2shop.git
cd snap2shop
pip install -r requirements.txt
streamlit run main.py
