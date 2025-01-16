# Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt

# Loading the Model
model = load_model('model.h5')

# Name of Classes
CLASS_NAMES = ('Tomato-Bacterial_spot', 'Potato-Barly blight', 'Corn-Common_rust')

# Customizing the app
st.set_page_config(
    page_title="🌱 Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Branding and Title
st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 3em;
        color: #2ECC71;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2em;
        color: #555;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("<h1 class='title'>🌱 Plant Disease Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Empowering Farmers with AI to Diagnose Plant Health</p>", unsafe_allow_html=True)

# Sidebar with "How it Works"
st.sidebar.title("How it Works")
st.sidebar.markdown("""
1. Upload an image of a plant leaf (jpg format).
2. AI model will analyze the image.
3. Get the disease name and confidence level.
4. View suggested treatments to protect your crops.
""")

# File uploader
plant_image = st.file_uploader(
    "Upload a plant leaf image (JPG only):",
    type="jpg"
)

# Predict Button
submit = st.button('🔍 Analyze Leaf')

# Predict when the button is clicked
if submit:
    if plant_image is not None:
        # Convert to OpenCV image
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        
        # Display Uploaded Image
        st.image(opencv_image, caption="Uploaded Image", use_column_width=True)

        # Preprocess Image for Model
        opencv_image = cv2.resize(opencv_image, (256, 256))
        opencv_image.shape = (1, 256, 256, 3)
        
        # Model Prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]
        confidence = np.max(Y_pred) * 100

        # Display Results with Confidence
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<h2 style='color: #28B463;'>Disease Detected:</h2>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color: #2874A6;'>{result.split('-')[0]}</h3>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<h2 style='color: #28B463;'>Confidence Level:</h2>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color: #2874A6;'>{confidence:.2f}%</h3>", unsafe_allow_html=True)

        # Visualization of Confidence Levels
        st.markdown("<h3 style='text-align: center;'>Class Confidence Levels</h3>", unsafe_allow_html=True)
        fig, ax = plt.subplots()
        ax.bar(CLASS_NAMES, Y_pred[0], color=['#E74C3C', '#F1C40F', '#1ABC9C'])
        ax.set_ylabel('Confidence')
        ax.set_xlabel('Classes')
        ax.set_title('Prediction Confidence')
        st.pyplot(fig)

        # Treatment Suggestions
        st.markdown("<h3 style='color: #16A085;'>Suggested Treatments:</h3>", unsafe_allow_html=True)
        st.markdown("""
        - **Tomato-Bacterial Spot**: Use copper-based fungicides.
        - **Potato-Barly Blight**: Ensure proper crop rotation and use fungicides.
        - **Corn-Common Rust**: Apply resistant seed varieties and fungicides.
        """)
    else:
        st.error("Please upload an image to analyze!")
else:
    st.info("Upload a plant leaf image and click 'Analyze Leaf'.")

# Footer
st.markdown("""
    <div style="text-align: center; margin-top: 50px; font-size: 16px; color: #555;">
        🌾 <b>Our Mission:</b> To empower farmers with the tools they need to protect crops, secure livelihoods, and nurture the world.<br>
        🌟 With every leaf analyzed, we bring farmers closer to healthier fields, abundant harvests, and a brighter future.<br>
        🤝 <b>Together, let's grow hope:</b> Combining the power of AI with the resilience of farmers to create a sustainable tomorrow.<br>
        ❤️ Built for the hands that feed us, with a commitment to innovation and care.<br><br>
        <b>🌍 Changing Agriculture, One Leaf at a Time.</b>
    </div>
""", unsafe_allow_html=True)

