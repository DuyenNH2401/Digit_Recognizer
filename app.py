import streamlit as st
import numpy as np
import torch
import cv2
from streamlit_drawable_canvas import st_canvas
from src.model import CNNModel
from src.utils import process_image, add_noise, get_gradcam, get_feature_maps, save_feedback
import pandas as pd
import altair as alt

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MNIST Digit Recognizer",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLING ---
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    h1 {
        text-align: center;
        color: #333;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    device = torch.device('cpu') # Deploy often uses CPU
    model = CNNModel()
    try:
        model.load_state_dict(torch.load('models/model.pth', map_location=device))
        model.eval()
        return model
    except FileNotFoundError:
        st.error("Model file 'models/model.pth' not found. Please verify the path.")
        return None

model = load_model()

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.markdown("### 📡 Input Mode")
    input_mode = st.selectbox("Choose Input Source", ["Draw on Canvas", "Upload Image"], index=0)
    
    st.markdown("---")
    
    if input_mode == "Draw on Canvas":
        stroke_width = st.slider("Stroke Width", 10, 50, 20)
    
    st.markdown("### 🧪 Robustness Test")
    noise_level = st.slider("🌫️ Noise Injection Level", 0.0, 0.5, 0.0, help="Add random noise to the input to test model robustness.")
    
    st.info("💡 **Tip:** Draw a digit in the center for best results.")
    st.markdown("---")
    st.markdown("Created with ❤️ by Duyen Nguyen")

# --- MAIN CONTENT ---
st.title("🧠 AI Digit Recognizer")
st.markdown("### Draw a digit (0-9) and analyze the AI's thinking process.")

col1, col2 = st.columns([1, 1.2])

with col1:
    st.markdown("#### 🎨 Input")
    
    image_data = None
    
    if input_mode == "Draw on Canvas":
        img_size = 280
        # Wrap canvas in a styled container
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=stroke_width,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=img_size,
            width=img_size,
            drawing_mode="freedraw",
            key="canvas",
        )
        if canvas_result.image_data is not None and np.sum(canvas_result.image_data) > 0:
            image_data = canvas_result.image_data
        
        if st.button("🗑️ Clear Canvas", key="clear"):
            st.rerun()
            
    elif input_mode == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image of a digit", type=["png", "jpg", "jpeg"])
        invert = st.checkbox("Invert Colors (Check this if image is Black Text on White Paper)", value=True)
        
        if uploaded_file is not None:
             # Load image
            from src.utils import load_image_from_bytes
            image_val = load_image_from_bytes(uploaded_file)
            st.image(image_val, caption="Uploaded Image", width=200)
            
            # Convert to Gray immediately for uniformity
            gray_img = cv2.cvtColor(image_val, cv2.COLOR_RGB2GRAY)
            
            if invert:
                gray_img = cv2.bitwise_not(gray_img)
                
            image_data = gray_img

with col2:
    st.markdown("#### 🤖 Results & Analysis")
    
    if image_data is not None:
        # --- PREPROCESSING & NOISE ---
        tensor_orig, processed_img = process_image(image_data)
        
        # Add Noise
        final_tensor = add_noise(tensor_orig, noise_level)
        
        # --- PREDICTION ---
        if model:
            # Predict
            model.eval()
            with torch.no_grad():
                outputs = model(final_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                prediction = torch.argmax(probabilities).item()
                confidence = probabilities[prediction].item()

            st.markdown(f"""
            <div class="metric-card">
                <h2 style='font-size: 80px; margin: 0; color: #4CAF50;'>{prediction}</h2>
                <p style='font-size: 24px; color: #666;'>Confidence: <b>{confidence*100:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True)

            # --- VISUALIZATION TABS ---
            tab1, tab2, tab3 = st.tabs(["📊 Prediction", "🧠 Explainability", "🔁 Feedback"])
            
            with tab1:
                st.markdown("##### Confidence Breakdown")
                probs_np = probabilities.numpy()
                df = pd.DataFrame({'Digit': range(10), 'Probability': probs_np})
                chart = alt.Chart(df).mark_bar().encode(
                    x=alt.X('Digit:O', title='Digit'),
                    y=alt.Y('Probability:Q', title='Confidence Score', scale=alt.Scale(domain=[0, 1])),
                    color=alt.condition(alt.datum.Digit == prediction, alt.value('#4CAF50'), alt.value('#e0e0e0')),
                    tooltip=['Digit', alt.Tooltip('Probability', format='.1%')]
                ).properties(height=200)
                st.altair_chart(chart, use_container_width=True)

            with tab2:
                # 1. Inputs
                st.markdown("##### 1. Model Inputs")
                c1, c2 = st.columns(2)
                with c1:
                    st.image((processed_img*255).astype(np.uint8), caption="Smart Preprocessed")
                with c2:
                    if noise_level > 0:
                        noisy_display = final_tensor.squeeze().numpy()
                        st.image((noisy_display*255).astype(np.uint8), caption=f"Noisy Input (Lvl {noise_level})")
                    else:
                        st.markdown("*No Noise*")

                st.markdown("---")
                # 2. Grad-CAM
                st.markdown("##### 2. Grad-CAM (Attention Map)")
                st.caption("Warm colors (Red/Yellow) show where the model is looking.")
                
                try:
                    # Grad-CAM requires gradient, so we must allow it locally
                    with torch.set_grad_enabled(True):
                        # We need a new tensor that requires grad for the cam computation flow interaction (or just model weights)
                        # Actually getting gradcam usually needs the graph. 
                        # We re-feed the tensor.
                        cam_input = final_tensor.clone().requires_grad_(True)
                        heatmap = get_gradcam(model, cam_input, prediction)
                        
                        # Apply colormap
                        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
                        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                        
                        # Blend
                        processed_rgb = (np.stack((processed_img,)*3, axis=-1) * 255).astype(np.uint8)
                        superimposed = cv2.addWeighted(processed_rgb, 0.6, heatmap_color, 0.4, 0)
                        
                        st.image(superimposed, caption=f"Attention for Class '{prediction}'", width=150)
                except Exception as e:
                    st.error(f"Could not generate Grad-CAM: {e}")

                st.markdown("---")
                # 3. Feature Maps
                st.markdown("##### 3. Feature Maps (Conv1)")
                st.caption("Total 32 filters. Showing first 8.")
                fmaps = get_feature_maps(model, 'conv1', final_tensor)
                if fmaps is not None:
                    # fmaps: (1, 32, 26, 26)
                    cols = st.columns(4)
                    for i in range(8):
                        fmap = fmaps[0, i].cpu().numpy()
                        fmap_norm = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
                        with cols[i % 4]:
                            st.image(fmap_norm, clamp=True)
                            
            with tab3:
                st.markdown("##### 🔁 Active Learning Loop")
                st.write("Is the prediction correct?")
                
                col_y, col_n = st.columns(2)
                if col_y.button("✅ Yes, Correct"):
                    st.toast("Thank you! Feedback recorded.", icon="🎉")
                    
                if col_n.button("❌ No, Incorrect"):
                    st.session_state['show_correction'] = True

                if st.session_state.get('show_correction'):
                    st.markdown("Please select the **correct label**:")
                    correct_label = st.selectbox("True Digit", range(10), key="correct_val")
                    
                    if st.button("💾 Save to Database"):
                        save_feedback(processed_img, correct_label, prediction)
                        st.success(f"Saved! Image marked as '{correct_label}'.")
                        st.session_state['show_correction'] = False

    else:
        st.info("Draw on the left to start.")
    
# --- FOOTER ---
st.markdown("---")