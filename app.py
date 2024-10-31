import streamlit as st
import google.generativeai as genai
import google.ai.generativelanguage as glm
import torch
import numpy as np
import pickle
import io
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as albu
import base64

# Set up API key for Gemini-Pro
API_KEY = 'AIzaSyDiyi0-uvjv_8-2QrbK8BuVkY1kboCakxc'
genai.configure(api_key=API_KEY)

# Load MicroNet model from pickle
pickle_path = "micronet_model_steel_segmentation.pkl"
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: 
            return super().find_class(module, name)

with open(pickle_path, 'rb') as file:
    model = CPU_Unpickler(file, encoding='latin1').load()

# Define preprocessing for MicroNet
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    return albu.Compose([
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ])

preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet50', 'imagenet')
preprocessing_transform = get_preprocessing(preprocessing_fn)

def preprocess(img):
    temp = img / 255.0
    img = preprocessing_transform(image=temp)
    return torch.from_numpy(img['image']).to('cpu').unsqueeze(0)

def download_link(img_array, filename="segmented_image.png"):
    # Ensure the array has the correct shape and data type
    img_array = np.squeeze(img_array)
    if img_array.ndim == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)

    pil_image = Image.fromarray((img_array * 255).astype(np.uint8))
    img_bytes = io.BytesIO()
    pil_image.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    # Create download link
    b64 = base64.b64encode(img_bytes.read()).decode()
    return f'<a href="data:file/png;base64,{b64}" download="{filename}">Download Segmented Image</a>'

# Streamlit app layout
st.set_page_config(page_title="Microscopic Image Analysis with Gemini-Pro & MicroNet", layout="centered")
st.header("Microscopic Image Analysis: Gemini-Pro & MicroNet Integration")

# Image upload
uploaded_file = st.file_uploader("Choose a Microscopic Image file", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    bytes_data = uploaded_file.getvalue()

    # Text box for additional user input
    additional_info = st.text_input("Ask for additional information related to the microscopic steel image (optional):")

    # Session state to store the description text to prevent disappearing
    if "gemini_description" not in st.session_state:
        st.session_state.gemini_description = None
    
    # Generate Description button for LLM call
    if st.button("Generate Description"):
        # Initial prompt
        prompt_text = """
            Analyze the microscopic image of low carbon steel dual-phase. Based on the visual features, provide the following details:
            Phases: Identify the primary and secondary phases present (e.g., ferrite, martensite, bainite, etc.).
            Heat Treatment Process: Deduce the likely heat treatment process used to produce the observed microstructure.
            Compositions: Estimate the composition, especially the carbon content and other alloying elements.
        """

        # Update prompt with additional information if provided
        if additional_info:
            prompt_text += f"\n\nAdditional information request: {additional_info}. If this request is not related to microscopic images, do not process it."

        # Display loading message
        st.write("Generating descriptive analysis with Gemini-Pro...")
        
        # LLM call with the updated prompt
        model_gemini = genai.GenerativeModel('gemini-1.5-flash')
        response = model_gemini.generate_content(
            glm.Content(
                parts=[
                    glm.Part(text=prompt_text),
                    glm.Part(inline_data=glm.Blob(mime_type='image/jpeg', data=bytes_data)),
                ]
            )
        )
        
        # Store and display the Gemini-Pro response
        st.session_state.gemini_description = response.text

    # Display the saved Gemini-Pro analysis result
    if st.session_state.gemini_description:
        st.write("Gemini-Pro Analysis Result:")
        st.write(st.session_state.gemini_description)

    # Generate Mask button appears only after Generate Description is clicked
    if st.session_state.gemini_description and st.button("Generate Mask Image"):
        resized_image_o = image.resize((256, 256)).convert("RGB")
        resized_image = np.array(resized_image_o)

        # Generate segmented image
        segmented_tensor = model.predict(preprocess(resized_image))
        segmented_image = segmented_tensor[0].cpu().numpy().transpose(1, 2, 0)
        
        # Display segmented image
        st.image(segmented_image, caption="Segmented Image", use_column_width=True)
        
        # Option to download segmented image
        download_button = st.markdown(download_link(segmented_image), unsafe_allow_html=True)
