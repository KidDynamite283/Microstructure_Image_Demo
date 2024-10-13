import streamlit as st
import google.generativeai as genai
import google.ai.generativelanguage as glm
from PIL import Image

# Set your API key here
API_KEY = 'AIzaSyDiyi0-uvjv_8-2QrbK8BuVkY1kboCakxc'

# Configure the API with the provided key
genai.configure(api_key=API_KEY)

# Streamlit page configuration
st.set_page_config(page_title="Gemini Pro Vision Image Analysis Project", 
                   page_icon="📸", 
                   layout="centered", 
                   initial_sidebar_state='collapsed')

# Page header
st.header("Google AI Studio + Gemini Pro")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an Image file", accept_multiple_files=False, type=['jpg', 'png'])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Get the raw bytes of the uploaded image
    bytes_data = uploaded_file.getvalue()

    # Button to trigger content generation
    generate = st.button("Generate!")

    if generate:
        # Instantiate the Gemini Pro Vision model
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Create a prompt and pass the image data for description
        response = model.generate_content(
            glm.Content(
                parts=[
                    glm.Part(text="""
                        Analyze the microscopic image of low carbon steel dual-phase. Based on the visual features, provide the following details:

                        Phases: Identify the primary and secondary phases present (e.g., ferrite, martensite, bainite, etc.).
                        Heat Treatment Process: Deduce the likely heat treatment process used to produce the observed microstructure (e.g., annealing, quenching, tempering, etc.) also include heat treatment temperature value.
                        Compositions: Estimate the composition, especially the carbon content and other alloying elements if possible, based on the microstructure features.
                    """),
                    glm.Part(inline_data=glm.Blob(mime_type='image/jpeg', data=bytes_data)),
                ]
            ), 
            stream=True
        )


        # Resolve the response and display the result
        response.resolve()
        st.write(response.text)
