import streamlit as str
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# 1. Page Configuration
str.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="📸",
    layout="centered"
)

# 2. Cache the Model Loading
# This ensures the model only loads into memory once, making subsequent captions instant.
@str.cache_resource
def load_model():
    # Use CPU or GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the pre-trained BLIP processor and model from Hugging Face
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    
    return processor, model, device

# Initialize model and processor
with str.spinner("Loading AI Model into memory... please wait..."):
    processor, model, device = load_model()

# 3. User Interface Layout
str.title("📸 AI Image Caption Generator")
str.write("Upload an image, and a pre-trained BLIP vision-language model will describe it for you.")

str.divider()

# File uploader widget widget accepting jpg, jpeg, and png
uploaded_file = str.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    str.image(image, caption="Uploaded Image", use_column_width=True)
    
    str.divider()
    
    # Action button to trigger description generation
    if str.button("✨ Generate Caption", type="primary"):
        with str.spinner("Analyzing image features and generating context..."):
            try:
                # Preprocess the image and move tensors to the correct device (CPU/GPU)
                inputs = processor(image, return_tensors="pt").to(device)
                
                # Generate text tokens using greedy search optimization
                outputs = model.generate(**inputs, max_new_tokens=50)
                
                # Decode tokens back into readable string, ignoring special tokens like padding
                caption = processor.decode(outputs[0], skip_special_tokens=True)
                
                # Display the finalized output
                str.success("### Generated Caption:")
                str.info(f"**{caption.capitalize()}**")
                
            except Exception as e:
                str.error(f"An error occurred during inference: {e}")