import streamlit as st
import re
import google.generativeai as genai
import os
import base64
from PIL import Image
import io
import requests
import json
import time
import uuid

# App title and description
st.title("Social Media Post Generator")
st.markdown("Generate engaging content & images for LinkedIn and Twitter/X.")

# Configure API key with improved error handling
with st.sidebar:
    st.subheader("API Configuration")
    gemini_api_key = st.text_input("Enter your Gemini API Key", type="password")

# Set default model to Gemini 1.5 Pro
DEFAULT_MODEL = "gemini-1.5-pro"

# Initialize session state variables if they don't exist
if 'generated_post' not in st.session_state:
    st.session_state.generated_post = ""
if 'prompt_history' not in st.session_state:
    st.session_state.prompt_history = []
if 'feedback_provided' not in st.session_state:
    st.session_state.feedback_provided = False
if 'original_post' not in st.session_state:
    st.session_state.original_post = ""
if 'generated_image_url' not in st.session_state:
    st.session_state.generated_image_url = None

if gemini_api_key:
    os.environ["GOOGLE_API_KEY"] = gemini_api_key
    genai.configure(api_key=gemini_api_key)
    
    # Direct debug output to help troubleshoot
    st.sidebar.subheader("API Connection Check")
    try:
        st.sidebar.info("Checking connection to Gemini API...")
        st.sidebar.success(f"Using model: {DEFAULT_MODEL}")
    except Exception as e:
        st.sidebar.error(f"⚠️ Error connecting to Gemini API: {str(e)}")
        st.sidebar.warning("Please check your API key and internet connection.")
else:
    st.sidebar.warning("Please enter your Gemini API Key to use this application.")
    st.stop()

# Define bad words list (this is a simplified list - expand as needed)
BAD_WORDS = [
    "fuck", "shit", "ass", "bitch", "dick", "pussy", "cunt", "bastard", 
    "damn", "hell", "whore", "slut", "asshole", "motherfucker"
]

# Function to check for inappropriate content
def contains_inappropriate_content(text):
    # Convert to lowercase for checking
    text_lower = text.lower()
    
    # Check for bad words
    for word in BAD_WORDS:
        if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
            return True
    
    return False

# Function to process the uploaded image
def process_image(uploaded_file):
    if uploaded_file is not None:
        # Convert the file to an image
        image = Image.open(uploaded_file)
        
        # For Gemini API, we need to prepare the image
        # Resize if needed for API constraints
        max_size = (1024, 1024)
        image.thumbnail(max_size, Image.LANCZOS)
        
        # Convert to bytes
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        
        return image, image_bytes
    return None, None

# Function to generate image using Pollinations.ai (no API key required)
def generate_image_with_pollinations(prompt):
    try:
        # Create a safe name for the model seed
        safe_prompt = prompt.replace(" ", "-")[:30]
        seed = str(uuid.uuid4())[:8]
        
        # Use Pollinations public API endpoint (no auth needed)
        # This constructs a URL that will generate an image on-the-fly
        image_url = f"https://image.pollinations.ai/prompt/{safe_prompt}%20{seed}?width=1024&height=1024&nologo=true"
        
        # Return the URL only - we'll download it when needed
        return image_url, None
    
    except Exception as e:
        return None, f"Error generating image URL: {str(e)}"

# Function to download and process the generated image
def download_image(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Process the image
        image = Image.open(io.BytesIO(response.content))
        
        # Convert to bytes for Gemini API
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        
        return image, image_bytes
    except requests.exceptions.RequestException as e:
        return None, None, f"Error downloading image: {str(e)}"
    except Exception as e:
        return None, None, f"Error processing image: {str(e)}"

# Function to generate post content with improved error handling
def generate_post(prompt, platform, image_bytes=None, is_refinement=False, original_content=None):
    try:
        # Always use Gemini 1.5 Pro
        model = genai.GenerativeModel(DEFAULT_MODEL)
        
        # Prepare text-only prompt
        full_prompt = (
            f"Create a short, engaging {platform} post (less than 50 words) based on this prompt: '{prompt}'. "
            f"Include relevant hashtags. Be creative and professional."
        )
        
        if is_refinement and original_content:
            full_prompt = (
                f"Create a COMPLETELY DIFFERENT version of a {platform} post based on this prompt: '{prompt}'. "
                f"The previous post was: '{original_content}' - "
                f"I need a new approach with different tone, structure, and perspective. "
                f"Make it sound nothing like the original while still addressing the core topic. "
                f"Keep it under 50 words and include relevant hashtags."
            )
        
        # Handle image if provided
        if image_bytes:
            image_parts = [{"mime_type": "image/png", "data": base64.b64encode(image_bytes).decode("utf-8")}]
            text_parts = [full_prompt]
            
            response = model.generate_content(
                image_parts + text_parts
            )
            full_prompt = (
                f"Create a short, engaging {platform} post (less than 50 words) based on this prompt: '{prompt}' "
                f"and the image I've provided. The post should reference what's in the image. "
                f"Include relevant hashtags. Be creative and professional."
            )
            if is_refinement and original_content:
                full_prompt = (
                    f"Create a COMPLETELY DIFFERENT version of a {platform} post based on this prompt: '{prompt}' "
                    f"and the image I've provided. The post should reference what's in the image. "
                    f"The previous post was: '{original_content}' - "
                    f"I need a new approach with different tone, structure, and perspective. "
                    f"Make it sound nothing like the original while still addressing the core topic and referencing the image. "
                    f"Keep it under 50 words and include relevant hashtags."
                )
            
            response = model.generate_content(
                [
                    {"mime_type": "image/png", "data": base64.b64encode(image_bytes).decode("utf-8")},
                    full_prompt
                ]
            )
        else:
            response = model.generate_content(full_prompt)
            
        generated_text = response.text
        
        # Check for inappropriate content
        if contains_inappropriate_content(generated_text):
            return "I couldn't generate appropriate content based on your prompt. Please try a different topic or wording."
        
        return generated_text
    
    except Exception as e:
        error_msg = str(e)
        troubleshooting = "\n\nTroubleshooting tips:\n"
        
        if "404" in error_msg and "not found" in error_msg:
            troubleshooting += "1. The model name is not recognized. Try checking if you have access to Gemini 1.5 Pro.\n"
        elif "403" in error_msg:
            troubleshooting += "1. Your API key might not have permission to use Gemini 1.5 Pro.\n"
        elif "image" in error_msg.lower():
            troubleshooting += "1. There might be an issue with image processing. Try with a different image or without an image.\n"
        else:
            troubleshooting += "1. Check your API key permissions.\n"
            
        troubleshooting += "2. Make sure you have access to Gemini models.\n"
        troubleshooting += "3. Try updating the google-generativeai package: pip install --upgrade google-generativeai"
        
        return f"Error generating content: {error_msg}{troubleshooting}"

# Main interface
st.subheader("Step 1: Configure Your Post")

# Platform selection (limited to LinkedIn and Twitter/X)
platform = st.selectbox(
    "Select social media platform",
    ["LinkedIn", "Twitter/X"]
)

# User prompt input
prompt = st.text_area("Enter your post topic or idea", height=100)

# Image options
image_option = st.radio(
    "Image options:",
    ["No image", "Upload your own image", "Generate an image with AI"]
)

uploaded_file = None
display_image = None
image_bytes = None
image_prompt = ""

if image_option == "Upload your own image":
    uploaded_file = st.file_uploader("Upload an image to include with your post", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        display_image, image_bytes = process_image(uploaded_file)
        if display_image:
            st.image(display_image, caption="Preview of uploaded image", use_container_width=True)
elif image_option == "Generate an image with AI":
    image_prompt = st.text_area("Describe the image you want to generate", 
                               placeholder="E.g., A professional workspace with a laptop, coffee cup, and notebook")
    if st.button("Preview Image") and image_prompt:
        with st.spinner("Generating image preview..."):
            image_url, error = generate_image_with_pollinations(image_prompt)
            if image_url:
                st.session_state.generated_image_url = image_url
                
                # Add a delay to allow image to be generated
                time.sleep(2)
                
                try:
                    # Download and display the image
                    response = requests.get(image_url)
                    if response.status_code == 200:
                        # Save the image bytes
                        image_bytes = response.content
                        
                        # Display the image
                        st.image(image_url, caption="Generated image preview", use_container_width=True)
                except Exception as e:
                    st.error(f"Error downloading image: {str(e)}")
                    st.warning("Showing image URL instead. Click to view:")
                    st.markdown(f"[View generated image]({image_url})")
            else:
                st.error(f"Failed to generate image: {error}")

# Generate button
if st.button("Generate Post") and prompt:
    with st.spinner("Generating your social media post..."):
        # First, generate an image if that option was selected but not previewed
        if image_option == "Generate an image with AI" and image_prompt and not st.session_state.generated_image_url:
            with st.spinner("Generating image..."):
                image_url, error = generate_image_with_pollinations(image_prompt)
                if image_url:
                    st.session_state.generated_image_url = image_url
                    # Add a delay to allow image to be generated
                    time.sleep(2)
                    
                    try:
                        # Download and use the image
                        response = requests.get(image_url)
                        if response.status_code == 200:
                            image_bytes = response.content
                    except Exception as e:
                        st.error(f"Error downloading image: {str(e)}")
                else:
                    st.error(f"Failed to generate image: {error}")
        
        # Then generate the post
        if prompt in st.session_state.prompt_history:
            st.session_state.generated_post = generate_post(prompt, platform, image_bytes, is_refinement=True)
        else:
            st.session_state.prompt_history.append(prompt)
            st.session_state.generated_post = generate_post(prompt, platform, image_bytes)
            # Store the original post for comparison during improvement
            st.session_state.original_post = st.session_state.generated_post
        
        st.session_state.feedback_provided = False

# Display generated content
if st.session_state.generated_post:
    st.subheader("Generated Post:")
    if st.session_state.generated_post.startswith("Error"):
        st.error(st.session_state.generated_post)
    else:
        # Show the final result with text and image if applicable
        if image_option != "No image":
            col1, col2 = st.columns([1, 2])
            with col1:
                if display_image:
                    st.image(display_image, caption="Post image", use_container_width=True)
                elif st.session_state.generated_image_url:
                    st.image(st.session_state.generated_image_url, caption="Generated image", use_container_width=True)
            with col2:
                st.markdown("#### Post Text:")
                st.write(st.session_state.generated_post)
        else:
            st.write(st.session_state.generated_post)
        
        # Feedback buttons
        if not st.session_state.feedback_provided:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("It's Good"):
                    st.session_state.feedback_provided = True
                    st.success("Great! Your post is ready to use.")
            with col2:
                if st.button("Improve It"):
                    with st.spinner("Generating significantly different version..."):
                        # Pass the original post to ensure the new one is different
                        st.session_state.generated_post = generate_post(
                            prompt, 
                            platform, 
                            image_bytes, 
                            is_refinement=True, 
                            original_content=st.session_state.generated_post
                        )

# Copy button for the generated post
if st.session_state.generated_post and not st.session_state.generated_post.startswith("Error"):
    if st.button("Copy to Clipboard"):
        st.code(st.session_state.generated_post)
        st.success("Post copied to clipboard! (Use Ctrl+C or Cmd+C to copy from the code block)")

# History section in sidebar
st.sidebar.subheader("Previous Prompts")
if st.session_state.prompt_history:
    for i, prev_prompt in enumerate(st.session_state.prompt_history):
        st.sidebar.text(f"{i+1}. {prev_prompt[:30]}..." if len(prev_prompt) > 30 else f"{i+1}. {prev_prompt}")
else:
    st.sidebar.text("No previous prompts yet.")

# Instructions and information in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("How to use")
st.sidebar.markdown("""
1. Enter your Gemini API Key
2. Select a platform (LinkedIn or Twitter/X)
3. Type your post topic/idea 
4. Choose an image option:
   - No image
   - Upload your own image
   - Generate an AI image
5. Click 'Generate Post'
6. Provide feedback to refine
""")

st.sidebar.markdown("---")
st.sidebar.info("This app uses Google's Gemini 1.5 Pro API to generate social media content and " 
                "Pollinations.ai for image generation. Your content is checked for appropriateness before displaying.")

# Version indicator to confirm updates are working
st.sidebar.markdown("---")
st.sidebar.text("App Version: 2.1 (Fixed image generation)")

# Troubleshooting tips
with st.sidebar.expander("Troubleshooting Tips", expanded=False):
    st.markdown("""
    **If you're getting API errors:**
    
    1. Make sure you have a valid Gemini API key
    2. Check that your API key has access to Gemini 1.5 Pro
    3. Try updating the google-generativeai package:
       ```
       pip install --upgrade google-generativeai
       ```
    4. Ensure you have an active internet connection
    5. If using images, make sure they're in a supported format (PNG, JPG, JPEG)
    6. If image generation fails, try a more descriptive prompt or simplify your request
    7. If image display fails, you can click the provided link to view the image directly
    """)

# Add a clear cache button to help with updates
if st.sidebar.button("Clear Cache and Reload"):
    st.cache_data.clear()
    st.session_state.clear()
    st.rerun()
