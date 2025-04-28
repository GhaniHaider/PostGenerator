import streamlit as st
import re
import google.generativeai as genai
import os

# App title and description
st.title("Social Media Post Generator")
st.markdown("Generate engaging content for LinkedIn and Twitter/X.")

# Configure API key with improved error handling
api_key = st.sidebar.text_input("Enter your Gemini API Key", type="password")

# Set default model to Gemini 1.5 Pro
DEFAULT_MODEL = "gemini-1.5-pro"

# Initialize session state variables if they don't exist
if 'generated_post' not in st.session_state:
    st.session_state.generated_post = ""
if 'prompt_history' not in st.session_state:
    st.session_state.prompt_history = []
if 'feedback_provided' not in st.session_state:
    st.session_state.feedback_provided = False

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)
    
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

# Function to generate post content with improved error handling
def generate_post(prompt, platform, is_refinement=False):
    try:
        # Always use Gemini 1.5 Pro
        model = genai.GenerativeModel(DEFAULT_MODEL)
        
        # Prepare text-only prompt
        full_prompt = (
            f"Create a short, engaging {platform} post (less than 50 words) based on this prompt: '{prompt}'. "
            f"Include relevant hashtags. Be creative and professional."
        )
        
        if is_refinement:
            full_prompt += " This is a refinement of a previous generation that wasn't satisfactory, so make it significantly different."
        
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
        else:
            troubleshooting += "1. Check your API key permissions.\n"
            
        troubleshooting += "2. Make sure you have access to Gemini models.\n"
        troubleshooting += "3. Try updating the google-generativeai package: pip install --upgrade google-generativeai"
        
        return f"Error generating content: {error_msg}{troubleshooting}"

# Platform selection (limited to LinkedIn and Twitter/X)
platform = st.selectbox(
    "Select social media platform",
    ["LinkedIn", "Twitter/X"]
)

# User prompt input
prompt = st.text_area("Enter your post topic or idea", height=100)

# Generate button
if st.button("Generate Post") and prompt:
    with st.spinner("Generating your social media post..."):
        if prompt in st.session_state.prompt_history:
            st.session_state.generated_post = generate_post(prompt, platform, is_refinement=True)
        else:
            st.session_state.prompt_history.append(prompt)
            st.session_state.generated_post = generate_post(prompt, platform)
        
        st.session_state.feedback_provided = False

# Display generated content
if st.session_state.generated_post:
    st.subheader("Generated Post:")
    if st.session_state.generated_post.startswith("Error"):
        st.error(st.session_state.generated_post)
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
                    with st.spinner("Generating improved version..."):
                        st.session_state.generated_post = generate_post(prompt, platform, is_refinement=True)

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
1. Enter your Gemini API key
2. Select a platform (LinkedIn or Twitter/X)
3. Type your post topic/idea
4. Click 'Generate Post'
5. Provide feedback to refine
""")

st.sidebar.markdown("---")
st.sidebar.info("This app uses Google's Gemini 1.5 Pro API to generate social media content. " 
                "Your content is checked for appropriateness before displaying.")

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
    """)
