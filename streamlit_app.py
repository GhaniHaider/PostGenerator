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
from datetime import datetime
import traceback

# Enable debugging
st.set_option('deprecation.showfileUploaderEncoding', False)

# App title and description
st.title("Multi-Agent Social Media Post Generator")
st.markdown("Generate engaging content & images with AI agents working together.")

# Configure API key with improved error handling
with st.sidebar:
    st.subheader("API Configuration")
    gemini_api_key = st.text_input("Enter your Gemini API Key", type="password")
    st.session_state['api_key_entered'] = bool(gemini_api_key)

# Set default model to Gemini 1.5 Pro
DEFAULT_MODEL = "gemini-1.5-pro"

# Initialize session state variables if they don't exist
if 'generated_post' not in st.session_state:
    st.session_state.generated_post = ""
if 'generated_image_prompt' not in st.session_state:
    st.session_state.generated_image_prompt = ""
if 'prompt_history' not in st.session_state:
    st.session_state.prompt_history = []
if 'feedback_provided' not in st.session_state:
    st.session_state.feedback_provided = False
if 'original_post' not in st.session_state:
    st.session_state.original_post = ""
if 'generated_image_url' not in st.session_state:
    st.session_state.generated_image_url = None
if 'image_analysis' not in st.session_state:
    st.session_state.image_analysis = None
if 'agent_conversation' not in st.session_state:
    st.session_state.agent_conversation = []
if 'final_post' not in st.session_state:
    st.session_state.final_post = ""
if 'show_agent_convo' not in st.session_state:
    st.session_state.show_agent_convo = False
if 'debug_info' not in st.session_state:
    st.session_state.debug_info = []

# Improved API key handling
if gemini_api_key:
    os.environ["GOOGLE_API_KEY"] = gemini_api_key
    genai.configure(api_key=gemini_api_key)
    
    # Direct debug output to help troubleshoot
    st.sidebar.subheader("API Connection Check")
    try:
        # Test the API connection
        model = genai.GenerativeModel(DEFAULT_MODEL)
        test_response = model.generate_content("Hello")
        st.sidebar.success(f"✅ Connected to Gemini API successfully")
        st.sidebar.success(f"✅ Using model: {DEFAULT_MODEL}")
    except Exception as e:
        st.sidebar.error(f"⚠️ Error connecting to Gemini API: {str(e)}")
        st.sidebar.warning("Please check your API key and internet connection.")
        # Add detailed error to debug info
        st.session_state.debug_info.append(f"API Error: {str(e)}")
        st.session_state.debug_info.append(traceback.format_exc())
else:
    st.sidebar.warning("Please enter your Gemini API Key to use this application.")

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
        try:
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
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.session_state.debug_info.append(f"Image Processing Error: {str(e)}")
            st.session_state.debug_info.append(traceback.format_exc())
    return None, None

# ================== TEXT AGENT FUNCTIONS ==================

class TextGenerationAgent:
    def __init__(self, model_name=DEFAULT_MODEL):
        try:
            self.model = genai.GenerativeModel(model_name)
            self.name = "TextAgent"
        except Exception as e:
            st.error(f"Error initializing TextGenerationAgent: {str(e)}")
            st.session_state.debug_info.append(f"TextGenerationAgent Init Error: {str(e)}")
            st.session_state.debug_info.append(traceback.format_exc())
        
    def generate_initial_post(self, prompt, platform):
        """Generate initial post based on user prompt"""
        try:
            full_prompt = (
                f"Create a short, engaging {platform} post (less than 50 words) based on this prompt: '{prompt}'. "
                f"Include relevant hashtags. Be creative and professional."
            )
            
            st.session_state.debug_info.append(f"Sending prompt to TextAgent: {full_prompt[:100]}...")
            
            response = self.model.generate_content(full_prompt)
            generated_text = response.text
            
            if contains_inappropriate_content(generated_text):
                return "I couldn't generate appropriate content based on your prompt. Please try a different topic or wording."
            
            # Add to agent conversation
            message = {"agent": self.name, "message": f"Initial post generated: {generated_text}", "time": get_timestamp()}
            st.session_state.agent_conversation.append(message)
            
            return generated_text
            
        except Exception as e:
            error_msg = str(e)
            st.session_state.debug_info.append(f"TextAgent Generate Post Error: {error_msg}")
            st.session_state.debug_info.append(traceback.format_exc())
            return f"Error generating content: {error_msg}"
    
    def extract_visual_elements(self, text):
        """Extract key visual elements from generated text to guide image creation"""
        try:
            prompt = (
                f"Analyze this social media post text and extract 3-5 key visual elements or concepts "
                f"that would make for a compelling accompanying image. Format your response as a JSON array of strings. "
                f"The post text is: '{text}'"
            )
            
            response = self.model.generate_content(prompt)
            result = response.text
            
            # Try to extract JSON if present
            try:
                # Find JSON array in the response
                match = re.search(r'\[(.*?)\]', result.replace('\n', ''), re.DOTALL)
                if match:
                    json_str = match.group(0)
                    visual_elements = json.loads(json_str)
                else:
                    # Fallback if JSON not found
                    visual_elements = [text.split()[:3]]
            except json.JSONDecodeError:
                # If JSON parsing fails, extract keywords
                visual_elements = result.split(',')[:5]
            
            # Format as a comma-separated list
            if isinstance(visual_elements, list):
                visual_elements_str = ", ".join(visual_elements)
            else:
                visual_elements_str = str(visual_elements)
                
            # Add to agent conversation
            message = {"agent": self.name, "message": f"Extracted visual elements: {visual_elements_str}", "time": get_timestamp()}
            st.session_state.agent_conversation.append(message)
            
            return visual_elements_str
            
        except Exception as e:
            error_msg = str(e)
            st.session_state.debug_info.append(f"TextAgent Extract Elements Error: {error_msg}")
            st.session_state.debug_info.append(traceback.format_exc())
            return f"Error extracting visual elements: {error_msg}"
    
    def create_image_prompt(self, user_prompt, visual_elements):
        """Create a detailed image prompt based on visual elements and user prompt"""
        try:
            prompt = (
                f"Create a detailed image generation prompt based on these elements:\n"
                f"1. User's original idea: '{user_prompt}'\n"
                f"2. Key visual elements: {visual_elements}\n\n"
                f"The prompt should be vivid, specific, and optimized for AI image generation. "
                f"Include details about style, lighting, composition, and mood. "
                f"Keep it under 100 words."
            )
            
            response = self.model.generate_content(prompt)
            image_prompt = response.text.strip()
            
            # Add to agent conversation
            message = {"agent": self.name, "message": f"Created image prompt: {image_prompt}", "time": get_timestamp()}
            st.session_state.agent_conversation.append(message)
            
            return image_prompt
            
        except Exception as e:
            error_msg = str(e)
            st.session_state.debug_info.append(f"TextAgent Image Prompt Error: {error_msg}")
            st.session_state.debug_info.append(traceback.format_exc())
            return f"Error creating image prompt: {error_msg}"

    def refine_post_with_image_feedback(self, original_post, image_analysis, platform):
        """Refine the post based on image analysis feedback"""
        try:
            prompt = (
                f"Refine this {platform} post to better match the image that was generated:\n\n"
                f"Original post: '{original_post}'\n\n"
                f"Image analysis: {image_analysis}\n\n"
                f"Create an improved version that references specific elements from the image "
                f"while maintaining the core message. Keep the post under 50 words "
                f"and include relevant hashtags. Make it feel like a cohesive package with the image."
            )
            
            response = self.model.generate_content(prompt)
            refined_post = response.text.strip()
            
            # Add to agent conversation
            message = {"agent": self.name, "message": f"Refined post based on image feedback: {refined_post}", "time": get_timestamp()}
            st.session_state.agent_conversation.append(message)
            
            return refined_post
            
        except Exception as e:
            error_msg = str(e)
            st.session_state.debug_info.append(f"TextAgent Refine Post Error: {error_msg}")
            st.session_state.debug_info.append(traceback.format_exc())
            return f"Error refining post: {error_msg}"

# ================== IMAGE AGENT FUNCTIONS ==================

class ImageGenerationAgent:
    def __init__(self):
        self.name = "ImageAgent"
    
    def generate_image(self, image_prompt):
        """Generate image using Pollinations API based on the enhanced prompt"""
        try:
            # Create a safe name for the model seed
            safe_prompt = image_prompt.replace(" ", "-")[:50]
            seed = str(uuid.uuid4())[:8]
            
            # Use Pollinations public API endpoint (no auth needed)
            image_url = f"https://image.pollinations.ai/prompt/{safe_prompt}%20{seed}?width=1024&height=1024&nologo=true"
            
            # Test if the URL is accessible
            test_response = requests.head(image_url)
            test_response.raise_for_status()
            
            # Add to agent conversation
            message = {"agent": self.name, "message": f"Generated image using prompt: {image_prompt}", "time": get_timestamp()}
            st.session_state.agent_conversation.append(message)
            
            return image_url, None
        
        except Exception as e:
            error_msg = str(e)
            st.session_state.debug_info.append(f"ImageAgent Generate Error: {error_msg}")
            st.session_state.debug_info.append(traceback.format_exc())
            return None, f"Error generating image: {error_msg}"
    
    def analyze_generated_image(self, image_url, text_agent):
        """Analyze the generated image and provide feedback"""
        try:
            # Since we can't analyze the image directly with Gemini without downloading it,
            # we'll use the image generation prompt as a basis for simulated analysis
            
            prompt = (
                f"Imagine you're analyzing an AI-generated image created with this prompt: '{st.session_state.generated_image_prompt}'\n\n"
                f"Write a brief analysis of what the image likely contains and how well it might complement a social media post. "
                f"Focus on visual elements, mood, colors, and composition. Keep it under 75 words."
            )
            
            response = text_agent.model.generate_content(prompt)
            analysis = response.text.strip()
            
            # Add to agent conversation
            message = {"agent": self.name, "message": f"Image analysis: {analysis}", "time": get_timestamp()}
            st.session_state.agent_conversation.append(message)
            
            return analysis
            
        except Exception as e:
            error_msg = str(e)
            st.session_state.debug_info.append(f"ImageAgent Analysis Error: {error_msg}")
            st.session_state.debug_info.append(traceback.format_exc())
            return f"Error analyzing image: {error_msg}"

# ================== COLLABORATION AGENT ==================

class CollaborationAgent:
    def __init__(self, model_name=DEFAULT_MODEL):
        try:
            self.model = genai.GenerativeModel(model_name)
            self.name = "CollaborationAgent"
        except Exception as e:
            st.error(f"Error initializing CollaborationAgent: {str(e)}")
            st.session_state.debug_info.append(f"CollaborationAgent Init Error: {str(e)}")
            st.session_state.debug_info.append(traceback.format_exc())
    
    def finalize_content(self, original_prompt, post_text, image_prompt, image_analysis):
        """Create a final recommendation based on all components"""
        try:
            prompt = (
                f"Review and optimize this social media content package:\n\n"
                f"Original user request: '{original_prompt}'\n\n"
                f"Generated post: '{post_text}'\n\n"
                f"Image prompt used: '{image_prompt}'\n\n"
                f"Image analysis: {image_analysis}\n\n"
                f"As a collaboration agent, suggest 1-2 minor tweaks to make the text and image work better together. "
                f"Then provide a final recommendation for the post. Keep your response short and focused."
            )
            
            response = self.model.generate_content(prompt)
            recommendation = response.text.strip()
            
            # Add to agent conversation
            message = {"agent": self.name, "message": f"Final recommendation: {recommendation}", "time": get_timestamp()}
            st.session_state.agent_conversation.append(message)
            
            # Extract the recommended final post
            # Look for the post text in the recommendation
            post_pattern = r"(?:Final Post:|Recommended Post:|Here's the final post:|Finalized Post:)(.*?)(?:\n\n|$)"
            match = re.search(post_pattern, recommendation, re.DOTALL | re.IGNORECASE)
            
            if match:
                final_post = match.group(1).strip()
            else:
                # If no specific section found, use original post
                final_post = post_text
                
            return recommendation, final_post
            
        except Exception as e:
            error_msg = str(e)
            st.session_state.debug_info.append(f"CollaborationAgent Finalize Error: {error_msg}")
            st.session_state.debug_info.append(traceback.format_exc())
            return f"Error finalizing content: {error_msg}", post_text

# Helper function to get timestamp
def get_timestamp():
    return datetime.now().strftime("%H:%M:%S")

# Function to download and process the generated image
def download_image(url):
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Process the image
        image = Image.open(io.BytesIO(response.content))
        
        # Convert to bytes for Gemini API
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        
        return image, image_bytes, None
    except requests.exceptions.RequestException as e:
        st.session_state.debug_info.append(f"Image Download Error: {str(e)}")
        st.session_state.debug_info.append(traceback.format_exc())
        return None, None, f"Error downloading image: {str(e)}"
    except Exception as e:
        st.session_state.debug_info.append(f"Image Processing Error: {str(e)}")
        st.session_state.debug_info.append(traceback.format_exc())
        return None, None, f"Error processing image: {str(e)}"

# Main interface
st.subheader("Step 1: Configure Your Post")

# Platform selection (limited to LinkedIn and Twitter/X)
platform = st.selectbox(
    "Select social media platform",
    ["LinkedIn", "Twitter/X"]
)

# User prompt input
prompt = st.text_area("Enter your post topic or idea", height=100)

# Generate button
col1, col2 = st.columns([1, 3])
with col1:
    generate_button = st.button("Generate Content Package")

if generate_button and prompt:
    if not gemini_api_key:
        st.error("❌ Please enter your Gemini API Key in the sidebar first.")
    else:
        # Store prompt in history
        if prompt not in st.session_state.prompt_history:
            st.session_state.prompt_history.append(prompt)
        
        # Clear previous session state for content
        st.session_state.agent_conversation = []
        st.session_state.generated_post = ""
        st.session_state.generated_image_prompt = ""
        st.session_state.generated_image_url = None
        st.session_state.image_analysis = None
        st.session_state.final_post = ""
        st.session_state.debug_info = []
        
        st.session_state.debug_info.append(f"Starting generation with prompt: {prompt}")
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Initialize agents
            status_text.text("Initializing agents...")
            progress_bar.progress(10)
            
            text_agent = TextGenerationAgent()
            image_agent = ImageGenerationAgent()
            collaboration_agent = CollaborationAgent()
            
            # Record user prompt
            message = {"agent": "User", "message": f"Request: {prompt}", "time": get_timestamp()}
            st.session_state.agent_conversation.append(message)
            
            # Step 2: Generate initial post with text agent
            status_text.text("Generating initial post...")
            progress_bar.progress(20)
            
            st.session_state.generated_post = text_agent.generate_initial_post(prompt, platform)
            
            if not st.session_state.generated_post.startswith("Error"):
                # Step 3: Extract visual elements from the post
                status_text.text("Extracting visual elements...")
                progress_bar.progress(30)
                
                visual_elements = text_agent.extract_visual_elements(st.session_state.generated_post)
                
                # Step 4: Create enhanced image prompt
                status_text.text("Creating image prompt...")
                progress_bar.progress(40)
                
                st.session_state.generated_image_prompt = text_agent.create_image_prompt(prompt, visual_elements)
                
                # Step 5: Generate image with the image agent
                status_text.text("Generating image...")
                progress_bar.progress(50)
                
                image_url, error = image_agent.generate_image(st.session_state.generated_image_prompt)
                if image_url:
                    st.session_state.generated_image_url = image_url
                    
                    # Step 6: Analyze the generated image
                    status_text.text("Analyzing the image...")
                    progress_bar.progress(60)
                    
                    st.session_state.image_analysis = image_agent.analyze_generated_image(image_url, text_agent)
                    
                    # Step 7: Refine the post based on image analysis
                    status_text.text("Refining the post...")
                    progress_bar.progress(70)
                    
                    refined_post = text_agent.refine_post_with_image_feedback(
                        st.session_state.generated_post,
                        st.session_state.image_analysis,
                        platform
                    )
                    
                    # Step 8: Collaborative agent finalizes the content package
                    status_text.text("Finalizing content package...")
                    progress_bar.progress(90)
                    
                    recommendation, final_post = collaboration_agent.finalize_content(
                        prompt,
                        refined_post,
                        st.session_state.generated_image_prompt,
                        st.session_state.image_analysis
                    )
                    
                    st.session_state.final_post = final_post
                    
                    # Complete
                    status_text.text("Content package generated successfully!")
                    progress_bar.progress(100)
                else:
                    status_text.text("Failed to generate image.")
                    st.error(f"Failed to generate image: {error}")
                    progress_bar.progress(100)
            else:
                status_text.text("Failed to generate initial post.")
                st.error(st.session_state.generated_post)
                progress_bar.progress(100)
                
        except Exception as e:
            st.error(f"An error occurred during content generation: {str(e)}")
            st.session_state.debug_info.append(f"Main Process Error: {str(e)}")
            st.session_state.debug_info.append(traceback.format_exc())
            progress_bar.progress(100)
            status_text.text("Generation failed. Check debug info for details.")

# Display generated content
if st.session_state.final_post:
    st.subheader("Generated Content Package:")
    
    # Display image and text together
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.session_state.generated_image_url:
            try:
                st.image(st.session_state.generated_image_url, caption="Generated image", use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying image: {str(e)}")
                st.session_state.debug_info.append(f"Image Display Error: {str(e)}")
    
    with col2:
        st.markdown("#### Final Post Text:")
        st.write(st.session_state.final_post)
    
    # Copy button for the final post
    if st.button("Copy to Clipboard"):
        st.code(st.session_state.final_post)
        st.success("Post copied to clipboard! (Use Ctrl+C or Cmd+C to copy from the code block)")

    # Toggle to view agent conversation
    if st.checkbox("Show Agent Conversation", value=st.session_state.show_agent_convo):
        st.session_state.show_agent_convo = True
        st.subheader("Agent Collaboration Process:")
        
        for message in st.session_state.agent_conversation:
            with st.expander(f"[{message['time']}] {message['agent']}"):
                st.write(message['message'])
    else:
        st.session_state.show_agent_convo = False

# Debug information section - very useful for troubleshooting
with st.expander("Debug Information", expanded=False):
    st.subheader("Debug Log")
    if st.session_state.debug_info:
        for i, debug_msg in enumerate(st.session_state.debug_info):
            st.text(f"{i+1}. {debug_msg}")
    else:
        st.text("No debug information available.")

# History section in sidebar
st.sidebar.subheader("Previous Prompts")
if st.session_state.prompt_history:
    for i, prev_prompt in enumerate(st.session_state.prompt_history):
        st.sidebar.text(f"{i+1}. {prev_prompt[:30]}..." if len(prev_prompt) > 30 else f"{i+1}. {prev_prompt}")
else:
    st.sidebar.text("No previous prompts yet.")

# Instructions and information in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("How Multi-Agent System Works")
st.sidebar.markdown("""
1. **Text Agent** generates initial post
2. **Text Agent** extracts visual elements
3. **Text Agent** creates image prompt
4. **Image Agent** generates image
5. **Image Agent** analyzes image
6. **Text Agent** refines post based on image
7. **Collaboration Agent** finalizes content
""")

st.sidebar.markdown("---")
st.sidebar.info("This app uses Google's Gemini 1.5 Pro API for text generation and reasoning, " 
                "and Pollinations.ai for image generation. The agents work together to create a cohesive content package.")

# Version indicator
st.sidebar.markdown("---")
st.sidebar.text("App Version: 3.1 (Multi-Agent System + Debug)")

# Troubleshooting tips
with st.sidebar.expander("Troubleshooting Tips", expanded=True):
    st.markdown("""
    **If you're getting API errors:**
    
    1. Make sure you have a valid Gemini API key
    2. Check that your API key has access to Gemini 1.5 Pro
    3. Try updating the google-generativeai package:
       ```
       pip install --upgrade google-generativeai
       ```
    4. Ensure you have an active internet connection
    5. Check the Debug Information section for detailed error logs
    """)

# Add a clear cache button to help with updates
if st.sidebar.button("Clear Cache and Reload"):
    st.cache_data.clear()
    st.session_state.clear()
    st.rerun()
