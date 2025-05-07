import streamlit as st
import re
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

# App title and description
st.title("Multi-Agent Social Media Post Generator")
st.markdown("Generate engaging content & images with AI agents working together.")

# Configure API key with improved error handling
with st.sidebar:
    st.subheader("API Configuration")
    gemini_api_key = st.text_input("Enter your Gemini API Key", type="password")
    
    # Model selection to help with quota issues
    model_options = {
        "gemini-1.5-pro": "Gemini 1.5 Pro (Best quality, higher quota usage)",
        "gemini-1.5-flash": "Gemini 1.5 Flash (Faster, lower quota usage)",
        "gemini-1.0-pro": "Gemini 1.0 Pro (Lower quota usage)"
    }
    selected_model = st.selectbox(
        "Select Gemini Model",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=1  # Default to flash model to avoid quota issues
    )
    st.session_state['api_key_entered'] = bool(gemini_api_key)
    st.session_state['selected_model'] = selected_model

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
if 'api_available' not in st.session_state:
    st.session_state.api_available = False

# Try to import Gemini - with fallback
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    st.sidebar.error("⚠️ google-generativeai package not found. Some features will be limited.")
    st.session_state.debug_info.append("ImportError: google-generativeai package not found")

# Improved API key handling with rate limiting protection
if gemini_api_key and GEMINI_AVAILABLE:
    os.environ["GOOGLE_API_KEY"] = gemini_api_key
    genai.configure(api_key=gemini_api_key)
    
    # Direct debug output to help troubleshoot
    st.sidebar.subheader("API Connection Check")
    try:
        # Test the API connection with retry logic
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                model = genai.GenerativeModel(st.session_state['selected_model'])
                # Use a very minimal prompt to test connection
                test_response = model.generate_content("Hi")
                st.sidebar.success(f"✅ Connected to Gemini API successfully")
                st.sidebar.success(f"✅ Using model: {st.session_state['selected_model']}")
                st.session_state.api_available = True
                break
            except Exception as e:
                error_str = str(e)
                retry_count += 1
                
                # Check if it's a quota error
                if "429" in error_str and "quota" in error_str.lower():
                    st.sidebar.error(f"⚠️ Quota limit reached. Try switching to a different model or waiting.")
                    st.session_state.debug_info.append(f"API Quota Error: {error_str[:200]}...")
                    break
                elif retry_count >= max_retries:
                    st.sidebar.error(f"⚠️ Error connecting to Gemini API after {max_retries} attempts: {error_str[:100]}...")
                    st.session_state.debug_info.append(f"API Error: {error_str[:200]}...")
                else:
                    time.sleep(2)  # Wait before retrying
    except Exception as e:
        st.sidebar.error(f"⚠️ Error connecting to Gemini API: {str(e)[:100]}...")
        st.sidebar.warning("Please check your API key and internet connection.")
        st.session_state.debug_info.append(f"API Setup Error: {str(e)[:200]}...")
else:
    if not GEMINI_AVAILABLE:
        st.sidebar.warning("Please install the google-generativeai package to use Gemini API.")
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
    def __init__(self, model_name=None):
        if not model_name:
            model_name = st.session_state.get('selected_model', 'gemini-1.5-flash')
            
        try:
            if GEMINI_AVAILABLE and st.session_state.api_available:
                self.model = genai.GenerativeModel(model_name)
                self.api_available = True
            else:
                self.api_available = False
            self.name = "TextAgent"
            self.model_name = model_name
        except Exception as e:
            st.error(f"Error initializing TextGenerationAgent: {str(e)}")
            st.session_state.debug_info.append(f"TextGenerationAgent Init Error: {str(e)}")
            st.session_state.debug_info.append(traceback.format_exc())
            self.api_available = False
    
    def _safe_generate(self, prompt, max_retries=2, safety_threshold=0.8):
        """Safely generate content with retry logic and rate limiting protection"""
        if not self.api_available:
            return self._fallback_generate(prompt)
            
        retry_count = 0
        wait_time = 1
        
        while retry_count <= max_retries:
            try:
                # Add generation config to reduce token usage
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.7,
                        "top_p": 0.95,
                        "top_k": 40,
                        "max_output_tokens": 250,  # Limit output size
                    }
                )
                return response.text
            except Exception as e:
                error_str = str(e)
                retry_count += 1
                st.session_state.debug_info.append(f"Generation retry {retry_count}: {error_str[:150]}...")
                
                # Check if it's a quota error
                if "429" in error_str and "quota" in error_str.lower():
                    return self._fallback_generate(prompt)
                
                # Wait with exponential backoff
                if retry_count <= max_retries:
                    time.sleep(wait_time)
                    wait_time *= 2
        
        # If we've exhausted retries, use fallback
        return self._fallback_generate(prompt)
    
    def _fallback_generate(self, prompt):
        """Provide fallback content when API is unavailable"""
        st.session_state.debug_info.append(f"Using fallback generator for: {prompt[:50]}...")
        
        # Extract key terms from prompt
        terms = prompt.split()
        key_terms = [term for term in terms if len(term) > 4][:5]
        
        if "post" in prompt.lower() and "platform" in prompt.lower():
            return f"A compelling post about {' '.join(key_terms)}. #content #engagement"
        
        if "visual" in prompt.lower() or "image" in prompt.lower():
            return f"Key visual elements: professional, modern, {', '.join(key_terms)}"
            
        if "image prompt" in prompt.lower():
            return f"A professional image showing {' '.join(key_terms)} with vibrant colors and modern style."
            
        return f"Content related to {' '.join(key_terms)}. #trending"
        
    def generate_initial_post(self, prompt, platform):
        """Generate initial post based on user prompt"""
        try:
            full_prompt = (
                f"Create a short, engaging {platform} post (less than 50 words) based on this prompt: '{prompt}'. "
                f"Include relevant hashtags. Be creative and professional."
            )
            
            st.session_state.debug_info.append(f"Sending prompt to TextAgent: {full_prompt[:100]}...")
            
            generated_text = self._safe_generate(full_prompt)
            
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
                f"that would make for a compelling accompanying image. Format your response as a comma-separated list. "
                f"The post text is: '{text}'"
            )
            
            result = self._safe_generate(prompt)
            
            # Try to extract a clean list if JSON/list format isn't returned
            try:
                # Find JSON array in the response if present
                match = re.search(r'\[(.*?)\]', result.replace('\n', ''), re.DOTALL)
                if match:
                    json_str = match.group(0)
                    visual_elements = json.loads(json_str)
                    visual_elements_str = ", ".join(visual_elements)
                else:
                    # Otherwise use the text directly
                    visual_elements_str = result.strip()
            except json.JSONDecodeError:
                # If JSON parsing fails, use the raw text
                visual_elements_str = result.strip()
                
            # Add to agent conversation
            message = {"agent": self.name, "message": f"Extracted visual elements: {visual_elements_str}", "time": get_timestamp()}
            st.session_state.agent_conversation.append(message)
            
            return visual_elements_str
            
        except Exception as e:
            error_msg = str(e)
            st.session_state.debug_info.append(f"TextAgent Extract Elements Error: {error_msg}")
            st.session_state.debug_info.append(traceback.format_exc())
            return f"Visual elements related to the post"
    
    def create_image_prompt(self, user_prompt, visual_elements):
        """Create a detailed image prompt based on visual elements and user prompt"""
        try:
            prompt = (
                f"Create a short image generation prompt based on these elements:\n"
                f"1. User's original idea: '{user_prompt}'\n"
                f"2. Key visual elements: {visual_elements}\n\n"
                f"The prompt should be vivid and specific. "
                f"Keep it under 75 words."
            )
            
            image_prompt = self._safe_generate(prompt).strip()
            
            # Add to agent conversation
            message = {"agent": self.name, "message": f"Created image prompt: {image_prompt}", "time": get_timestamp()}
            st.session_state.agent_conversation.append(message)
            
            return image_prompt
            
        except Exception as e:
            error_msg = str(e)
            st.session_state.debug_info.append(f"TextAgent Image Prompt Error: {error_msg}")
            st.session_state.debug_info.append(traceback.format_exc())
            return f"Professional image about {user_prompt} showing {visual_elements}"

    def refine_post_with_image_feedback(self, original_post, image_analysis, platform):
        """Refine the post based on image analysis feedback"""
        try:
            prompt = (
                f"Refine this {platform} post to better match the image that was generated:\n\n"
                f"Original post: '{original_post}'\n\n"
                f"Image analysis: {image_analysis}\n\n"
                f"Create an improved version that references specific elements from the image "
                f"while maintaining the core message. Keep the post under 50 words "
                f"and include relevant hashtags."
            )
            
            refined_post = self._safe_generate(prompt).strip()
            
            # Add to agent conversation
            message = {"agent": self.name, "message": f"Refined post based on image feedback: {refined_post}", "time": get_timestamp()}
            st.session_state.agent_conversation.append(message)
            
            return refined_post
            
        except Exception as e:
            error_msg = str(e)
            st.session_state.debug_info.append(f"TextAgent Refine Post Error: {error_msg}")
            st.session_state.debug_info.append(traceback.format_exc())
            return original_post

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
            # Add cache buster to avoid browser caching
            cache_buster = int(time.time())
            image_url = f"https://image.pollinations.ai/prompt/{safe_prompt}%20{seed}?width=1024&height=1024&nologo=true&cb={cache_buster}"
            
            # Test if the URL is accessible with timeout and retry
            max_retries = 3
            retry_count = 0
            success = False
            
            while retry_count < max_retries and not success:
                try:
                    test_response = requests.head(image_url, timeout=5)
                    test_response.raise_for_status()
                    success = True
                except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        st.session_state.debug_info.append(f"Image generation failed after {max_retries} attempts: {str(e)}")
                        return None, f"Error generating image after multiple attempts: {str(e)}"
                    time.sleep(1)  # Wait before retry
            
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
            
            if text_agent.api_available:
                analysis = text_agent._safe_generate(prompt).strip()
            else:
                # Fallback if API not available
                elements = st.session_state.generated_image_prompt.split()
                key_terms = [term for term in elements if len(term) > 3][:5]
                analysis = f"The image likely shows {', '.join(key_terms)} with vibrant colors and a professional composition that would complement the social media post well."
            
            # Add to agent conversation
            message = {"agent": self.name, "message": f"Image analysis: {analysis}", "time": get_timestamp()}
            st.session_state.agent_conversation.append(message)
            
            return analysis
            
        except Exception as e:
            error_msg = str(e)
            st.session_state.debug_info.append(f"ImageAgent Analysis Error: {error_msg}")
            st.session_state.debug_info.append(traceback.format_exc())
            return f"The image shows the key elements mentioned in the post with a professional style."

# ================== COLLABORATION AGENT ==================

class CollaborationAgent:
    def __init__(self, model_name=None):
        if not model_name:
            model_name = st.session_state.get('selected_model', 'gemini-1.5-flash')
            
        try:
            if GEMINI_AVAILABLE and st.session_state.api_available:
                self.model = genai.GenerativeModel(model_name)
                self.api_available = True
            else:
                self.api_available = False
            self.name = "CollaborationAgent"
            self.model_name = model_name
        except Exception as e:
            st.error(f"Error initializing CollaborationAgent: {str(e)}")
            st.session_state.debug_info.append(f"CollaborationAgent Init Error: {str(e)}")
            st.session_state.debug_info.append(traceback.format_exc())
            self.api_available = False
    
    def _safe_generate(self, prompt, max_retries=2):
        """Safely generate content with retry logic and rate limiting protection"""
        if not self.api_available:
            return self._fallback_generate(prompt)
            
        retry_count = 0
        wait_time = 1
        
        while retry_count <= max_retries:
            try:
                # Add generation config to reduce token usage
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.7,
                        "top_p": 0.95,
                        "top_k": 40,
                        "max_output_tokens": 300,  # Limit output size
                    }
                )
                return response.text
            except Exception as e:
                error_str = str(e)
                retry_count += 1
                st.session_state.debug_info.append(f"Collab Agent retry {retry_count}: {error_str[:150]}...")
                
                # Check if it's a quota error
                if "429" in error_str and "quota" in error_str.lower():
                    return self._fallback_generate(prompt)
                
                # Wait with exponential backoff
                if retry_count <= max_retries:
                    time.sleep(wait_time)
                    wait_time *= 2
        
        # If we've exhausted retries, use fallback
        return self._fallback_generate(prompt)
    
    def _fallback_generate(self, prompt):
        """Provide fallback recommendations when API is unavailable"""
        st.session_state.debug_info.append(f"Using fallback collaboration for: {prompt[:50]}...")
        
        # Extract the post text from the prompt if possible
        post_match = re.search(r"Generated post: '(.*?)'", prompt, re.DOTALL)
        if post_match:
            post_text = post_match.group(1)
        else:
            post_text = "the generated post"
            
        return f"""Final Recommendation:
        
The image and text work well together. Consider emphasizing visual elements more directly in the text.

Final Post:
{post_text}"""
    
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
                f"Then provide a final recommendation for the post. Include a section labeled 'Final Post:' with the improved text."
            )
            
            recommendation = self._safe_generate(prompt).strip()
            
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
            return f"The content looks good as is. The image and text work well together.", post_text

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
                    st.error(f"Failed to generate
