# Replace the existing generate_image_with_pollinations function with this improved version
def generate_image_with_pollinations(prompt):
    try:
        # Create a safe prompt for the URL
        safe_prompt = prompt.replace(" ", "-").replace("&", "and")[:50]
        seed = str(uuid.uuid4())[:8]
        
        # Construct the image URL
        image_url = f"https://image.pollinations.ai/prompt/{safe_prompt}%20{seed}?width=1024&height=1024&nologo=true"
        
        # Just return the URL - don't try to download the image immediately
        return image_url, None
    
    except Exception as e:
        return None, f"Error generating image URL: {str(e)}"

# Update the image handling part of your code
if image_option == "Generate an image with AI":
    image_prompt = st.text_area("Describe the image you want to generate", 
                              placeholder="E.g., A professional workspace with a laptop, coffee cup, and notebook")
    
    if st.button("Preview Image") and image_prompt:
        with st.spinner("Generating image preview..."):
            image_url, error = generate_image_with_pollinations(image_prompt)
            
            if image_url:
                # Store the URL but don't try to download it yet
                st.session_state.generated_image_url = image_url
                
                # Display the image directly from URL
                st.image(image_url, caption="Generated image preview", use_container_width=True)
                
                # Only download the image when generating the post, not during preview
            else:
                st.error(f"Failed to generate image: {error}")

# Fix the part where you generate the post with the image
if st.button("Generate Post") and prompt:
    with st.spinner("Generating your social media post..."):
        # First, generate an image if that option was selected but not previewed
        if image_option == "Generate an image with AI" and image_prompt and not st.session_state.generated_image_url:
            with st.spinner("Generating image..."):
                image_url, error = generate_image_with_pollinations(image_prompt)
                if image_url:
                    st.session_state.generated_image_url = image_url
                else:
                    st.error(f"Failed to generate image: {error}")
        
        # For post generation, only download the image if absolutely necessary
        image_bytes = None
        if st.session_state.generated_image_url and (image_option == "Generate an image with AI"):
            try:
                # Only attempt to download the image when sending to Gemini API
                response = requests.get(st.session_state.generated_image_url, stream=True)
                if response.status_code == 200:
                    image_bytes = response.content
            except Exception as e:
                st.warning(f"Couldn't process the image for AI analysis, but it will still be displayed in your post.")

        # Then generate the post
        if prompt in st.session_state.prompt_history:
            st.session_state.generated_post = generate_post(prompt, platform, image_bytes, is_refinement=True)
        else:
            st.session_state.prompt_history.append(prompt)
            st.session_state.generated_post = generate_post(prompt, platform, image_bytes)
            # Store the original post for comparison during improvement
            st.session_state.original_post = st.session_state.generated_post
        
        st.session_state.feedback_provided = False

# Update the display section to use use_container_width instead of use_column_width
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
