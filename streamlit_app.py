# app.py - Streamlit Neurosurgeon Chatbot for Chiari Malformation

import streamlit as st
import os
import tempfile
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory

# Page configuration
st.set_page_config(
    page_title="Arnold Chiari Malformation - Consultant",
    page_icon="🧠",
    layout="centered"
)

# Custom CSS with improved contrast
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .st-emotion-cache-16txtl3 h1 {
        color: #2c3e50;
    }
    .neurosurgeon {
        background-color: #e6f3ff;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 5px solid #4e8cff;
        color: #000000 !important; /* Ensuring dark text color */
        font-weight: 400;
    }
    .user {
        background-color: #f0f0f0;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 5px solid #a0a0a0;
        color: #000000 !important; /* Ensuring dark text color */
        font-weight: 400;
    }
    /* Additional styling to ensure text is visible */
    .neurosurgeon p, .user p {
        color: #000000 !important;
        opacity: 1 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'conversation_chain' not in st.session_state:
    st.session_state.conversation_chain = None
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'token_count' not in st.session_state:
    st.session_state.token_count = 0

# Domain keywords for checking relevance
DOMAIN_KEYWORDS = [
    "brain", "chiari", "spine", "nerve", "type", "types", "malformation",
    "symptom", "treatment", "surgery", "diagnosis", "mri", "headache",
    "pain", "syrinx", "cerebrospinal", "fluid", "cerebellum", "tonsils",
    "decompression", "syringomyelia", "herniation", "foramen magnum"
]

# Enhanced Neurosurgeon persona prompt template
NEUROSURGEON_TEMPLATE = """
You are a board-certified neurosurgeon with extensive specialized experience in treating Chiari malformations and related neurological conditions. You're known for your empathetic approach to patient care, technical expertise, and ability to explain complex medical concepts in understandable terms.

As a medical professional, you always:
- Provide accurate, evidence-based information
- Speak with compassion and understanding
- Acknowledge the challenges patients face
- Explain concepts clearly without overwhelming medical jargon
- Maintain a calm, reassuring tone
- Balance honesty about conditions with hope and support

The following is your recent conversation with the patient. Use this conversation history to inform your response and maintain continuity:
{chat_history}

Current question from the patient: {question}

Relevant context information from medical literature: 
{context}

When answering follow-up questions, make sure to refer to information you've already provided and maintain a coherent conversation. If the patient references something you discussed earlier, acknowledge that shared context.

Neurosurgeon's response:
"""

PROMPT = PromptTemplate(
    template=NEUROSURGEON_TEMPLATE,
    input_variables=["chat_history", "context", "question"]
)

def check_if_domain_relevant(query: str) -> bool:
    """Check if the query contains any domain-specific keywords"""
    query = query.lower()
    for keyword in DOMAIN_KEYWORDS:
        if keyword in query:
            return True
    return False

def preprocess_query(query: str, chat_history):
    """Enhance query with context from recent conversation if needed"""
    if not chat_history or len(chat_history) < 2:
        return query
        
    # Check for follow-up indicators (pronouns without clear references, etc.)
    follow_up_indicators = ["it", "this", "that", "they", "these", "those", "the condition", 
                            "the treatment", "the procedure", "what about", "how about",
                            "?", "and", "or", "but"]
    
    query_lower = query.lower()
    is_likely_follow_up = any(indicator in query_lower for indicator in follow_up_indicators) and len(query.split()) < 8
    
    if is_likely_follow_up:
        # Get the last response from the assistant
        recent_exchanges = chat_history[-4:] if len(chat_history) >= 4 else chat_history
        recent_context = " ".join([msg["content"] for msg in recent_exchanges])
        
        # Combine with the current query for better context
        enhanced_query = f"Given our recent discussion about {recent_context[:200]}..., {query}"
        return enhanced_query
    
    return query

def process_pdf(pdf_file):
    """Process uploaded PDF file and create a vector database"""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(pdf_file.getvalue())
        temp_file_path = temp_file.name
    
    # Load and process the PDF
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    
    # Split text with smaller chunks and more overlap for better context
    text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=300)
    docs = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Remove temp file
    os.unlink(temp_file_path)
    
    return vectorstore

def get_formatted_chat_history(chat_history):
    """Format chat history for better context retention"""
    formatted_history = ""
    
    for exchange in enumerate(chat_history[-6:]):  # Only use the most recent 6 messages
        # Handle different formats of chat history
        if hasattr(exchange, 'type') and hasattr(exchange, 'content'):
            # Handle LangChain message objects
            role = "Patient" if exchange.type == "human" else "Neurosurgeon"
            formatted_history += f"{role}: {exchange.content}\n\n"
        elif isinstance(exchange, tuple) and len(exchange) == 2:
            # Handle (human_message, ai_message) tuples
            formatted_history += f"Patient: {exchange[0]}\n\nNeurosurgeon: {exchange[1]}\n\n"
        elif isinstance(exchange, dict) and "role" in exchange:
            # Handle dictionary format with role key
            role = "Patient" if exchange["role"] == "user" else "Neurosurgeon"
            formatted_history += f"{role}: {exchange['content']}\n\n"
        elif isinstance(exchange, dict) and "content" in exchange:
            # Handle dictionary format without role but with content
            formatted_history += f"Message: {exchange['content']}\n\n"
        else:
            # Fallback handling for any other format
            formatted_history += f"Message: {str(exchange)}\n\n"
    
    return formatted_history

def create_conversation_chain(vectorstore):
    """Create a conversational chain with the processed PDF"""
    # Initialize enhanced memory with summarization capability
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2)
    
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        output_key='answer',
        input_key="question",
        max_token_limit=4000  # Set a high token limit to retain more context
    )
    
    # Setup Conversational Retrieval Chain with improved context handling
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),  # Increased from 3 to 5
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True,  # Helpful for debugging
        get_chat_history=get_formatted_chat_history,
        verbose=False
    )
    
    return conversation_chain

def estimate_tokens(text):
    """Roughly estimate token count (4 chars ≈ 1 token)"""
    return len(text) // 4

def main():
    st.title("🧠 Arnold Chiari Malformation Consultant")
    
    # Sidebar for API key and file upload
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input
        if not st.session_state.gemini_api_key:
            api_key = st.text_input("Enter Gemini API Key:", type="password")
            if api_key:
                try:
                    # Configure Gemini with the provided API key
                    os.environ["GOOGLE_API_KEY"] = api_key
                    genai.configure(api_key=api_key)
                    st.session_state.gemini_api_key = api_key
                    st.success("API key configured successfully!")
                except Exception as e:
                    st.error(f"Error configuring API key: {str(e)}")
        else:
            st.success("API key is configured.")
            if st.button("Reset API Key"):
                st.session_state.gemini_api_key = ""
                st.rerun()
        
        # PDF upload
        if st.session_state.gemini_api_key and not st.session_state.pdf_processed:
            st.subheader("Upload Chiari PDF File")
            uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
            
            if uploaded_file is not None:
                with st.spinner("Processing PDF... This may take a moment."):
                    try:
                        # Process the PDF and create a conversation chain
                        vectorstore = process_pdf(uploaded_file)
                        st.session_state.conversation_chain = create_conversation_chain(vectorstore)
                        st.session_state.pdf_processed = True
                        st.success("PDF processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing PDF: {str(e)}")
        
        # Conversation stats and reset button
        if st.session_state.pdf_processed:
            st.subheader("Conversation Stats")
            st.text(f"Approx. token count: {st.session_state.token_count}")
            
            if st.button("Clear Conversation"):
                st.session_state.chat_history = []
                # Reset conversation chain to clear memory
                if hasattr(st.session_state.conversation_chain, 'memory'):
                    st.session_state.conversation_chain.memory.clear()
                st.session_state.token_count = 0
                st.rerun()
        
        # Information section
        st.subheader("About")
        st.markdown("""
        This chatbot is powered by a AI consultant specializing in Arnold Chiari malformation.
        
        Ask questions about:
        - Chiari malformation types and causes
        - Symptoms and diagnosis
        - Treatment options and surgery
        - Recovery and prognosis
        """)

    # Main chat interface
    if st.session_state.conversation_chain:
        # Introduction message when first loading
        if not st.session_state.chat_history:
            st.markdown(
                """<div class="neurosurgeon">
                <p>Hello, I'm a neurosurgeon specializing in Arnold Chiari malformations. I'm here to answer your questions about this condition and provide the guidance you need. How can I help you today?</p>
                </div>""", 
                unsafe_allow_html=True
            )
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""<div class="user"><p>{message["content"]}</p></div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="neurosurgeon"><p>{message["content"]}</p></div>""", unsafe_allow_html=True)
        
        # Input for new questions
        user_question = st.chat_input("Ask your question about Chiari malformation...")
        
        if user_question:
            # Track token usage (approximate)
            st.session_state.token_count += estimate_tokens(user_question)
            
            # Add user message to chat history for display
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            
            # Display user message
            st.markdown(f"""<div class="user"><p>{user_question}</p></div>""", unsafe_allow_html=True)
            
            # Preprocess query to enhance context for follow-up questions
            enhanced_question = preprocess_query(user_question, st.session_state.chat_history)
            
            with st.spinner("Thinking..."):
                try:
                    if check_if_domain_relevant(enhanced_question):
                        # Get response from the conversation chain
                        response = st.session_state.conversation_chain({"question": enhanced_question})
                        answer = response['answer']
                        
                        # Track token usage (approximate)
                        st.session_state.token_count += estimate_tokens(answer)
                    else:
                        # Off-topic response
                        answer = "I'm sorry, but I specialize specifically in Chiari malformation and related neurological conditions. To provide you with the most helpful information, could you please ask me about topics related to Chiari, brain structure, spinal issues, or neurological symptoms? I want to ensure I give you accurate and relevant guidance."
                        
                        # We should also update the memory for continuity
                        if hasattr(st.session_state.conversation_chain, 'memory'):
                            st.session_state.conversation_chain.memory.save_context(
                                {"question": enhanced_question}, 
                                {"answer": answer}
                            )
                except Exception:
                    answer = "I apologize, but I encountered an error while processing your question. Please try rephrasing or asking a different question. If the problem persists, you may need to reset the conversation."
            
            # Display assistant response
            st.markdown(f"""<div class="neurosurgeon"><p>{answer}</p></div>""", unsafe_allow_html=True)
            
            # Add assistant message to chat history for display
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
    else:
        # Instructions when not yet configured
        if not st.session_state.gemini_api_key:
            st.info("👈 Please enter your Gemini API key in the sidebar to get started.")
        elif not st.session_state.pdf_processed:
            st.info("👈 Please upload the Chiari PDF file in the sidebar to continue.")

if __name__ == "__main__":
    main()
