import os
import google.generativeai as genai
import streamlit as st
import langchain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import tiktoken
import re
from pathlib import Path

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Assignment 2 - Anubhav',
    page_icon=':book:',  # Use ':books:' for multiple books or ':robot_face:' for a robot
)

# -----------------------------------------------------------------------------


# Display versions (optional)
st.sidebar.markdown("### Library Versions")
st.sidebar.markdown(f"google.generativeai: {genai.__version__}")
st.sidebar.markdown(f"streamlit: {st.__version__}")
st.sidebar.markdown(f"langchain: {langchain.__version__}")

# Initialize token counters in session state
if 'tokens_consumed' not in st.session_state:
    st.session_state.tokens_consumed = 0
if 'query_tokens' not in st.session_state:
    st.session_state.query_tokens = 0
if 'response_tokens' not in st.session_state:
    st.session_state.response_tokens = 0

# Get API key from Streamlit secrets
if 'GOOGLE_API_KEY' in st.secrets:
    api_key = st.secrets['GOOGLE_API_KEY']
    genai.configure(api_key=api_key)
else:
    st.error("API key not found in secrets. Please add GOOGLE_API_KEY to your Streamlit secrets.")
    st.stop()

# Initialize tiktoken encoder for token counting
encoder = tiktoken.get_encoding("cl100k_base")

# Initialize the Gemini AI model
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest",
        temperature=0.7,
        max_tokens=8000
    )

llm = load_llm()

# SWOT analysis prompt template
prompt_template = """
You are a senior consultant at McKinsey, an expert at analyzing companies.
You specialize in presenting a detailed SWOT analysis for a company based on the information provided below.
Here is the information that should be considered:
{company_info}

Please ensure, the analysis is clear, concise, and highlights the most important factors for each quadrant.
Please provide a SWOT analysis in the following format:
**Strengths:**
- [Strength 1]
- [Strength 2]
...

**Weaknesses:**
- [Weakness 1]
- [Weakness 2]
...

**Opportunities:**
- [Opportunity 1]
- [Opportunity 2]
...

**Threats:**
- [Threat 1]
- [Threat 2]
...
"""

# Create a PromptTemplate instance
prompt = PromptTemplate(input_variables=["company_info"], template=prompt_template)

# Initialize the LLMChain with the prompt and LLM
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Function for generating SWOT analysis
def get_swot_analysis(company_info: str):
    return llm_chain.run(company_info)

# Helper functions for processing and displaying SWOT analysis
def convert_md_bold_to_html(text: str) -> str:
    """Converts double-asterisk Markdown to HTML bold tags."""
    return re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)

def remove_single_asterisks(text: str) -> str:
    """Removes single asterisks followed by a space."""
    return re.sub(r"^\*\s+", "", text, flags=re.MULTILINE)

def parse_subheading_bullets(text: str):
    """Finds lines that begin with '* ' or '- ', then cleans and returns them."""
    lines = re.findall(r"^(?:\*|-)\s+(.*)", text, flags=re.MULTILINE)
    bullet_points = []
    for line in lines:
        line = convert_md_bold_to_html(line)
        line = remove_single_asterisks(line)
        bullet_points.append(line.strip())
    return bullet_points

def display_swot_analysis(strengths, weaknesses, opportunities, threats):
    def render_quadrant(content: str, title: str, color: str):
        st.markdown(
            f"""
            <div style="background-color:{color}; padding: 10px; border-radius: 10px;">
                <h3 style="color:white;">{title}</h3>
                <ul style="color:white;">
            """,
            unsafe_allow_html=True
        )

        bullet_lines = parse_subheading_bullets(content)
        if not bullet_lines:
            bullet_lines = [content.strip()]

        for line in bullet_lines:
            st.markdown(f"<li>{line}</li>", unsafe_allow_html=True)

        st.markdown("</ul></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        render_quadrant(strengths, "Strengths", "#4CAF50")
    with col2:
        render_quadrant(weaknesses, "Weaknesses", "#F44336")

    col3, col4 = st.columns(2)
    with col3:
        render_quadrant(opportunities, "Opportunities", "#2196F3")
    with col4:
        render_quadrant(threats, "Threats", "#FF9800")

# Main Streamlit app
st.set_page_config(page_title="SWOT Analysis", page_icon="📊")
st.title("SWOT Analysis Application")
st.write("Upload a text file or enter text below to generate SWOT Analysis:")

# File uploader
uploaded_file = st.file_uploader("Choose a text file", type=["txt"])

# Text input
text_input = st.text_area("Or enter text directly:")

# Process the text
if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
elif text_input:
    text = text_input
else:
    text = None

query_tokens = 0
response_tokens = 0

if st.button("Generate SWOT Analysis"):
    if text:
        # Generate and display the SWOT Analysis
        st.subheader("Output")
        with st.spinner('Generating SWOT Analysis...'):
            swot_output = get_swot_analysis(text)
        
        sections = ["Strengths", "Weaknesses", "Opportunities", "Threats"]
        swot_blocks = {s: "" for s in sections}
    
        # Regex pattern to match each section until the next section or end of string
        for section in sections:
            pattern = rf"\*\*{section}:\*\*\s*((?:(?!\*\*(?:Strengths|Weaknesses|Opportunities|Threats):\*\*).)*)"
            match = re.search(pattern, swot_output, re.DOTALL)
            if match:
                swot_blocks[section] = match.group(1).strip()
            else:
                swot_blocks[section] = ""
    
        # Display the SWOT quadrants
        st.title('SWOT Analysis Results')
        display_swot_analysis(
            swot_blocks["Strengths"],
            swot_blocks["Weaknesses"],
            swot_blocks["Opportunities"],
            swot_blocks["Threats"]
        )
        
        # Count tokens
        query_tokens = len(encoder.encode(text))
        response_tokens = len(encoder.encode(swot_output))
        
        # Update token counts in session state
        st.session_state.query_tokens += query_tokens
        st.session_state.response_tokens += response_tokens
        st.session_state.tokens_consumed += (query_tokens + response_tokens)
    else:
        st.info("Please upload a file or enter text to generate the SWOT analysis.")

# Display token usage in sidebar
st.sidebar.markdown("### Token Usage")
st.sidebar.markdown(f"Total Tokens Consumed: {st.session_state.tokens_consumed}")
st.sidebar.markdown(f"Query Tokens: {st.session_state.query_tokens}")
st.sidebar.markdown(f"Response Tokens: {st.session_state.response_tokens}")
