import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
import os
from dotenv import load_dotenv
import openai
import PyPDF2
import docx
import pandas as pd
import json
import re

# Load environment variables for local development
load_dotenv()

# Configure APIs - Use Streamlit secrets for deployed apps, otherwise use .env
try:
    # Try to get the secret from Streamlit's secrets management
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except (StreamlitSecretNotFoundError, KeyError):
    # If the secret is not found (e.g., running locally), use the environment variable
    openai.api_key = os.getenv("OPENAI_API_KEY")

# Prompts for text-based analysis
CATEGORY_PROMPTS = {
    'academic': {
        'summary': """Create a concise academic summary (max 150 words) covering:\n• Main research objective\n• Key findings\n• Primary conclusion\n\nKeep it brief and to the point.""",
        'critique': """Provide a detailed academic critique covering:\n• **Strengths**: Methodology, evidence, clarity\n• **Areas for Improvement**: Gaps, limitations, suggestions\n• **Technical Analysis**: Research design, statistical methods\n• **Impact Assessment**: Contribution to field, practical implications""",
        'improve': """Suggest specific improvements for:\n• **Content Enhancement**: Additional research, better evidence\n• **Structural Improvements**: Organization, flow, clarity\n• **Methodological Refinements**: Better approaches, stronger analysis\n• **Presentation**: Formatting, citations, accessibility"""
    },
    'professional': {
        'summary': """Create a concise professional summary (max 150 words) highlighting:\n• Main objective\n• Key outcomes\n• Primary recommendation\n\nKeep it brief and actionable.""",
        'critique': """Provide a professional critique covering:\n• **Business Value**: ROI, strategic alignment, market relevance\n• **Technical Quality**: Accuracy, completeness, feasibility\n• **Communication**: Clarity, persuasiveness, audience appropriateness\n• **Implementation**: Practicality, timeline, resource requirements""",
        'improve': """Suggest professional improvements for:\n• **Business Impact**: Enhanced value proposition, better metrics\n• **Technical Excellence**: More robust analysis, better data\n• **Communication**: Clearer messaging, better presentation\n• **Execution**: More actionable recommendations, better planning"""
    },
    'media': {
        'summary': """Create a concise media summary (max 150 words) covering:\n• Main story\n• Key points\n• Impact\n\nKeep it brief and engaging.""",
        'critique': """Provide a media critique covering:\n• **Content Quality**: Accuracy, balance, newsworthiness\n• **Presentation**: Writing style, visual appeal, engagement\n• **Impact**: Reach, influence, public understanding\n• **Ethics**: Fairness, bias, responsible reporting""",
        'improve': """Suggest media improvements for:\n• **Content Enhancement**: Better sources, deeper analysis\n• **Presentation**: More engaging format, better visuals\n• **Impact**: Wider reach, stronger engagement\n• **Credibility**: Better fact-checking, more balanced coverage"""
    },
    'marketing': {
        'summary': """Create a concise marketing summary (max 150 words) highlighting:\n• Target audience\n• Key message\n• Main outcome\n\nKeep it brief and focused.""",
        'critique': """Provide a marketing critique covering:\n• **Strategy**: Target audience fit, positioning, competitive advantage\n• **Execution**: Creative quality, channel effectiveness, timing\n• **Performance**: Metrics, ROI, conversion rates\n• **Brand Alignment**: Consistency, authenticity, long-term impact""",
        'improve': """Suggest marketing improvements for:\n• **Strategy**: Better targeting, stronger positioning\n• **Creative**: More compelling messaging, better visuals\n• **Execution**: Optimized channels, better timing\n• **Measurement**: Better metrics, improved tracking"""
    }
}

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'csv', 'xlsx', 'pptx', 'md', 'json', 'html'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_prompt(category, prompt_type):
    if category in CATEGORY_PROMPTS:
        return CATEGORY_PROMPTS[category][prompt_type]
    return CATEGORY_PROMPTS['academic'][prompt_type]  # Default to academic prompts

def extract_text_from_file(file, file_extension):
    try:
        if file_extension == 'txt':
            return file.read().decode('utf-8')
        elif file_extension == 'pdf':
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        elif file_extension == 'docx':
            doc = docx.Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        elif file_extension == 'csv':
            df = pd.read_csv(file)
            return df.to_string()
        elif file_extension == 'xlsx':
            df = pd.read_excel(file)
            return df.to_string()
        elif file_extension == 'md':
            return file.read().decode('utf-8')
        elif file_extension == 'json':
            data = json.load(file)
            return json.dumps(data, indent=2)
        elif file_extension == 'html':
            content = file.read().decode('utf-8')
            clean_text = re.sub(r'<[^>]+>', '', content)
            return clean_text
        else:
            return "Unsupported file type"
    except Exception as e:
        return f"Error extracting text: {str(e)}"

# Inject custom CSS for theme (mimic original look)
with open("static/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
    <div class="header">
        <h1>Essay Critic</h1>
        <p>Intelligent Text & File Analysis</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar for navigation
section = st.sidebar.radio(
    "Choose Analysis Type:",
    ("Text Analysis", "File Analysis"),
    index=0
)

# --- TEXT ANALYSIS ---
if section == "Text Analysis":
    st.subheader("📝 Text Analysis")
    # Category selection
    categories = list(CATEGORY_PROMPTS.keys())
    category = st.selectbox("Select Category", categories, index=0)
    text = st.text_area("Paste your text here:", height=200)
    if st.button("Analyze Text", type="primary"):
        if not openai.api_key:
            st.error("OpenAI API key not configured. Please add OPENAI_API_KEY to your .env file or Streamlit secrets.")
        elif not text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing text with GPT-4..."):
                try:
                    # Get prompts
                    summary_prompt = get_prompt(category, 'summary')
                    critique_prompt = get_prompt(category, 'critique')
                    improve_prompt = get_prompt(category, 'improve')
                    # Summary
                    summary_response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a professional text analyst. Format your response with markdown. Keep summaries to 150 words."},
                            {"role": "user", "content": f"{summary_prompt}\n\nText to summarize:\n{text}"}
                        ],
                        temperature=0.7,
                        max_tokens=400
                    )
                    summary = summary_response.choices[0].message.content.strip()
                    # Critique
                    critique_response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a professional text analyst. Format your response with markdown. Provide detailed, constructive critiques."},
                            {"role": "user", "content": f"{critique_prompt}\n\nText to critique:\n{text}"}
                        ],
                        temperature=0.7,
                        max_tokens=800
                    )
                    critique = critique_response.choices[0].message.content.strip()
                    # Improvements
                    improve_response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a professional text analyst. Format your response with markdown. Provide specific, actionable improvements."},
                            {"role": "user", "content": f"{improve_prompt}\n\nText to improve:\n{text}"}
                        ],
                        temperature=0.7,
                        max_tokens=800
                    )
                    improved_summary = improve_response.choices[0].message.content.strip()
                    # Display results
                    st.markdown("<div class='results'>", unsafe_allow_html=True)
                    st.markdown("## Summary\n" + summary)
                    st.markdown("## Critique\n" + critique)
                    st.markdown("## Suggestions for Improvement\n" + improved_summary)
                    st.markdown("</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# --- FILE ANALYSIS ---
if section == "File Analysis":
    st.subheader("📁 File Analysis")
    categories = list(CATEGORY_PROMPTS.keys())
    category = st.selectbox("Select Category", categories, index=0)
    uploaded_file = st.file_uploader("Upload a file:", type=list(ALLOWED_EXTENSIONS))
    if uploaded_file is not None:
        file_extension = uploaded_file.name.rsplit('.', 1)[1].lower()
        if allowed_file(uploaded_file.name):
            text_content = extract_text_from_file(uploaded_file, file_extension)
            if text_content.startswith("Error"):
                st.error(text_content)
            else:
                st.text_area("Extracted Text (editable):", value=text_content, height=200, key="file_text_area")
                if st.button("Analyze File", key="analyze_file_btn", type="primary"):
                    if not openai.api_key:
                        st.error("OpenAI API key not configured. Please add OPENAI_API_KEY to your .env file or Streamlit secrets.")
                    else:
                        with st.spinner("Analyzing file content with GPT-4..."):
                            try:
                                analysis_text = st.session_state.file_text_area
                                summary_prompt = get_prompt(category, 'summary')
                                critique_prompt = get_prompt(category, 'critique')
                                improve_prompt = get_prompt(category, 'improve')
                                # Summary, Critique, and Improvement calls (similar to Text Analysis)
                                summary_response = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "system", "content": "You are a professional text analyst..."}, {"role": "user", "content": f"{summary_prompt}\n\n{analysis_text}"}], temperature=0.7, max_tokens=400)
                                summary = summary_response.choices[0].message.content.strip()
                                critique_response = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "system", "content": "You are a professional text analyst..."}, {"role": "user", "content": f"{critique_prompt}\n\n{analysis_text}"}], temperature=0.7, max_tokens=800)
                                critique = critique_response.choices[0].message.content.strip()
                                improve_response = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "system", "content": "You are a professional text analyst..."}, {"role": "user", "content": f"{improve_prompt}\n\n{analysis_text}"}], temperature=0.7, max_tokens=800)
                                improved_summary = improve_response.choices[0].message.content.strip()
                                # Display results
                                st.markdown("<div class='results'>", unsafe_allow_html=True)
                                st.markdown("## Summary\n" + summary)
                                st.markdown("## Critique\n" + critique)
                                st.markdown("## Suggestions for Improvement\n" + improved_summary)
                                st.markdown("</div>", unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"An error occurred: {e}")
        else:
            st.error("Invalid file type.") 
