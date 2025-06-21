import streamlit as st
import os
from dotenv import load_dotenv
import openai
import PyPDF2
import docx
import pandas as pd
import json
import re
import base64
import io
from PIL import Image
import requests

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

# Configure Google Cloud Vision API (alternative: Azure Computer Vision)
GOOGLE_CLOUD_API_KEY = os.getenv('GOOGLE_CLOUD_API_KEY')
AZURE_VISION_KEY = os.getenv('AZURE_VISION_KEY')
AZURE_VISION_ENDPOINT = os.getenv('AZURE_VISION_ENDPOINT')

# Check if OCR services are available
OCR_AVAILABLE = bool(GOOGLE_CLOUD_API_KEY or AZURE_VISION_KEY)

# Configure file upload
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'csv', 'xlsx', 'pptx', 'md', 'json', 'html'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

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

def extract_text_from_image_google_vision(image_data):
    try:
        if not GOOGLE_CLOUD_API_KEY:
            return {"error": "Google Cloud Vision API key not configured. Please add GOOGLE_CLOUD_API_KEY to your .env file.", "success": False}
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_CLOUD_API_KEY}"
        payload = {
            "requests": [
                {
                    "image": {"content": image_base64},
                    "features": [{"type": "TEXT_DETECTION"}]
                }
            ]
        }
        response = requests.post(url, json=payload)
        result = response.json()
        if 'responses' in result and result['responses'][0].get('textAnnotations'):
            text_annotations = result['responses'][0]['textAnnotations']
            full_text = text_annotations[0]['description'] if text_annotations else ""
            words = []
            for annotation in text_annotations[1:]:
                vertices = annotation['boundingPoly']['vertices']
                x_coords = [v['x'] for v in vertices]
                y_coords = [v['y'] for v in vertices]
                words.append({
                    'text': annotation['description'],
                    'bounds': {
                        'x1': min(x_coords),
                        'y1': min(y_coords),
                        'x2': max(x_coords),
                        'y2': max(y_coords)
                    }
                })
            return {'full_text': full_text, 'words': words, 'success': True}
        else:
            return {"error": "No text found in image", "success": False}
    except Exception as e:
        return {"error": f"Error extracting text: {str(e)}", "success": False}

def extract_text_from_image_azure_vision(image_data):
    try:
        if not AZURE_VISION_KEY or not AZURE_VISION_ENDPOINT:
            return {"error": "Azure Computer Vision API not configured. Please add AZURE_VISION_KEY and AZURE_VISION_ENDPOINT to your .env file.", "success": False}
        url = f"{AZURE_VISION_ENDPOINT}/vision/v3.2/read/analyze"
        headers = {
            'Ocp-Apim-Subscription-Key': AZURE_VISION_KEY,
            'Content-Type': 'application/octet-stream'
        }
        response = requests.post(url, headers=headers, data=image_data)
        if response.status_code == 202:
            operation_url = response.headers['Operation-Location']
            import time
            for _ in range(10):
                time.sleep(1)
                result_response = requests.get(operation_url, headers={'Ocp-Apim-Subscription-Key': AZURE_VISION_KEY})
                if result_response.status_code == 200:
                    result = result_response.json()
                    if result['status'] == 'succeeded':
                        words = []
                        full_text = ""
                        for read_result in result['analyzeResult']['readResults']:
                            for line in read_result['lines']:
                                full_text += line['text'] + "\n"
                                for word in line['words']:
                                    bounds = word['boundingBox']
                                    words.append({
                                        'text': word['text'],
                                        'bounds': {
                                            'x1': bounds[0],
                                            'y1': bounds[1],
                                            'x2': bounds[2],
                                            'y2': bounds[3]
                                        }
                                    })
                        return {'full_text': full_text, 'words': words, 'success': True}
            return {"error": "Timeout waiting for OCR results", "success": False}
        else:
            return {"error": f"API request failed: {response.status_code}", "success": False}
    except Exception as e:
        return {"error": f"Error extracting text: {str(e)}", "success": False}

def extract_text_from_selected_area(words, selection_bounds):
    selected_text = []
    for word in words:
        word_bounds = word['bounds']
        if (word_bounds['x1'] >= selection_bounds['x1'] and 
            word_bounds['x2'] <= selection_bounds['x2'] and
            word_bounds['y1'] >= selection_bounds['y1'] and 
            word_bounds['y2'] <= selection_bounds['y2']):
            selected_text.append(word['text'])
    return ' '.join(selected_text)

# Inject custom CSS for theme (mimic original look)
with open("static/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
    <div class="header">
        <h1>Essay Critic</h1>
        <p>Intelligent Text Analysis & File Summarizer</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar for navigation
section = st.sidebar.radio(
    "Choose Analysis Type:",
    ("Text Analysis", "File Analysis", "Image OCR"),
    index=0
)

# Category selection
categories = list(CATEGORY_PROMPTS.keys())
category = st.sidebar.selectbox("Select Category", categories, index=0)

# --- TEXT ANALYSIS ---
if section == "Text Analysis":
    st.subheader("📝 Text Analysis")
    text = st.text_area("Paste your text here:", height=200)
    if st.button("Analyze Text", type="primary"):
        if not openai.api_key:
            st.error("OpenAI API key not configured. Please add OPENAI_API_KEY to your .env file.")
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
                            {"role": "system", "content": "You are a professional text analyst specializing in creating concise summaries (max 150 words). Always format your response with proper markdown formatting including headers, bullet points, and emphasis where appropriate. Use ## for main headers, ### for subheaders, and proper bullet points with - for lists. Keep summaries brief and to the point."},
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
                            {"role": "system", "content": "You are a professional text analyst specializing in providing detailed, constructive critiques. Always format your response with proper markdown formatting including headers, bullet points, and emphasis where appropriate. Use ## for main headers, ### for subheaders, and proper bullet points with - for lists. Structure your critique with clear sections like 'Strengths', 'Areas for Improvement', 'Technical Analysis', etc."},
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
                            {"role": "system", "content": "You are a professional text analyst specializing in improving text quality. Always format your response with proper markdown formatting including headers, bullet points, and emphasis where appropriate. Use ## for main headers, ### for subheaders, and proper bullet points with - for lists. Provide specific, actionable improvements with clear examples."},
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
                    st.error(f"Error: {e}")

# --- FILE ANALYSIS ---
if section == "File Analysis":
    st.subheader("📁 File Analysis")
    uploaded_file = st.file_uploader("Upload a file (txt, pdf, docx, csv, xlsx, pptx, md, json, html):", type=list(ALLOWED_EXTENSIONS))
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
                        st.error("OpenAI API key not configured. Please add OPENAI_API_KEY to your .env file.")
                    else:
                        with st.spinner("Analyzing file content with GPT-4..."):
                            try:
                                summary_prompt = get_prompt(category, 'summary')
                                critique_prompt = get_prompt(category, 'critique')
                                improve_prompt = get_prompt(category, 'improve')
                                # Summary
                                summary_response = openai.ChatCompletion.create(
                                    model="gpt-4",
                                    messages=[
                                        {"role": "system", "content": "You are a professional text analyst specializing in creating concise summaries (max 150 words). Always format your response with proper markdown formatting including headers, bullet points, and emphasis where appropriate. Use ## for main headers, ### for subheaders, and proper bullet points with - for lists. Keep summaries brief and to the point."},
                                        {"role": "user", "content": f"{summary_prompt}\n\nText to summarize:\n{text_content}"}
                                    ],
                                    temperature=0.7,
                                    max_tokens=400
                                )
                                summary = summary_response.choices[0].message.content.strip()
                                # Critique
                                critique_response = openai.ChatCompletion.create(
                                    model="gpt-4",
                                    messages=[
                                        {"role": "system", "content": "You are a professional text analyst specializing in providing detailed, constructive critiques. Always format your response with proper markdown formatting including headers, bullet points, and emphasis where appropriate. Use ## for main headers, ### for subheaders, and proper bullet points with - for lists. Structure your critique with clear sections like 'Strengths', 'Areas for Improvement', 'Technical Analysis', etc."},
                                        {"role": "user", "content": f"{critique_prompt}\n\nText to critique:\n{text_content}"}
                                    ],
                                    temperature=0.7,
                                    max_tokens=800
                                )
                                critique = critique_response.choices[0].message.content.strip()
                                # Improvements
                                improve_response = openai.ChatCompletion.create(
                                    model="gpt-4",
                                    messages=[
                                        {"role": "system", "content": "You are a professional text analyst specializing in improving text quality. Always format your response with proper markdown formatting including headers, bullet points, and emphasis where appropriate. Use ## for main headers, ### for subheaders, and proper bullet points with - for lists. Provide specific, actionable improvements with clear examples."},
                                        {"role": "user", "content": f"{improve_prompt}\n\nText to improve:\n{text_content}"}
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
                                st.error(f"Error: {e}")
        else:
            st.error("Invalid file type.")

# --- IMAGE OCR ---
if section == "Image OCR":
    st.subheader("🖼️ Image OCR & Analysis")
    if not OCR_AVAILABLE:
        st.warning("OCR services not configured. Please add either GOOGLE_CLOUD_API_KEY or AZURE_VISION_KEY to your .env file.")
    else:
        uploaded_image = st.file_uploader("Upload an image (png, jpg, jpeg, gif, bmp, tiff, webp):", type=list(ALLOWED_IMAGE_EXTENSIONS))
        if uploaded_image is not None:
            image_bytes = uploaded_image.read()
            st.image(image_bytes, caption="Uploaded Image", use_column_width=True)
            if st.button("Extract Text from Image", key="extract_image_btn", type="primary"):
                with st.spinner("Extracting text from image..."):
                    result = extract_text_from_image_google_vision(image_bytes)
                    if not result.get('success') and AZURE_VISION_KEY:
                        result = extract_text_from_image_azure_vision(image_bytes)
                    if result.get('success'):
                        full_text = result['full_text']
                        st.text_area("Extracted Text:", value=full_text, height=200, key="image_text_area")
                        if st.button("Analyze Extracted Text", key="analyze_image_text_btn", type="primary"):
                            if not openai.api_key:
                                st.error("OpenAI API key not configured. Please add OPENAI_API_KEY to your .env file.")
                            else:
                                with st.spinner("Analyzing extracted text with GPT-4..."):
                                    try:
                                        summary_prompt = get_prompt(category, 'summary')
                                        critique_prompt = get_prompt(category, 'critique')
                                        improve_prompt = get_prompt(category, 'improve')
                                        # Summary
                                        summary_response = openai.ChatCompletion.create(
                                            model="gpt-4",
                                            messages=[
                                                {"role": "system", "content": "You are a professional text analyst specializing in creating concise summaries (max 150 words). Always format your response with proper markdown formatting including headers, bullet points, and emphasis where appropriate. Use ## for main headers, ### for subheaders, and proper bullet points with - for lists. Keep summaries brief and to the point."},
                                                {"role": "user", "content": f"{summary_prompt}\n\nText to summarize:\n{full_text}"}
                                            ],
                                            temperature=0.7,
                                            max_tokens=400
                                        )
                                        summary = summary_response.choices[0].message.content.strip()
                                        # Critique
                                        critique_response = openai.ChatCompletion.create(
                                            model="gpt-4",
                                            messages=[
                                                {"role": "system", "content": "You are a professional text analyst specializing in providing detailed, constructive critiques. Always format your response with proper markdown formatting including headers, bullet points, and emphasis where appropriate. Use ## for main headers, ### for subheaders, and proper bullet points with - for lists. Structure your critique with clear sections like 'Strengths', 'Areas for Improvement', 'Technical Analysis', etc."},
                                                {"role": "user", "content": f"{critique_prompt}\n\nText to critique:\n{full_text}"}
                                            ],
                                            temperature=0.7,
                                            max_tokens=800
                                        )
                                        critique = critique_response.choices[0].message.content.strip()
                                        # Improvements
                                        improve_response = openai.ChatCompletion.create(
                                            model="gpt-4",
                                            messages=[
                                                {"role": "system", "content": "You are a professional text analyst specializing in improving text quality. Always format your response with proper markdown formatting including headers, bullet points, and emphasis where appropriate. Use ## for main headers, ### for subheaders, and proper bullet points with - for lists. Provide specific, actionable improvements with clear examples."},
                                                {"role": "user", "content": f"{improve_prompt}\n\nText to improve:\n{full_text}"}
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
                                        st.error(f"Error: {e}")
                    else:
                        st.error(result.get('error', 'Failed to extract text from image.')) 