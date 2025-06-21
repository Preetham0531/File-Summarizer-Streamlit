from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
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
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

# Configure Google Cloud Vision API (alternative: Azure Computer Vision)
GOOGLE_CLOUD_API_KEY = os.getenv('GOOGLE_CLOUD_API_KEY')
AZURE_VISION_KEY = os.getenv('AZURE_VISION_KEY')
AZURE_VISION_ENDPOINT = os.getenv('AZURE_VISION_ENDPOINT')

# Check if OCR services are available
OCR_AVAILABLE = bool(GOOGLE_CLOUD_API_KEY or AZURE_VISION_KEY)

# Configure file upload
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'csv', 'xlsx', 'pptx', 'md', 'json', 'html'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def extract_text_from_image_google_vision(image_data):
    """Extract text from image using Google Cloud Vision API"""
    try:
        if not GOOGLE_CLOUD_API_KEY:
            return {"error": "Google Cloud Vision API key not configured. Please add GOOGLE_CLOUD_API_KEY to your .env file.", "success": False}
        
        # Encode image to base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Prepare request
        url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_CLOUD_API_KEY}"
        payload = {
            "requests": [
                {
                    "image": {
                        "content": image_base64
                    },
                    "features": [
                        {
                            "type": "TEXT_DETECTION"
                        }
                    ]
                }
            ]
        }
        
        response = requests.post(url, json=payload)
        result = response.json()
        
        if 'responses' in result and result['responses'][0].get('textAnnotations'):
            # Extract all text
            text_annotations = result['responses'][0]['textAnnotations']
            full_text = text_annotations[0]['description'] if text_annotations else ""
            
            # Extract individual words with coordinates
            words = []
            for annotation in text_annotations[1:]:  # Skip the first one (full text)
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
            
            return {
                'full_text': full_text,
                'words': words,
                'success': True
            }
        else:
            return {"error": "No text found in image", "success": False}
            
    except Exception as e:
        return {"error": f"Error extracting text: {str(e)}", "success": False}

def extract_text_from_image_azure_vision(image_data):
    """Extract text from image using Azure Computer Vision API"""
    try:
        if not AZURE_VISION_KEY or not AZURE_VISION_ENDPOINT:
            return {"error": "Azure Computer Vision API not configured. Please add AZURE_VISION_KEY and AZURE_VISION_ENDPOINT to your .env file.", "success": False}
        
        # Prepare request
        url = f"{AZURE_VISION_ENDPOINT}/vision/v3.2/read/analyze"
        headers = {
            'Ocp-Apim-Subscription-Key': AZURE_VISION_KEY,
            'Content-Type': 'application/octet-stream'
        }
        
        response = requests.post(url, headers=headers, data=image_data)
        
        if response.status_code == 202:
            # Get operation location
            operation_url = response.headers['Operation-Location']
            
            # Poll for results
            import time
            for _ in range(10):  # Try up to 10 times
                time.sleep(1)
                result_response = requests.get(operation_url, headers={'Ocp-Apim-Subscription-Key': AZURE_VISION_KEY})
                
                if result_response.status_code == 200:
                    result = result_response.json()
                    
                    if result['status'] == 'succeeded':
                        # Extract text and coordinates
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
                        
                        return {
                            'full_text': full_text,
                            'words': words,
                            'success': True
                        }
            
            return {"error": "Timeout waiting for OCR results", "success": False}
        else:
            return {"error": f"API request failed: {response.status_code}", "success": False}
            
    except Exception as e:
        return {"error": f"Error extracting text: {str(e)}", "success": False}

def extract_text_from_selected_area(words, selection_bounds):
    """Extract text from a specific area based on coordinates"""
    selected_text = []
    
    for word in words:
        word_bounds = word['bounds']
        
        # Check if word is within selection bounds
        if (word_bounds['x1'] >= selection_bounds['x1'] and 
            word_bounds['x2'] <= selection_bounds['x2'] and
            word_bounds['y1'] >= selection_bounds['y1'] and 
            word_bounds['y2'] <= selection_bounds['y2']):
            selected_text.append(word['text'])
    
    return ' '.join(selected_text)

# Streamlined category-specific prompts with distinctive strategies
CATEGORY_PROMPTS = {
    'academic': {
        'summary': """Create a concise academic summary (max 150 words) covering:
• Main research objective
• Key findings
• Primary conclusion

Keep it brief and to the point.""",
        'critique': """Provide a detailed academic critique covering:
• **Strengths**: Methodology, evidence, clarity
• **Areas for Improvement**: Gaps, limitations, suggestions
• **Technical Analysis**: Research design, statistical methods
• **Impact Assessment**: Contribution to field, practical implications""",
        'improve': """Suggest specific improvements for:
• **Content Enhancement**: Additional research, better evidence
• **Structural Improvements**: Organization, flow, clarity
• **Methodological Refinements**: Better approaches, stronger analysis
• **Presentation**: Formatting, citations, accessibility"""
    },
    'professional': {
        'summary': """Create a concise professional summary (max 150 words) highlighting:
• Main objective
• Key outcomes
• Primary recommendation

Keep it brief and actionable.""",
        'critique': """Provide a professional critique covering:
• **Business Value**: ROI, strategic alignment, market relevance
• **Technical Quality**: Accuracy, completeness, feasibility
• **Communication**: Clarity, persuasiveness, audience appropriateness
• **Implementation**: Practicality, timeline, resource requirements""",
        'improve': """Suggest professional improvements for:
• **Business Impact**: Enhanced value proposition, better metrics
• **Technical Excellence**: More robust analysis, better data
• **Communication**: Clearer messaging, better presentation
• **Execution**: More actionable recommendations, better planning"""
    },
    'media': {
        'summary': """Create a concise media summary (max 150 words) covering:
• Main story
• Key points
• Impact

Keep it brief and engaging.""",
        'critique': """Provide a media critique covering:
• **Content Quality**: Accuracy, balance, newsworthiness
• **Presentation**: Writing style, visual appeal, engagement
• **Impact**: Reach, influence, public understanding
• **Ethics**: Fairness, bias, responsible reporting""",
        'improve': """Suggest media improvements for:
• **Content Enhancement**: Better sources, deeper analysis
• **Presentation**: More engaging format, better visuals
• **Impact**: Wider reach, stronger engagement
• **Credibility**: Better fact-checking, more balanced coverage"""
    },
    'marketing': {
        'summary': """Create a concise marketing summary (max 150 words) highlighting:
• Target audience
• Key message
• Main outcome

Keep it brief and focused.""",
        'critique': """Provide a marketing critique covering:
• **Strategy**: Target audience fit, positioning, competitive advantage
• **Execution**: Creative quality, channel effectiveness, timing
• **Performance**: Metrics, ROI, conversion rates
• **Brand Alignment**: Consistency, authenticity, long-term impact""",
        'improve': """Suggest marketing improvements for:
• **Strategy**: Better targeting, stronger positioning
• **Creative**: More compelling messaging, better visuals
• **Execution**: Optimized channels, better timing
• **Measurement**: Better metrics, improved tracking"""
    }
}

def get_prompt(category, prompt_type):
    """Get the appropriate prompt for the given category and type."""
    if category in CATEGORY_PROMPTS:
        return CATEGORY_PROMPTS[category][prompt_type]
    return CATEGORY_PROMPTS['academic'][prompt_type]  # Default to academic prompts

def extract_text_from_file(file_path, file_extension):
    """Extract text content from various file types."""
    try:
        if file_extension == 'txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        
        elif file_extension == 'pdf':
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        
        elif file_extension == 'docx':
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        
        elif file_extension == 'csv':
            df = pd.read_csv(file_path)
            return df.to_string()
        
        elif file_extension == 'xlsx':
            df = pd.read_excel(file_path)
            return df.to_string()
        
        elif file_extension == 'md':
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        
        elif file_extension == 'json':
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                return json.dumps(data, indent=2)
        
        elif file_extension == 'html':
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # Remove HTML tags for text analysis
                clean_text = re.sub(r'<[^>]+>', '', content)
                return clean_text
        
        else:
            return "Unsupported file type"
    
    except Exception as e:
        return f"Error extracting text: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html', ocr_available=OCR_AVAILABLE)

@app.route('/api-status')
def api_status():
    """Check API key status"""
    return jsonify({
        'openai_configured': bool(openai.api_key),
        'ocr_available': OCR_AVAILABLE,
        'google_vision_configured': bool(GOOGLE_CLOUD_API_KEY),
        'azure_vision_configured': bool(AZURE_VISION_KEY and AZURE_VISION_ENDPOINT)
    })

@app.route('/extract-text-from-image', methods=['POST'])
def extract_text_from_image():
    """Extract text from uploaded image"""
    if not OCR_AVAILABLE:
        return jsonify({'error': 'OCR services not configured. Please add either GOOGLE_CLOUD_API_KEY or AZURE_VISION_KEY to your .env file.'}), 400
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    if file and allowed_image_file(file.filename):
        try:
            # Read image data
            image_data = file.read()
            
            # Try Google Cloud Vision first, then Azure
            result = extract_text_from_image_google_vision(image_data)
            
            if not result.get('success') and AZURE_VISION_KEY:
                result = extract_text_from_image_azure_vision(image_data)
            
            if result.get('success'):
                return jsonify(result)
            else:
                return jsonify({'error': result.get('error', 'Failed to extract text')}), 500
                
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid image file type'}), 400

@app.route('/analyze-image-selection', methods=['POST'])
def analyze_image_selection():
    """Analyze text from a specific area of an image"""
    if not OCR_AVAILABLE:
        return jsonify({'error': 'OCR services not configured. Please add either GOOGLE_CLOUD_API_KEY or AZURE_VISION_KEY to your .env file.'}), 400
    
    data = request.get_json()
    image_data = data.get('image_data')  # Base64 encoded image
    selection_bounds = data.get('selection_bounds')
    category = data.get('category', 'academic')
    
    if not image_data or not selection_bounds:
        return jsonify({'error': 'Missing image data or selection bounds'}), 400
    
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        
        # Extract text from image
        ocr_result = extract_text_from_image_google_vision(image_bytes)
        
        if not ocr_result.get('success') and AZURE_VISION_KEY:
            ocr_result = extract_text_from_image_azure_vision(image_bytes)
        
        if not ocr_result.get('success'):
            return jsonify({'error': ocr_result.get('error', 'Failed to extract text')}), 500
        
        # Extract text from selected area
        selected_text = extract_text_from_selected_area(ocr_result['words'], selection_bounds)
        
        if not selected_text.strip():
            return jsonify({'error': 'No text found in selected area'}), 400
        
        # Analyze the selected text
        return analyze_text_content(selected_text, category)
        
    except Exception as e:
        return jsonify({'error': f'Error analyzing image selection: {str(e)}'}), 500

@app.route('/summarize', methods=['POST'])
def summarize():
    if not openai.api_key:
        return jsonify({'error': 'OpenAI API key not configured. Please add OPENAI_API_KEY to your .env file.'}), 400
    
    data = request.get_json()
    text = data.get('text')
    category = data.get('category', 'academic')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        # Get the summary prompt for the category
        summary_prompt = get_prompt(category, 'summary')
        
        # Generate summary
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
        return jsonify({'summary': summary})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    if not openai.api_key:
        return jsonify({'error': 'OpenAI API key not configured. Please add OPENAI_API_KEY to your .env file.'}), 400
    
    data = request.get_json()
    text = data.get('text')
    category = data.get('category', 'academic')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        # Get prompts for the category
        summary_prompt = get_prompt(category, 'summary')
        critique_prompt = get_prompt(category, 'critique')
        improve_prompt = get_prompt(category, 'improve')
        
        # Generate summary
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
        
        # Generate critique
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
        
        # Generate improvements
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
        
        return jsonify({
            'summary': summary,
            'critique': critique,
            'improved_summary': improved_summary
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-file', methods=['POST'])
def analyze_file():
    if not openai.api_key:
        return jsonify({'error': 'OpenAI API key not configured. Please add OPENAI_API_KEY to your .env file.'}), 400
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    category = request.form.get('category', 'academic')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Extract text from file
            file_extension = filename.rsplit('.', 1)[1].lower()
            text_content = extract_text_from_file(file_path, file_extension)
            
            if text_content.startswith("Error"):
                return jsonify({'error': text_content}), 500
            
            # Clean up the uploaded file
            os.remove(file_path)
            
            # Analyze the extracted text
            return analyze_text_content(text_content, category)
            
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/summarize-file', methods=['POST'])
def summarize_file():
    if not openai.api_key:
        return jsonify({'error': 'OpenAI API key not configured. Please add OPENAI_API_KEY to your .env file.'}), 400
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    category = request.form.get('category', 'academic')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Extract text from file
            file_extension = filename.rsplit('.', 1)[1].lower()
            text_content = extract_text_from_file(file_path, file_extension)
            
            if text_content.startswith("Error"):
                return jsonify({'error': text_content}), 500
            
            # Clean up the uploaded file
            os.remove(file_path)
            
            # Summarize the extracted text
            summary_prompt = get_prompt(category, 'summary')
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a professional text analyst specializing in creating concise summaries (max 150 words). Always format your response with proper markdown formatting including headers, bullet points, and emphasis where appropriate. Use ## for main headers, ### for subheaders, and proper bullet points with - for lists. Keep summaries brief and to the point."},
                    {"role": "user", "content": f"{summary_prompt}\n\nText to summarize:\n{text_content}"}
                ],
                temperature=0.7,
                max_tokens=400
            )
            
            summary = response.choices[0].message.content.strip()
            return jsonify({'summary': summary})
            
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

def analyze_text_content(text_content, category):
    """Helper function to analyze text content."""
    try:
        # Get prompts for the category
        summary_prompt = get_prompt(category, 'summary')
        critique_prompt = get_prompt(category, 'critique')
        improve_prompt = get_prompt(category, 'improve')
        
        # Generate summary
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
        
        # Generate critique
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
        
        # Generate improvements
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
        
        return jsonify({
            'summary': summary,
            'critique': critique,
            'improved_summary': improved_summary
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5005) 