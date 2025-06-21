# Essay Critic - Intelligent Text Analysis Platform

A comprehensive text analysis application that provides intelligent summaries, critiques, and improvements for various types of content using AI-powered analysis.

## 🚀 Features

### 📝 Text Analysis
- **Streamlined Categories**: Academic, Professional Documents, Media & Content, Product & Marketing
- **Specialized Analysis**: Each category has unique summary, critique, and improvement strategies
- **Real-time Search**: Quick category discovery with smart filtering
- **Markdown Formatting**: Professional, structured output with headers and bullet points

### 📁 File Analysis
- **Multiple Formats**: Support for .txt, .docx, .pdf, .csv, .xlsx, .pptx, .md, .json, .html
- **Drag & Drop**: Easy file upload with visual feedback
- **Automatic Text Extraction**: Seamless processing of various file types
- **Category-Specific Analysis**: Tailored insights based on content type

### 🖼️ Image Analysis (NEW!)
- **OCR Integration**: Extract text from images using Google Cloud Vision or Azure Computer Vision
- **Drag & Drop Selection**: Click and drag to select specific areas of text in images
- **Coordinate-Based Analysis**: Analyze only the text within your selected area
- **Multiple Image Formats**: Support for PNG, JPG, JPEG, GIF, BMP, TIFF, WebP
- **Full Image Analysis**: Option to analyze all text in the image

## 🛠️ Setup Instructions

### 1. Prerequisites
- Python 3.8 or higher
- pip package manager

### 2. Installation

```bash
# Clone the repository
git clone <repository-url>
cd Project_Essay_Critic

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. API Keys Setup

Create a `.env` file in the project root with the following API keys:

```env
# OpenAI API Key (Required for text analysis)
OPENAI_API_KEY=your_openai_api_key_here

# Google Cloud Vision API Key (Optional - for image OCR)
GOOGLE_CLOUD_API_KEY=your_google_cloud_vision_api_key_here

# Azure Computer Vision API (Optional - alternative to Google Cloud Vision)
AZURE_VISION_KEY=your_azure_vision_key_here
AZURE_VISION_ENDPOINT=https://your-resource-name.cognitiveservices.azure.com/
```

#### Getting API Keys:

**OpenAI API Key:**
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Create an account or sign in
3. Go to API Keys section
4. Create a new API key
5. Copy and paste into your `.env` file

**Google Cloud Vision API Key (Optional):**
1. Visit [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the Cloud Vision API
4. Go to Credentials → Create Credentials → API Key
5. Copy and paste into your `.env` file

**Azure Computer Vision API (Optional):**
1. Visit [Azure Portal](https://portal.azure.com/)
2. Create a new Computer Vision resource
3. Go to Keys and Endpoint
4. Copy Key 1 and Endpoint URL
5. Paste into your `.env` file

### 4. Run the Application

```bash
python app.py
```

The application will be available at `http://127.0.0.1:5000`

## 🎯 How to Use

### Text Analysis
1. Click "📝 Text Analysis"
2. Select a category (Academic, Professional, Media, or Marketing)
3. Enter your text in the textarea
4. Choose "Analyze Text" for full analysis or "Summarize Only"
5. View structured results with summary, critique, and improvements

### File Analysis
1. Click "📁 File Analysis"
2. Drag and drop a file or click to browse
3. Select the appropriate content category
4. Choose "Analyze File" or "Summarize Only"
5. Get specialized analysis based on file content

### Image Analysis
1. Click "🖼️ Image Analysis"
2. Upload an image (drag & drop or browse)
3. **For Selection Analysis:**
   - Click and drag to create a selection box around text
   - Select a category
   - Click "Analyze Selection"
4. **For Full Image Analysis:**
   - Select a category
   - Click "Analyze Full Image"
5. View AI-powered analysis of the extracted text

## 🔧 Technical Details

### Backend Technologies
- **Flask**: Web framework
- **OpenAI GPT-4**: Text analysis and generation
- **Google Cloud Vision/Azure Computer Vision**: OCR for images
- **PyPDF2**: PDF text extraction
- **python-docx**: Word document processing
- **pandas**: Data file processing

### Frontend Technologies
- **HTML5 Canvas**: Image selection interface
- **Tailwind CSS**: Modern, responsive design
- **JavaScript**: Interactive functionality
- **Axios**: HTTP client for API calls

### Supported File Types
- **Text**: .txt, .md, .html
- **Documents**: .docx, .pdf
- **Data**: .csv, .xlsx
- **Presentations**: .pptx
- **Structured**: .json
- **Images**: .png, .jpg, .jpeg, .gif, .bmp, .tiff, .webp

## 🎨 Features Overview

### Category-Specific Analysis

**🎓 Academic**
- Research papers, essays, literature
- Focus on methodology, arguments, findings
- Academic tone and structured analysis

**💼 Professional Documents**
- Business reports, legal docs, technical docs
- Emphasis on clarity, action items, implications
- Professional standards and effectiveness

**📰 Media & Content**
- News articles, blogs, reviews, surveys
- Storytelling quality, engagement, objectivity
- Audience-focused improvements

**🎯 Product & Marketing**
- Product descriptions, marketing copy
- Value proposition, persuasion, conversion
- Brand alignment and competitive advantage

### Image Analysis Capabilities
- **OCR Technology**: Advanced text recognition
- **Selection Interface**: Intuitive drag-and-drop selection
- **Coordinate Mapping**: Precise text extraction from selected areas
- **Multiple OCR Providers**: Google Cloud Vision and Azure Computer Vision support
- **Real-time Preview**: Visual feedback during selection

## 🔒 Security & Privacy
- All API keys are stored securely in environment variables
- No data is permanently stored on the server
- Files are processed temporarily and deleted after analysis
- HTTPS support for production deployment

## 🚀 Deployment
For production deployment, consider:
- Using a production WSGI server (Gunicorn)
- Setting up HTTPS with SSL certificates
- Implementing rate limiting
- Adding user authentication if needed
- Using environment-specific configuration

## 🤝 Contributing
Feel free to submit issues, feature requests, or pull requests to improve the application.

## 📄 License
This project is open source and available under the MIT License. 