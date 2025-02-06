from flask import Flask, request, session,jsonify
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
import os
import io

# Load environment variables
load_dotenv()

# Configure Gemini API
gemini_key = os.getenv('GEMINI_KEY')
genai.configure(api_key=gemini_key)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

# Function to generate a concise image description
def gemini_response(img):
    response = gemini_model.generate_content([
        "Provide a concise and structured description of the image in 5 bullet points. Focus on key elements, colors, objects, and any notable details. If there are any products, mention them specifically. Avoid unnecessary text.",
        img
    ])
    return response.text

# Function to generate a detailed image description
def gemini_response_desc(img):
    response = gemini_model.generate_content([
        "Describe the image in detail, covering all visible elements, their relationships, colors, textures, and any potential context or story the image might convey. If there are any products, provide detailed information about them including brand, type, and any visible features. Be thorough and engaging.",
        img
    ])
    return response.text

# Function to generate a response using Gemini
def generate_response(question, context):
    response = gemini_model.generate_content([
        f"Context: {context}\n\nAnswer the following question in a detailed yet concise manner: {question}"
    ])
    return response.text

@app.route('/', methods=['GET', 'POST'])
def index():
    description = None
    chat_history = session.get('chat_history', [])
    response_text = None
    explanation_2 = None  # Ensure it is defined
    output = {}  # Initialize output to avoid UnboundLocalError

    if request.method == 'POST':
        if 'image' in request.files:
            image_file = request.files['image']
            if image_file.filename != '':
                image = Image.open(io.BytesIO(image_file.read()))
                explanation = gemini_response(image)
                explanation_2 = gemini_response_desc(image)
                session['description'] = explanation_2
                description = explanation

        if 'question' in request.form and session.get('description'):
            user_query = request.form['question']
            context = session['description']
            response_text = generate_response(user_query, context)
            chat_history.append((user_query, response_text))
            session['chat_history'] = chat_history

    output = {
        'description': description,
        'detail_description': explanation_2,
        'chat_history': chat_history,
        'response_text': response_text
    }

    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=False)
