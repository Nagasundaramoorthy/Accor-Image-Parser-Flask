from flask import Flask, request, render_template, session
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
import os
import io
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Configure Gemini API
gemini_key = os.getenv('GEMINI_KEY')
genai.configure(api_key=gemini_key)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Configure Hugging Face embeddings and LLM
embeddings = HuggingFaceEmbeddings()
llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.9, "max_length": 500})

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

# Function to set up RAG (Retrieval-Augmented Generation)
def setup_rag(description):
    vector_store = FAISS.from_texts([description], embeddings)
    retriever = vector_store.as_retriever()

    prompt_template = """
    You are a knowledgeable and friendly assistant. Your task is to provide detailed, accurate, and engaging answers to the user's questions based on the context provided below.
    
    Context: {context}
    Question: {question}
    
    Answer in a conversational tone, ensuring clarity and depth. If the question cannot be answered from the context, politely inform the user.
    """
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs={"prompt": prompt})

# Function to generate a response using Gemini
def hybrid_response(question, context):
    response = gemini_model.generate_content([
        f"Context: {context}\n\nAnswer the following question in detail and clear and short: {question}"
    ])
    return response.text

@app.route('/', methods=['GET', 'POST'])
def index():
    description = None
    chat_history = session.get('chat_history', [])
    response_text = None

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
            qa_chain = setup_rag(session['description'])
            rag_response = qa_chain.run(user_query)
            response_text = hybrid_response(user_query, rag_response)
            chat_history.append((user_query, response_text))
            session['chat_history'] = chat_history

    return render_template('index.html', description=description, chat_history=chat_history, response_text=response_text)

if __name__ == '__main__':
    app.run(debug=True)
