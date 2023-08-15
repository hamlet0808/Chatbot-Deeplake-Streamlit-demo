import os

from langchain.callbacks import get_openai_callback
from chatbot import Chatbot
from dotenv import load_dotenv
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

load_dotenv()
API_KEY = os.environ['OPENAI_API_KEY']
PINECONE_KEY = os.environ['PINECONE_API_KEY']
PINECONE_ENV = os.environ['PINECONE_ENV']

DeepLake_path = '../my_deeplake_new/'

chatbot = Chatbot('gpt-3.5-turbo')

@app.route('/')
def index():
    return render_template('index.html')

def get_answer(user_question, pine=True):
    
    if pine:
        vectorstore = chatbot.get_vectorstore(PINECONE_KEY, PINECONE_ENV, index_name='additional-blogposts-data') # initialize vectorstore
    else:
        vectorstore = chatbot.get_vectorstore_deeplake(DeepLake_path)

    if len(chatbot.chat_history) > 0:
        refined_query = chatbot.query_refiner(user_question, API_KEY)
        print('refined_query:', refined_query)
        context = vectorstore.similarity_search(refined_query, k=2)
    else:
        context = vectorstore.similarity_search(user_question, k=2)

    conversation = chatbot.conversational_chat()

    with get_openai_callback() as cb:
        response = conversation.predict(input=f" \n\n Context:\n {context} \n\n question:\n {user_question}\n (You MUST provide an answer that is no more than 100 words.)")
        if cb.total_tokens > 3000:
            chatbot.memory.buffer.pop(0)
            
    chatbot.chat_history.append({'question': user_question, 'response': response})
    #print("chatbot_history", chatbot.chat_history)
    return response

@app.route('/data', methods=['POST'])
def get_data():
    
    data = request.get_json()
    user_input = data.get('data')
    print(user_input)
    
    try:
        model_reply = get_answer(user_input, pine=False)
        print("model_reply", model_reply)
        print(len(model_reply.split()))
        return jsonify({"response": True, "message": model_reply})
    
    except Exception as e:
        print(e)
        error_message = f'Error: {str(e)}'
        return jsonify({"message": error_message, "response": False})


if __name__ == '__main__':
    app.run(port=5001)