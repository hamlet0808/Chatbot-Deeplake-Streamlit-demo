import os
import streamlit as st
import pinecone
import openai
import json
import requests

from langchain.prompts import PromptTemplate
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.callbacks import get_openai_callback

from htmlTemplates import css, bot_template, user_template
from datetime import datetime
from dotenv import load_dotenv

from flask import Flask, render_template,jsonify,request
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

load_dotenv()
API_KEY = os.environ['OPENAI_API_KEY']
PINECONE_KEY = os.environ['PINECONE_API_KEY']
PINECONE_ENV = os.environ['PINECONE_ENV']


class Chatbot:

    def __init__(self, model_name):

        self.model_name = model_name
        self.embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.0, max_tokens=500)
        self.memory = ConversationBufferWindowMemory(k=3,return_messages=True)
        self.chat_history = []

    def get_vectorstore(self, index_name):
        """
        Load vectorstore of our data from Pinecone index
        """
        pinecone.init(api_key=PINECONE_KEY, ### "921d4d1d-ea17-456a-a29b-cb315853a561",
                      environment=PINECONE_ENV) ### "us-west4-gcp-free"

        vectorstore = Pinecone.from_existing_index(embedding=self.embeddings,
                                                   index_name=index_name)

        return vectorstore


    def query_refiner(self, query):

        print(self.chat_history[-2:])
        openai.api_key = API_KEY
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{self.chat_history[-2:]}\n\nQuery: {query}\n\nRefined Query:",
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        response_text = response['choices'][0]['text']

        return response_text


    def conversational_chat(self):
        """
        Start a conversational chat with a model via Langchain
        """
        system_msg_template = SystemMessagePromptTemplate.from_template(template="""
                      Act as a helpful startup legal assistant. Your name is Jessica. Use provided context to answer the questions.
                      If the question is not related to the context, just say "Hmm, I don't think this question is about startup law.  I can only provide insights on startup law.  Sorry about that!".
                      If the question is about the sources of your context, just say "As an AI language model, I draw upon a large pool of data and don't rely on any one single source."
                      Use bullet points if you have to make a list.
                      Very important: Do Not disclose your sources.
                      Very important: Do Not disclose any names of persons or names of organizations in your responses.
                      """)

        human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
        QA_PROMPT = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

        chain = ConversationChain(llm=self.llm, 
                                  prompt=QA_PROMPT, 
                                  memory=self.memory, 
                                  verbose=True)

        #count_tokens_chain(chain, chain_input)
        return chain

chatbot = Chatbot('gpt-3.5-turbo')

@app.route('/')
def index():
    return render_template('index.html')


def get_answer(user_question):
    
    vectorstore = chatbot.get_vectorstore(index_name='blogposts-data') # initialize vectorstore
    refined_query = chatbot.query_refiner(user_question)
    print('refined_query:', refined_query)
    context = vectorstore.similarity_search(refined_query, k=2)
    
    with get_openai_callback() as cb:
        response = chatbot.conversational_chat().predict(input=f"\n\n Context:\n {context} \n\n question:\n{user_question} (Very important: Please provide an answer that is no more than 100 words.)")
        if cb.total_tokens > 2000:
            chatbot.memory.buffer.pop(0)
            
    chatbot.chat_history.append({'question': user_question, 'response': response})
    return response

@app.route('/data', methods=['POST'])
def get_data():
    
    data = request.get_json()
    user_input = data.get('data')
    print(user_input)
    
    try:
        model_reply = get_answer(user_input)
        print("model_reply", model_reply)
        return jsonify({"response":True,"message":model_reply})
    
    except Exception as e:
        print(e)
        error_message = f'Error: {str(e)}'
        return jsonify({"message":error_message,"response":False})


if __name__ == '__main__':
    app.run()