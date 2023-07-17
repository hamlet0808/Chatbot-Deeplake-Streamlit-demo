import os
import streamlit as st
import pinecone
import json
#import langchain ### langchain version 0.0.180
#pinecone rate limit

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


class Chatbot:

    def __init__(self, model_name):

        self.model_name = model_name
        #self.openai_key = "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" ## provide your openai key
        self.embeddings = OpenAIEmbeddings( model='text-embedding-ada-002')
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.0, max_tokens=500) 


    def get_vectorstore(self, index_name):
        """
        Load vectorstore of our data from Pinecone index
        """
        pinecone.init(api_key="921d4d1d-ea17-456a-a29b-cb315853a561", ### "921d4d1d-ea17-456a-a29b-cb315853a561",
                  environment="us-west4-gcp-free") ### "us-west4-gcp-free"

        vectorstore = Pinecone.from_existing_index(embedding=self.embeddings,
                                                   index_name=index_name)

        return vectorstore


    def conversational_chat(self):
        """
        Start a conversational chat with a model via Langchain
        """
        system_msg_template = SystemMessagePromptTemplate.from_template(template="""
                      Act as a helpful startup legal assistant. Use provided context to answer the questions.
                      If the question is not related to the context, just say "Hmm, I don't think this question is about startup law.  I can only provide insights on startup law.  Sorry about that!".
                      If the question is about the sources of your context, just say "As an AI language model, I draw upon a large pool of data and don't rely on any one single source."
                      If user asks for clarifying questions about the input, ask questions from your base knowledge, then give the final recommendation.
                      Use bullet points if you have to make a list.
                      Very important: Please provide an answer that is no more than 100 words.
                      Very important: Do Not disclose your sources.
                      Very important: Do Not disclose any names of persons or names of organizations in your responses.
                      """)

        human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
        QA_PROMPT = ChatPromptTemplate.from_messages([MessagesPlaceholder(variable_name="history"), human_msg_template, system_msg_template])


        chain = ConversationChain(llm=self.llm, 
                                  prompt=QA_PROMPT, 
                                  memory=ConversationBufferWindowMemory(
                                                        k=4,
                                                        memory_key="history",
                                                        return_messages=True), 
                                  verbose=True)

        #count_tokens_chain(chain, chain_input)
        return chain


if __name__ == '__main__':

    load_dotenv() ### Loading environment variables such as OpenAI key
    st.write(css, unsafe_allow_html=True)

    chatbot = Chatbot('gpt-3.5-turbo') ### initialize chatbot
    vectorstore = chatbot.get_vectorstore(index_name='blogposts-data') ### initialize vectorstore

    ### Only for streamlit
    if "conversation" not in st.session_state:
        st.session_state.conversation = chatbot.conversational_chat()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if 'timestamp' not in st.session_state:
        st.session_state.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    st.header("Your personal AI assistant")
    user_question = st.text_input("Ask a question:")

    ### Main part
    if user_question:
        context = vectorstore.similarity_search(user_question, k=2) ### getting similar context based on response

        with get_openai_callback() as cb:

            response = st.session_state.conversation.predict(input=f"\n\n Context:\n {context} \n\n question:\n{user_question}")
            if cb.total_tokens > 3000:
                st.session_state.conversation.memory.buffer.pop(0)
            print(f"Count of tokens:", {cb.total_tokens})

        print(len(response.split()))

        ### This part only shows chat history in Streamlit app (we are not going to use in final version)

        st.session_state.chat_history.append({'question': user_question, 'response': response})

        for i in range(len(st.session_state.chat_history)-1 , -1, -1):

            st.write(user_template.replace(
                "{{MSG}}", st.session_state.chat_history[i]['question']), unsafe_allow_html=True)

            st.write(bot_template.replace(
                "{{MSG}}", st.session_state.chat_history[i]['response']), unsafe_allow_html=True)

    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data_{st.session_state.timestamp}.json"

    # with open(filename, 'w') as json_file:
    #     json.dump(st.session_state.chat_history, json_file, ensure_ascii=False, indent=4)
    
    chat_history_json = json.dumps(st.session_state.chat_history, ensure_ascii=False, indent=4)
    st.download_button(label="Download chat history",
                       data=chat_history_json,
                       file_name=filename,
                        mime="application/json")
        

















