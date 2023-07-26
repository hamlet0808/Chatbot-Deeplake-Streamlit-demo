import os
import streamlit as st
import openai
import json

from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
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
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.0, max_tokens=500)
        self.embeddings_trans = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    def get_vec_faiss(self,index_name):
        """
        Load faiss vectorstore
        """

        new_db = FAISS.load_local(index_name, self.embeddings_trans)

        return new_db
    
    def query_refiner(self,conversation, query):
        #print("num of words:", len(conversation.split()))

        response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
                    temperature=0.0,
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
                                  memory=st.session_state.buffer_memory,
                                  verbose=True)

        #count_tokens_chain(chain, chain_input)
        return chain


if __name__ == '__main__':

    load_dotenv() ### Loading environment variables such as OpenAI key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    st.write(css, unsafe_allow_html=True)

    if 'buffer_memory' not in st.session_state:
        st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

    chatbot = Chatbot('gpt-3.5-turbo') ### initialize chatbot
    #vectorstore = chatbot.get_vectorstore(index_name='blogposts-data') ### initialize vectorstore
    vectorstore_faiss = chatbot.get_vec_faiss(index_name='faiss_index')

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
        refined_query = chatbot.query_refiner(st.session_state.chat_history[-2:], user_question)
        #print("Refined Query:", refined_query)
        # st.subheader("Refined Query:")
        # st.write(refined_query)
        #context = vectorstore.similarity_search(refined_query, k=2) ### getting similar context based on response
        context = vectorstore_faiss.similarity_search(refined_query, k=2)
        
        with get_openai_callback() as cb:
            response = st.session_state.conversation.predict(input=f"\n\n Context:\n {context} \n\n question:\n{user_question} (You MUST provide an answer that is between 100 and 150 words)")

            if cb.total_tokens > 3000:
                st.session_state.conversation.memory.buffer.pop(0)

            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost (USD): ${cb.total_cost}")

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
    
    chat_history_json = json.dumps(st.session_state.chat_history, ensure_ascii=False, indent=4)
    st.download_button(label="Download chat history",
                       data=chat_history_json,
                       file_name=filename,
                       mime="application/json")

        

















