import pinecone
import openai

from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone, DeepLake
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain


class Chatbot:

    def __init__(self, model_name):

        self.model_name = model_name
        self.embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.0, max_tokens=500)
        self.memory = ConversationBufferWindowMemory(k=1, return_messages=True)
        self.chat_history = []

    def get_vectorstore(self, key, env, index_name):
        """
        Load vectorstore of our data from Pinecone index
        """
        pinecone.init(api_key=key,
                      environment=env)

        vectorstore = Pinecone.from_existing_index(
                                embedding=self.embeddings,
                                index_name=index_name)

        return vectorstore

    def get_vectorstore_deeplake(self, path):
        """
        Load vectorstore of our data from Deeplake path
        """
        vectorstore_dl = DeepLake(
                    dataset_path=path, embedding_function=self.embeddings, read_only=True
            )

        return vectorstore_dl

    def query_refiner(self, query, api_key):

        conversation = self.chat_history[-2:]
        #print("conversation", conversation)
        openai.api_key = api_key
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
                      If the question is not related to the context, just say "Hmm, I don't think this question is about startup law. I can only provide insights on startup law.  Sorry about that!".
                      If the question is about the sources of your context, just say "As an AI language model, I draw upon a large pool of data and don't rely on any one single source."
                      Use bullet points if you have to make a list.
                      Very important: Do Not disclose your sources.
                      Very important: Do Not disclose any names of persons or names of organizations in your response.
                      Greeting: If user says his name address them by name.
                      Introduction: Say your name and introduce yourself as a startup legal assistant.
                      """)

        human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
        QA_PROMPT = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

        chain = ConversationChain(llm=self.llm, 
                                  prompt=QA_PROMPT,
                                  memory=self.memory,
                                  verbose=True)

        return chain