import pinecone
import openai

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


class Chatbot:

    def __init__(self, model_name):

        self.model_name = model_name
        self.embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.0, max_tokens=500)
        self.memory = ConversationBufferWindowMemory(k=3, return_messages=True)
        self.chat_history = []

    def get_vectorstore(self, key, env, index_name):
        """
        Load vectorstore of our data from Pinecone index
        """
        pinecone.init(api_key=key, ### "921d4d1d-ea17-456a-a29b-cb315853a561",
                      environment=env) ### "us-west4-gcp-free"

        vectorstore = Pinecone.from_existing_index(embedding=self.embeddings,
                                                   index_name=index_name)

        return vectorstore

    def query_refiner(self, query, api_key):

        conversation = self.chat_history[-2:]
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
                                  verbose=False)

        #count_tokens_chain(chain, chain_input)
        return chain