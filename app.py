# Library Imports
import os
import bs4
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.runnables.base import Runnable
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory, StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

api_key = os.getenv('GROQ_API_KEY')

url = 'https://www.nature.com/articles/s41467-020-16278-6'

# Define the LLM to be used
def load_llm():
    llm = ChatGroq(groq_api_key=api_key, model_name="llama3-8b-8192", temperature=0)
    return llm

# Create the vector store that will be used as a retriever
def prepare_retriever():
    # PDF option (commented out for now, but works)
    # loader = PyPDFLoader('pdf\Attention-is-all-you-need.pdf')
    
    # URL scraping option
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            # Will have to be adjusted according to the html classes or tags we want to scrape
            # This works only for the current url
            parse_only=bs4.SoupStrainer(
                class_=("c-article-title", "c-article-section__content")
            )
        ),
    )
    documents = loader.load()
    # Initialize the text splitter object and split the scraped text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts_chunks = text_splitter.split_documents(documents)

    # Specify the embeddings model that will be used to create embeddings from the chunks
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    # Using the chunks and the above embeddings create the vector store
    db = Chroma.from_documents(documents=texts_chunks, embedding=embeddings)
    # Set the vector store to be used as retriever
    retriever = db.as_retriever()
    return retriever

def generate_history_aware_retriever(llm, retriever):
    # Create a prompt that will be used to create a new question based on the chat history and the user's question
    contextualize_q_system_prompt = (
        'Taking into account the chat history and the latest user question that may be referencing the chat history,'
        'generate a new question that can be understood without the chat history. DO NOT answer that question,'
        'just reformulate it if needed and otherwise return it as is.'
        )
    
    # Create a ChatPromptTemplate object that uses the above prompt
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create a retriever that is history-aware using the selected llm, the retreiver and the 
    # ChatPromptTemplate object and return it
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever

# Create the QA chain using the history aware retriever
def create_qa_chain(llm, history_aware_retriever):
    # Define the prompt for the question answering task for the LLM
    system_prompt = (
        'You are a helpful assistant that answers questions.'
        'Use the retrieved context to answer the question.'
        'If you do not know the answer your reply should be "I dont know."'
        'Try to keep the answers short unless otherwise specifed by the question.'
        '\n\n'
        '{context}'
        )
    
    # Use the ChatPromptTemplate object again this time to create a prompt for the question answering task
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create a documents chain for the question answering task
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Integrate the history aware retriever into the QA chain
    qa_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return qa_chain

# Initialize an empty dictionary that will store the session data
if 'store' not in st.session_state:
    st.session_state.store = {}

# Load the LLM
if 'llm' not in st.session_state:
    st.session_state.llm = load_llm()

# Create a retriever from the vector store
if 'retriever' not in st.session_state:
    st.session_state.retriever = prepare_retriever()

# Create a history-aware retriever
if 'history_aware_retriever' not in st.session_state:
    st.session_state.history_aware_retriever = generate_history_aware_retriever(st.session_state.llm, st.session_state.retriever)

# With the chosen LLM and the history aware retriever create the QA chain
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = create_qa_chain(st.session_state.llm, st.session_state.history_aware_retriever)

# Get the chat history from the current session
def get_session_history(session_id: str):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = StreamlitChatMessageHistory()
    return st.session_state.store[session_id]

# Create the QA chain for the user interaction
conversational_rag_chain = RunnableWithMessageHistory(
    st.session_state.rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

############################################################################
# Streamlit Implementation
############################################################################

st.title("Support Chatbot")
st.text("A helpful chatbot that can provide answers from a given context (PDF or web URL).")
st.text(f"Currently answering questions about: \n{url}")

# Initialize chat history in the session state if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": 'How can I help?'})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# When a new question is posed
if prompt := st.chat_input("Your questions here"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate a response using the RAG chain
    response = conversational_rag_chain.invoke(
        {"input": prompt},
        config={
            "configurable": {"session_id": "abc123"}
        }, 
    )["answer"]

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add the assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
