import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import SeleniumURLLoader, PyPDFLoader
from langchain.chains import RetrievalQA

load_dotenv()

api_key = os.getenv('GROQ_API_KEY')

def load_llm(model):
    
    # load the llm from Groq
    if model=="Llama3 70B":
        llm = ChatGroq(groq_api_key=api_key, model_name="llama3-70b-8192", temperature=0.1)
    elif model=="Llama3 8B":
        llm = ChatGroq(groq_api_key=api_key, model_name="llama3-8b-8192", temperature=0.1)
    return llm


def load_vector_store(urls):
    loader = SeleniumURLLoader(urls=urls)
    documents = loader.load()

    # load the vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="thenlper/gte-large",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts_chunks = text_splitter.split_documents(documents)

    db = Chroma.from_documents(texts_chunks, embeddings, persist_directory="db")
    return db

def load_prompt_template():
    # prepare the template we will use when prompting the AI
    template = """Use the provided context to answer the user's question.
    If you don't know the answer, respond with "I do not know".

    Context: {context}
    Question: {question}
    Answer:
    """

    prompt = PromptTemplate(
    template=template,
    input_variables=['context', 'question'])

    return prompt 

def create_qa_chain():
    # load the llm, vector store, and the prompt
    llm = load_llm(model)
    db = load_vector_store(urls)
    prompt = load_prompt_template()

    # create the qa_chain
    retriever = db.as_retriever(search_kwargs={'k': 2})
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type='stuff',
                                        retriever=retriever,
                                        return_source_documents=True,
                                        chain_type_kwargs={'prompt': prompt})
    
    return qa_chain

def generate_response(query, qa_chain):
    # use the qa_chain to answer the given query
    return qa_chain({'query':query})['result']

urls = [
    "https://incelligent.net/case-studies/",
    "https://incelligent.net/about-us/",
]

st.title("Support Chatbot")

model = st.sidebar.selectbox(
    "Which LLM do you want to use for this chatbot?",
    ("Llama3 8B", "Llama3 70B"),
    index=None,
    placeholder="Select a model..."
    )

if model:
    qa_chain = create_qa_chain()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = generate_response(prompt, qa_chain)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
