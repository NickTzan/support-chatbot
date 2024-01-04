import chromadb
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# Load the pdf and split text into chunks
loader = PyPDFLoader(pdf_path)
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = splitter.split_documents(documents)

# Create an embeddings instance
embeddings = SentenceTransformerEmbeddings(model_name = 'all-MiniLM-L6-v2')

# Create vectorstore
client = chromadb.PersistentClient(path="./vectorstore")

