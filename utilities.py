import asyncio
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings


# Get data from the website's posts
def get_post_data(sitemap_url):

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loader = SitemapLoader(sitemap_url)
    docs = loader.load()
    return docs

# Split post data into chunks
def split_data(docs):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    docs_chunks = text_splitter.split_documents(docs)
    return docs_chunks

# Create the embeddings instance
def create_embeddings():

    embeddings = SentenceTransformerEmbeddings(model_name = 'all-MiniLM-L6-v2')
    return embeddings
