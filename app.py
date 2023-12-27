import streamlit as st
from utilities import *


if st.session_state['HuggingFace_API_Key']!='' and st.session_state['Pinecone_API_Key']!='':

    # Get post data
    st.write('Getting post data...')
    post_data = get_post_data('https://pathologia.eu/post-sitemap.xml')
    st.write('Post data acquired.')

    # Split data into chunks
    st.write('Splitting data into smaller chunks...')
    chunks = split_data(post_data)
    st.write('Splitting complete.')

    # Create embeddings
    embeddings = create_embeddings()
    st.write('Created an embeddings instance.')