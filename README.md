# Customer Support Chatbot - MSc AI Thesis by Nick Tzanis

This repository contains the implementation of a customer support chatbot developed as part of my MSc thesis in Artificial Intelligence. The chatbot is designed to deliver accurate, context-aware responses by integrating large language models (LLMs) and various AI tools. Below is an overview of the project’s features and the methodologies employed.

## Project Overview

### Chatbot Development

- **Retrieval-Augmented Generation (RAG):** A RAG system was implemented to enhance the chatbot’s ability to use context for generating accurate answers. This system allows the chatbot to retrieve relevant information before generating a response, improving overall accuracy.

- **API Platform Transition:** The API platform was switched from HuggingFace to Groq, optimizing for faster inference times with large LLMs. The current iteration utilizes Meta's Llama 3 70b for efficient and reliable response generation.
  
- **Context Sourcing:** The chatbot can source context from multiple inputs:
  - **Web Scraping:** Using BeautifulSoup, the chatbot can scrape information from specified websites.
  - **Local PDFs:** Alternatively, it can pull context from PDF files stored in a local directory.

- **Embedding Model:** The project integrates the all-mpnet-base-v2 embedding model (from HuggingFace) to convert contextual data into embeddings, which are then used to create a vector store for efficient information retrieval.

- **User Interface:** A Streamlit-based UI was developed to allow users to interact with the chatbot. The interface also supports conversational memory, enabling more natural and continuous interactions.

### Model Evaluation

- **Evaluation Dataset:** A custom dataset was created specifically for evaluating the chatbot's performance across different LLMs.

- **Automated Evaluation Workflow:** The `create_eval_sets.ipynb` notebook saves each model's answers to evaluation questions, streamlining the comparison process.

- **Performance Analysis:** The `evaluate.ipynb` notebook leverages the RAGAS library to assess and compare the performance of five different models: Llama 8b, Llama 70b, Mixtral 8x7b, Gemma 7b, and Gemma2 9b. Evaluation results are presented using visualizations generated with Matplotlib.
