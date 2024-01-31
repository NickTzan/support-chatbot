# Support Chatbot for Customer Question Answering
------------
## Nick Tzanis Thesis for AI MSc

### Ready:
1. Implemented API call of an open source model (in this case Mistral-7B) through HuggingFace for the question answering.
2. Basic quantization to reduce the memory and computation requirements (only 4 bits used for the model parameters).
3. Set up the tokenizer and the text generation parameters (using a very low temperature at the moment to restrict the creativity in the model's answers).
4. Set up the text generation pipeline.
5. Set up an embedding model (GTE, also from HuggingFace) to convert the context provided into embeddings in order to create a vector store.
6. Created a prompt template to guide the chatbot's answers and behaviour.
7. Used the Selenium library to scrape a wikipedia page in order to create context for the chatbot's answers. From that context a vector store is created (with ChromaDB).
8. Implemented memory for the conversation.

### To do:
1. Explore different models (preferably open-sourced).
2. Explore different prompt templates that would possibly yield better outcomes.
3. Explore alternative methods to provide context (eg. through PDF files).
4. Create a UI so that users can query the chatbot (will probably be done with streamlit).
