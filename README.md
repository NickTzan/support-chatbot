# Support Chatbot for Customer Question Answering
------------
## Nick Tzanis Thesis for AI MSc

### Ready:
1. Switched entirely the API call platform from HuggingFace to Groq to leverage faster API inference times with large LLMs (Meta's Llama 3 70b is used in the current iteration of the code).
2. Using the above API, created a RAG system that can use context to provide answers to a user.
3. Added the option to use either web scraping of websites (with Selenium) or pdf files in a local folder for the context.
4. Set up an embedding model (GTE, also from HuggingFace) to convert the context provided into embeddings in order to create a vector store.
5. Implemanted a UI with streamlit to query the chatbot.
6. Created a temporary prompt template to guide the chatbot's answers and behaviour.
7. Implemented memory for the conversation.
8. The Streamlit UI now allows the user to select the model used in the conversation.

### To do:
1. Try out and evaluate other models (probably from Groq too). At this time the models offered are Meta's Llama3 8b, Llama3 70b, Mixtral 8x7b and Gemma 7b.
2. Create a dataset for the evaluation of the models.
3. Adjust prompt templates that would possibly yield better outcomes for each model.
