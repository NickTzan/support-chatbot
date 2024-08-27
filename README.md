# Support Chatbot for Customer Question Answering
------------
## Nick Tzanis Thesis for AI MSc

### Chatbot:
1. Switched entirely the API call platform from HuggingFace to Groq to leverage faster API inference times with large LLMs (Meta's Llama 3 70b is used in the current iteration of the code).
2. Using the above API, created a RAG system that can use context to provide answers to a user.
3. Added the option to use either web scraping of websites (with Selenium) or pdf files in a local folder for the context.
4. Set up an embedding model (all-mpnet-base-v2, also from HuggingFace) to convert the context provided into embeddings in order to create a vector store.
5. Implemanted a UI with streamlit to query the chatbot.
6. Implemented memory for the conversation.

### Model Evaluation:
1. Created a dataset for the evaluation of the models.
2. The notebook "create_eval_sets.ipynb" saves each model's answers for the questions in the evaluation set.
3. Then in the "evaluate.ipynb" notebook using the ragas library metrics we evaluate each model's answers.
4. Five different models were evaluated (llama 8b, llama 70b, mixtral 8x7b, gemma 7b and gemma2 9b).
5. The metrics for each model are presented with visualisations using the matplotlib library.