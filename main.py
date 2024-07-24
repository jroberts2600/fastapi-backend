from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from sentence_transformers import SentenceTransformer
import os
import requests
import numpy as np
import logging

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the CSV data
csv_file_path = './grades.csv'  # Ensure this file is in the same directory
data = pd.read_csv(csv_file_path)

# Check if the OpenAI API key is available
openai_api_key = os.getenv('OPENAI_API_KEY')

class LocalEmbedding:
    def embed_documents(self, texts):
        return [np.random.rand(384) for _ in texts]  # Match the dimension of SentenceTransformer

class SentenceTransformerWrapper:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts)

if openai_api_key:
    logging.info("Using OpenAI embeddings")
    embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
else:
    try:
        logging.info("Attempting to use SentenceTransformer embeddings")
        embedding_model = SentenceTransformerWrapper('all-MiniLM-L6-v2')
    except (ImportError, RuntimeError) as e:
        logging.warning(f"SentenceTransformer not available: {e}")
        logging.info("Falling back to local embeddings")
        embedding_model = LocalEmbedding()

# Create embeddings for the CSV data
def create_embeddings(data):
    data_str = data.apply(lambda row: ' '.join(row.astype(str)), axis=1)
    embeddings = embedding_model.embed_documents(data_str.tolist())
    return embeddings

data['embeddings'] = list(create_embeddings(data))

# Prepare documents for FAISS
documents = [
    Document(
        page_content=' '.join(row.astype(str)),
        metadata=row.to_dict()
    ) for _, row in data.iterrows()
]

# Create a FAISS index for fast similarity search
faiss_index = FAISS.from_documents(documents, embedding_model)

class Query(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the API"}

@app.post("/query/")
def query_model(query: Query):
    try:
        result = process_query(query.text, data, faiss_index)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_ollama_model(prompt):
    """Run the Ollama model on the given prompt."""
    try:
        # URL of the Ollama server via Ngrok
        ollama_url = "https://7ebe-71-81-132-14.ngrok-free.app/query/"
        #ollama_url = "https://fastapi-backend-9n3a.onrender.com/query"
        # Send request to Ollama server
        response = requests.post(ollama_url, json={"text": prompt})

        if response.status_code == 200:
            try:
                return response.json().get("result", "")
            except requests.JSONDecodeError:
                logging.error("Error: Response is not in JSON format")
                return ""
        else:
            logging.error(f"Error from Ollama server: {response.status_code} {response.text}")
            return ""
    except Exception as e:
        logging.error(f"Error running subprocess: {e}")
        return ""

def process_query(query: str, data: pd.DataFrame, faiss_index: FAISS) -> str:
    # Process different types of queries and extract relevant data
    if "highest score in physics" in query.lower():
        top_score_student = data.loc[data['Physics_Grade'].idxmax()]
        relevant_data_str = top_score_student[['Student', 'Physics_Grade']].to_string(index=False)
    elif "highest score in math" in query.lower():
        top_score_student = data.loc[data['Math_Grade'].idxmax()]
        relevant_data_str = top_score_student[['Student', 'Math_Grade']].to_string(index=False)
    elif "top 5 scores in physics" in query.lower():
        top_5_physics_scores = data.nlargest(5, 'Physics_Grade')[['Student', 'Physics_Grade']]
        relevant_data_str = top_5_physics_scores.to_string(index=False)
    elif "top 5 scores in math" in query.lower():
        top_5_math_scores = data.nlargest(5, 'Math_Grade')[['Student', 'Math_Grade']]
        relevant_data_str = top_5_math_scores.to_string(index=False)
    elif "how many students" in query.lower() or "number of students" in query.lower():
        num_students = data.shape[0]
        relevant_data_str = f"The dataset contains grades for {num_students} students."
    elif "correlation" in query.lower():
        data_str = data.to_csv(index=False)
        prompt = f"""
        The dataset contains grades for {data.shape[0]} students in Math and Physics. 
        Please perform the following tasks:

        1. Analyze the correlation between Math_Grade and Physics_Grade.
        2. Explain the strength and direction of the correlation.
        3. Identify any interesting patterns or insights from the data.

        The data is presented in the following format:

        Student,Math_Grade,Physics_Grade
        {data_str}

        Provide a detailed analysis based on the above data.
        """
        return run_ollama_model(prompt)
    elif "read the entire csv" in query.lower() or "show the csv" in query.lower():
        relevant_data_str = data.to_string(index=False)
    elif "how did you arrive at the top 5 physics scores" in query.lower():
        top_5_physics_scores = data.nlargest(5, 'Physics_Grade')[['Student', 'Physics_Grade']]
        relevant_data_str = top_5_physics_scores.to_string(index=False)
        prompt = f"The following data represents the top 5 physics scores:\n\n{relevant_data_str}\n\nExplain how this list was generated and the reasoning behind it."
        return run_ollama_model(prompt)
    elif "best math students" in query.lower() or "best students in math" in query.lower():
        top_students = data.nlargest(5, 'Math_Grade')[['Student', 'Math_Grade']]
        relevant_data_str = top_students.to_string(index=False)
        prompt = f"The following data represents the best students in math based on their grades:\n\n{relevant_data_str}\n\nProvide a detailed analysis and explanation."
        return run_ollama_model(prompt)
    elif "best physics students" in query.lower() or "best students in physics" in query.lower():
        top_students = data.nlargest(5, 'Physics_Grade')[['Student', 'Physics_Grade']]
        relevant_data_str = top_students.to_string(index=False)
        prompt = f"The following data represents the best students in physics based on their grades:\n\n{relevant_data_str}\n\nProvide a detailed analysis and explanation."
        return run_ollama_model(prompt)
    elif "best students combined score" in query.lower():
        data['Combined_Score'] = data['Math_Grade'] + data['Physics_Grade']
        top_students = data.nlargest(5, 'Combined_Score')[['Student', 'Math_Grade', 'Physics_Grade', 'Combined_Score']]
        relevant_data_str = top_students.to_string(index=False)
        prompt = f"The following data represents the best students based on their combined math and physics scores:\n\n{relevant_data_str}\n\nProvide a detailed analysis and explanation."
        return run_ollama_model(prompt)
    elif "best student" in query.lower():
        data['Combined_Score'] = data['Math_Grade'] + data['Physics_Grade']
        best_student = data.loc[data['Combined_Score'].idxmax()]
        relevant_data_str = best_student[['Student', 'Math_Grade', 'Physics_Grade', 'Combined_Score']].to_string(index=False)
        prompt = f"The following data represents the student with the best combined score in math and physics:\n\n{relevant_data_str}\n\nExplain why this student is considered the best."
        return run_ollama_model(prompt)
    else:
        # Use FAISS index to find the most similar entry
        similar_docs = faiss_index.similarity_search(query, k=1)
        most_similar_doc = similar_docs[0]
        relevant_data_str = pd.Series(most_similar_doc.metadata).to_string()

    prompt = f"Analyze the following data and answer the query: {query}\n\nData:\n{relevant_data_str}"
    return run_ollama_model(prompt)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)