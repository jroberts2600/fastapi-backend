# 🎓 Student Grades Analysis Backend


This FastAPI application serves as the backend for analyzing student grades. It reads a CSV file containing student grades, creates embeddings, and allows querying the data using various APIs.


## Features

- 📊 Query and analyze student grade data
- 🔗 Real-time interaction with FastAPI backend
- 📈 Advanced data analysis using FAISS and local embeddings
- 🤖 Integration with Ollama for natural language processing
- Dockerfile included for running fastapi as a web service on Render

## How to run it on your own machine

## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/fastapi-backend.git
    cd fastapi-backend
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Run the FastAPI application:
    ```sh
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```



## Backend Configuration

The app connects to a FastAPI backend. Update the `api_url` in `streamlit_app.py` if your backend location changes:

```python
api_url = "https://<ngrok url>/query"
```
```ngrok
cmd to start ngrok: ngrok http http://localhost:8000
```

## Data Source

Ensure that the `grades.csv` file is in the same directory as `main.py`. This file contains the student grade data used by the application.

## Google Colab

You can also run this backend on Google Colab. Click the link below to open the Colab notebook:
[Open in Google Colab](your-google-colab-link)


## Query Examples

- "What is the highest score in physics?"
- "Show me the top 5 scores in math"
- "How many students are in the dataset?"
- "What's the correlation between Math and Physics grades?"
- "Who are the best students based on combined scores?"

## Contributing

We welcome contributions! Please fork the repository and create a pull request with your improvements.

## License

This project is licensed under the MIT License.

