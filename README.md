# 🎓 Student Grades Analysis App

A Streamlit app for analyzing student grades with a FastAPI backend!


## Features

- 📊 Query and analyze student grade data
- 🔗 Real-time interaction with FastAPI backend
- 👥 User-friendly interface for easy querying
- 📈 Advanced data analysis using FAISS and local embeddings
- 🤖 Integration with Ollama for natural language processing

## How to run it on your own machine

1. Clone the repository

   ```
   $ git clone https://github.com/yourusername/student-grades-analysis.git
   $ cd student-grades-analysis
   ```

2. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

3. Run the FastAPI backend

   ```
   $ uvicorn main:app --reload
   ```

4. Run the Streamlit app

   ```
   $ streamlit run streamlit_app.py
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

