# PDF Conversational AI Retriever 💁💬


# How to run?
### STEPS:

Clone the repository

```bash
Project repo: https://github.com/aryan-bhagwat/pdf-conversational-ai-retriever.git
```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n llmapp python -y
```

```bash
conda activate llmapp
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

### Create a `.env` file in the root directory and add your GOOGLE_API_KEY as follows:

```ini
GROQ_API_KEY= "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


```bash
# Finally run the following command
streamlit run app.py
```

Now,
```bash
open up : http://localhost:8501
```

## Usage
- Upload one or more PDF files using the sidebar.
- Click "Submit & Process" to process the documents.
- Ask questions about the content of your PDFs in the main input box.

### Techstack Used:

- Python
- Streamlit
- LangChain
- Groq (Llama 3 70B)
- FAISS (vector store)
- PyPDF2 (PDF parsing)
- HuggingFace Embeddings