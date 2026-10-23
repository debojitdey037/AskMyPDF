# AskMyPDF

A command-line RAG (Retrieval-Augmented Generation) chatbot that answers questions about a PDF using Google Gemini, LangChain, and ChromaDB. Answers are grounded in your document and include the page numbers they came from.

## How It Works

1. **Ingest** (`ingest.py`): loads a PDF, splits it into overlapping text chunks, embeds each chunk with Gemini embeddings, and stores them in a local Chroma vector database.
2. **Query** (`query.py`): embeds your question, retrieves the most relevant chunks from the database, and sends them to a Gemini chat model, which writes an answer based only on that context.

```
PDF -> chunks -> Gemini embeddings -> ChromaDB
                                          |
Question -> embedding -> top-k chunks -> Gemini -> answer + page sources
```

## Features

- Works with any text-based PDF
- Re-running ingestion does not create duplicates (deterministic chunk IDs)
- Automatic retry on rate limits and temporary network errors
- Answers cite the PDF page numbers
- Interactive chat mode or single-question mode

## Tech Stack

- Python 3.12
- LangChain
- Google Gemini (embeddings and chat model)
- ChromaDB

## Requirements

- **Python 3.12** (newer versions such as 3.13+ may fail to install `onnxruntime`, which ChromaDB depends on)
- A free [Gemini API key](https://aistudio.google.com/apikey)

## Setup

**1. Clone the repository and go into the folder**

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

**2. Create and activate a virtual environment**

```bash
python3.12 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

**3. Install dependencies**

```bash
python -m pip install python-dotenv pypdf langchain-community langchain-text-splitters langchain-google-genai langchain-chroma
```

**4. Add your API key**

Create a file named `.env` in the project folder:

```
GOOGLE_API_KEY=your-gemini-api-key
```

Optional: choose a different chat model.

```
GEMINI_MODEL=models/gemini-2.5-flash
```

## Usage

**Step 1: Index your PDF**

Place a PDF in the project folder, then run:

```bash
python ingest.py
```

Or point to a specific file:

```bash
python ingest.py path/to/your-file.pdf
```

**Step 2: Ask questions**

Ask a single question:

```bash
python query.py "What is a binary tree?"
```

Or start interactive mode:

```bash
python query.py
```

Type `exit`, `quit`, or `q` to stop.

### Example

```
Question: What is a binary tree?

Answer:
A binary tree is a hierarchical data structure in which each node has at most two children...

Sources: Page(s) 3, 4
```

## Project Structure

```
.
├── ingest.py        # Loads, chunks, embeds, and stores the PDF
├── query.py         # Retrieves context and answers questions
├── .env             # Your API key (not committed)
├── .env.example     # Template for the .env file
├── db/              # Local Chroma vector database (created by ingest.py)
└── venv/            # Virtual environment (not committed)
```

## Configuration

| Setting | File | Default |
| --- | --- | --- |
| Chunk size | `ingest.py` | 800 characters |
| Chunk overlap | `ingest.py` | 100 characters |
| Embedding model | `ingest.py`, `query.py` | `models/gemini-embedding-001` |
| Chat model | `query.py` or `GEMINI_MODEL` in `.env` | `models/gemini-3.5-flash` |
| Chunks retrieved per question | `query.py` (`TOP_K`) | 4 |

The embedding model must be the same in both files, and you must re-run `ingest.py` (after deleting `db/`) if you change it or the chunk settings.

## Troubleshooting

- **`ResolutionImpossible` or `onnxruntime` errors during install:** you are on a Python version that is too new. Use Python 3.12.
- **"Import could not be resolved" in VS Code:** select the interpreter inside `venv` (Cmd/Ctrl + Shift + P, then "Python: Select Interpreter").
- **Rate limit messages while ingesting:** normal on the free Gemini tier. The script waits and retries automatically.
- **"Model not found" error:** set `GEMINI_MODEL` in `.env` to a model available to your API key.
- **Empty or poor answers on a scanned PDF:** the PDF has no extractable text. Run OCR on it first.
- **Database not found:** run `python ingest.py` before `python query.py`.

## Limitations

- Handles one PDF collection at a time through a single local database. Delete the `db/` folder to start fresh with a different document.
- Scanned (image-only) PDFs are not supported without OCR.
- Answers are only as good as the retrieved chunks, so very broad questions may miss information spread across many pages.

## License

MIT License

Copyright (c) 2026 Debojit Dey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
