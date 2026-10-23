import os
import sys
import glob
import time
import hashlib
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma  # pip install langchain-chroma

# Load environment variables (e.g., GOOGLE_API_KEY from .env)
load_dotenv()

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
BATCH_SIZE = 50
MAX_ATTEMPTS = 5
PERSIST_DIR = "db"

# Error substrings that are worth retrying (rate limits, temporary network/server issues)
RETRYABLE_MARKERS = (
    "RESOURCE_EXHAUSTED",
    "429",
    "503",
    "UNAVAILABLE",
    "DEADLINE_EXCEEDED",
    "timeout",
    "timed out",
    "connection",
)


def is_rate_limit(err: str) -> bool:
    return "RESOURCE_EXHAUSTED" in err or "429" in err


def is_retryable(err: str) -> bool:
    err_lower = err.lower()
    return any(marker.lower() in err_lower for marker in RETRYABLE_MARKERS)


def make_id(chunk, index: int) -> str:
    """Deterministic ID so re-running ingestion overwrites instead of duplicating."""
    source = str(chunk.metadata.get("source", ""))
    page = str(chunk.metadata.get("page", ""))
    raw = f"{source}|{page}|{index}|{chunk.page_content}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def main():
    # 1. Determine the PDF file to load
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_files = glob.glob("*.pdf")
        if pdf_files:
            pdf_path = pdf_files[0]
            print(f"No file specified. Using found PDF: {pdf_path}")
        else:
            pdf_path = "document.pdf"

    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found.")
        print("Please place a PDF in this folder or run: python3 ingest.py <path_to_pdf>")
        sys.exit(1)

    # 2. Check for Google API Key (stop early if missing)
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not found.")
        print("Please set your API key in a .env file or run:")
        print("  export GOOGLE_API_KEY='your-gemini-api-key'")
        sys.exit(1)

    # 3. Load PDF
    print(f"Loading PDF from: {pdf_path} ...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} page(s).")

    # 4. Split document into chunks
    print(f"Splitting document into chunks (chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}) ...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = text_splitter.split_documents(pages)
    print(f"Total chunks created: {len(chunks)}")

    if not chunks:
        print("No text chunks were generated from the PDF.")
        print("If this is a scanned/image-only PDF, it needs OCR before ingestion.")
        return

    # 5. Generate embeddings and store in Chroma in batches
    print("Initializing Google Gemini embeddings ('models/gemini-embedding-001') ...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )

    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    print(f"Storing chunks in batches of {BATCH_SIZE} into local Chroma database ('{PERSIST_DIR}') ...")

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        ids = [make_id(chunk, i + j) for j, chunk in enumerate(batch)]
        start_idx = i + 1
        end_idx = min(i + BATCH_SIZE, len(chunks))
        print(f"  Embedding & saving chunks {start_idx} to {end_idx} of {len(chunks)} ...")

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                vector_store.add_documents(batch, ids=ids)
                break
            except Exception as e:
                err_str = str(e)

                # Non-retryable errors (bad API key, invalid model, etc.) fail immediately
                if not is_retryable(err_str):
                    print(f"    Non-retryable error: {e}")
                    raise

                if attempt == MAX_ATTEMPTS:
                    print(f"    Giving up after {MAX_ATTEMPTS} attempts.")
                    raise

                wait_time = 15 if is_rate_limit(err_str) else 3
                reason = "Rate limit reached" if is_rate_limit(err_str) else f"Temporary issue ({e})"
                print(f"    {reason}. Retrying in {wait_time}s (attempt {attempt}/{MAX_ATTEMPTS}) ...")
                time.sleep(wait_time)

        time.sleep(1.5)

    print(f"\nIngestion complete! All {len(chunks)} chunks have been stored in the './{PERSIST_DIR}' directory.")


if __name__ == "__main__":
    main()