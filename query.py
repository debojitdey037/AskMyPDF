import os
import sys
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma  # same package as ingest.py: pip install langchain-chroma
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

PERSIST_DIR = "db"
EMBEDDING_MODEL = "models/gemini-embedding-001"  # must match ingest.py
# Change the model here, or set GEMINI_MODEL in your .env file
CHAT_MODEL = os.environ.get("GEMINI_MODEL", "models/gemini-3.5-flash")
TOP_K = 4

PROMPT_TEMPLATE = """You are a helpful assistant answering questions based on the provided PDF course material.
Use the following pieces of retrieved context to answer the user's question accurately.
If the answer cannot be found in the context, say that you don't know based on the provided material.

Context:
{context}

Question: {question}

Answer:"""


def get_rag_chain():
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY is not set. Please check your .env file.")
        sys.exit(1)

    if not os.path.exists(PERSIST_DIR):
        print(f"Error: Vector database folder '{PERSIST_DIR}' not found.")
        print("Please run 'python3 ingest.py' first to index your PDF.")
        sys.exit(1)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )

    vector_store = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )

    # The folder can exist but be empty (e.g. ingestion failed part-way)
    if not vector_store.get(limit=1)["ids"]:
        print(f"Error: The database in '{PERSIST_DIR}' is empty.")
        print("Please run 'python3 ingest.py' to index your PDF.")
        sys.exit(1)

    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=api_key,
        temperature=0.2,
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    # Build the chain once instead of on every question
    chain = prompt | llm

    def page_number(doc):
        """PyPDFLoader pages are 0-indexed; convert to the 1-indexed page you see in the PDF."""
        page = doc.metadata.get("page")
        return page + 1 if isinstance(page, int) else None

    def answer_question(query: str):
        docs = retriever.invoke(query)

        context_text = "\n\n".join(
            f"[Page {page_number(d) or 'unknown'}]: {d.page_content}" for d in docs
        )

        response = chain.invoke({"context": context_text, "question": query})

        content = response.content
        if isinstance(content, list):
            answer_text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        else:
            answer_text = str(content)

        sources = sorted({page_number(d) for d in docs if page_number(d) is not None})
        return answer_text, sources

    return answer_question


def print_result(answer, sources):
    print("\nAnswer:")
    print(answer)
    if sources:
        print(f"\nSources: Page(s) {', '.join(map(str, sources))}")


def main():
    print("Loading vector database and Gemini model...")
    ask = get_rag_chain()
    print("Ready!\n" + "=" * 50)

    # If question passed via CLI arguments:
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"\nQuestion: {query}")
        print("\nThinking...")
        try:
            answer, sources = ask(query)
        except Exception as e:
            print(f"\nError while answering: {e}")
            sys.exit(1)
        print_result(answer, sources)
        return

    # Interactive mode:
    print("Ask any question about your PDF (type 'exit' or 'quit' to stop):")
    while True:
        try:
            query = input("\nAsk: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break

            print("\nThinking...")
            answer, sources = ask(query)
            print_result(answer, sources)
            print("-" * 50)
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        except Exception as e:
            # Don't crash the whole session on a temporary API error
            print(f"\nError while answering: {e}")


if __name__ == "__main__":
    main()