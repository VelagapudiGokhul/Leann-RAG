from leann import LeannBuilder, LeannSearcher, LeannChat
from pathlib import Path
from typing import Optional
from llama_index.core import Document
from llama_index.readers.file import PDFReader
from leann.chunking_utils import create_traditional_chunks

# Paths
DATA_DIR = Path("pdfs")          
INDEX_DIR = Path("leann_indexes_pdf").resolve()
INDEX_DIR.mkdir(exist_ok=True)
INDEX_PATH = str(INDEX_DIR / "pdf_demo.leann")

# Step 1: Load PDFs as Documents
pdf_reader = PDFReader()
docs = []

for pdf_path in DATA_DIR.glob("*.pdf"):
    pdf_docs = pdf_reader.load_data(file=pdf_path)
    for d in pdf_docs:
        docs.append(
            Document(
                text=d.text,
                metadata={"file_path": str(pdf_path)}
            )
        )

print(f"Loaded {len(docs)} PDF documents")

# Step 2: Chunk Documents
chunks = create_traditional_chunks(
    docs,
    chunk_size=512,     
    chunk_overlap=0
)

print(f"Created {len(chunks)} chunks")


# Step 3: Build LeANN Index
builder = LeannBuilder(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    embedding_mode="sentence-transformers",
    backend_name="hnsw",
    graph_degree=32,
    build_complexity=64,
    device="cpu"
)

for chunk in chunks:
    builder.add_text(chunk["text"])

builder.build_index(INDEX_PATH)

print(f"\nLeANN index built at: {INDEX_PATH}")



