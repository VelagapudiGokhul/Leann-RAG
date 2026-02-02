from leann import LeannBuilder, LeannSearcher, LeannChat
from pathlib import Path
from typing import Optional
from llama_index.core import Document
from leann.chunking_utils import (
    create_ast_chunks,
    create_text_chunks,
    create_traditional_chunks,
)


INDEX_DIR = Path("indexes").resolve()
INDEX_DIR.mkdir(exist_ok=True)

INDEX_PATH = str(INDEX_DIR / "demo.leann")

class MockDocument:
    """Mock LlamaIndex Document for testing."""

    def __init__(self, content: str, file_path: str = "", metadata: Optional[dict] = None):
        self.content = content
        self.metadata = metadata or {}
        if file_path:
            self.metadata["file_path"] = file_path

    def get_content(self) -> str:
        return self.content

data_dir = Path("documents")

docs = []
for file_path in data_dir.glob("*.txt"):
    text = file_path.read_text(encoding="utf-8")
    docs.append(
        Document(
            text=text,
            metadata={"file_path": str(file_path)}
        )
    )

        
chunks = create_traditional_chunks(docs, chunk_size=300, chunk_overlap=0)


print(len(chunks))
for i, chunk in enumerate(chunks[:5]):
    print(f"\n--- Chunk {i} ---")
    print(chunk["text"])
    
    
builder = LeannBuilder(
    embedding_model="facebook/contriever",
    embedding_mode="sentence-transformers",
    backend_name="hnsw",
    graph_degree=32,
    build_complexity=64
)

for chunk in chunks:
    builder.add_text(chunk["text"])

builder.build_index(INDEX_PATH)
