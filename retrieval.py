from leann.api import LeannSearcher
from pathlib import Path

INDEX_DIR = Path("indexes_pdf").resolve()

INDEX_PATH = str(INDEX_DIR / "pdf_demo.leann")

searcher = LeannSearcher(INDEX_PATH)
results = searcher.search("What was adidas’ outlook and strategic focus for 2021 and beyond according to the CEO?", top_k=3)

documents = [r.text for r in results]

for i, document in enumerate(documents, 1):
    print(f"Document {i}:\n{document}\n")

