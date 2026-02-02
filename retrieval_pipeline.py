from leann.api import LeannSearcher
from pathlib import Path
import pandas as pd
from openai import OpenAI
import os
import time

# Config
INDEX_DIR = Path("leann_indexes_pdf").resolve()
INDEX_PATH = str(INDEX_DIR / "pdf_demo.leann")
TOP_K = 3
OUTPUT_FILE = "leann_rag_results.xlsx"

# OpenAI setup
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# Questions 
questions = [

    "How did the COVID-19 pandemic impact adidas’ overall financial performance in 2020?",

    "What were the key financial highlights of adidas in 2020 compared to 2019 (net sales, operating profit, and net income)?",

    "What actions did adidas take to accelerate its digital and e-commerce growth during 2020?",

    "What sustainability initiatives did adidas focus on in 2020, particularly related to materials and climate goals?",

    "What was adidas’ outlook and strategic focus for 2021 and beyond according to the CEO?",

    "How did Domino’s Pizza Enterprises perform financially in the 2021 financial year?",

    "What role did franchisees play in Domino’s growth and store expansion during 2021?",

    "How did Domino’s adapt its operations and strategy during COVID-19 disruptions?",

    "What were Domino’s key performance highlights across regions such as Europe, Japan, and Australia/New Zealand?",

    "What is Domino’s stated purpose and core values, and how do they guide the company’s decisions?",

    "How did COVID-19 impact Crown Resorts’ financial performance in the 2021 financial year?",

    "What regulatory challenges did Crown face in 2021, and what remediation actions were taken?",

    "How did Crown Melbourne, Crown Perth, and Crown Sydney perform differently during 2021?",

    "What major leadership and governance changes occurred at Crown during 2021?",

    "What is Crown’s outlook and strategic focus for recovery and long-term sustainability?"
]


# Init searcher
searcher = LeannSearcher(INDEX_PATH)

rows = []

# Helper: OpenAI synthesis
def get_openai_answer(question, contexts):
    context_text = "\n\n".join(
        f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)
    )

    prompt = f"""
You are an expert business analyst.

Use ONLY the information from the contexts below.
Please provide a clear, helpful answer using only the information from these documents. 
If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
Question:
{question}

Contexts:
{context_text}

Final Answer (concise, factual):
"""

    # Retry logic in case of transient errors
    for attempt in range(3):
        try:
            response = client.responses.create(
                model="openai/gpt-oss-20b",
                input=prompt
            )
            return response.output_text.strip()
        except Exception as e:
            print(f"OpenAI request failed (attempt {attempt+1}): {e}")
            time.sleep(5)
    return "Error: Failed to get answer from OpenAI"


# Main loop
for question in questions:
    results = searcher.search(question, top_k=TOP_K)
    top_texts = [r.text for r in results]

    while len(top_texts) < TOP_K:
        top_texts.append("")

    final_answer = get_openai_answer(question, top_texts)

    rows.append({
        "Question": question,
        "TopK_1": top_texts[0],
        "TopK_2": top_texts[1],
        "TopK_3": top_texts[2],
        "AI_Final_Answer": final_answer
    })

# Save to Excel
df = pd.DataFrame(rows)
df.to_excel(OUTPUT_FILE, index=False)

print(f"Saved results to {OUTPUT_FILE}")
