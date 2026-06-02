import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import csv
import json
import asyncio

from openai import AsyncOpenAI  # Use AsyncOpenAI for async compatibility
from ragas import experiment, Dataset  # Corrected Dataset import
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric
from ragas.metrics.collections import ContextRecall, ContextPrecision

load_dotenv()

# Add current dir to sys.path
sys.path.insert(0, str(Path(__file__).parent))

# ---------- LLM setup ----------
# For async context, we use AsyncOpenAI
openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
llm = llm_factory("o4-mini-2025-04-16", client=openai_client)

# ---------- Dataset loading ----------

def load_json_source(json_path="my_dataset.json"):
    """
    Load JSON with question, optional source_text and answer_guidelines
    Returns dict: question -> {"source_text": ..., "grading_notes": ...}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        row.get("question", f"question_{i}"): {
            "source_text": row.get("source_text", ""),        # default "" nếu thiếu
            "grading_notes": row.get("answer_guidelines", "") # default "" nếu thiếu
        }
        for i, row in enumerate(data)
    }

def load_rag_csv_as_dataset(csv_path="rag_output.csv"):
    """
    Load CSV from RAG output and return as a Dataset
    """
    # Use "local/csv" backend for standard Ragas v0.4 compliance
    dataset = Dataset(
        name="hsf_rag_dataset",
        backend="local/csv", 
        root_dir="my_output"
    )

    rag_rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rag_rows.append(row)

    for row in rag_rows:
        question = row.get("question", "")
        dataset.append({
            "question": question,
            "response": row.get("answer", ""),
            "retrieved_context": row.get("context", ""),
            "time_sec": row.get("time_sec", ""),
        })

    dataset.save()  # Ragas datasets generally need to be saved before execution
    return dataset

# ---------- Metrics setup ----------

my_metric = DiscreteMetric(
    name="correctness",
    prompt=(
        "Check if the response contains points mentioned from the grading notes "
        "and return 'pass' or 'fail'.\nResponse: {response} Grading Notes: {grading_notes}"
    ),
    allowed_values=["pass", "fail"],
)
context_recall_metric = ContextRecall(llm=llm)
context_precision_metric = ContextPrecision(llm=llm)

# ---------- Experiment ----------

@experiment()
async def run_experiment(row):
    """
    row: dict with 'question', 'response', 'retrieved_context', 'time_sec'
    """
    question = row.get("question", "")
    response = row.get("response", "")
    retrieved_context = row.get("retrieved_context", "")
    
    expected_context = source_text_dict.get(question, {}).get("source_text", "")
    grading_notes = source_text_dict.get(question, {}).get("grading_notes", "")

    # Use await and .ascore() inside an async experiment
    score = await my_metric.ascore(
        llm=llm,
        response=response,
        grading_notes=grading_notes
    )

    # Compute context metrics only if expected_context exists
    if expected_context.strip():
        # Note the updated arguments for Ragas v0.4 metrics
        recall = await context_recall_metric.ascore(
            user_input=question,
            retrieved_contexts=[retrieved_context],  # expects a list
            reference=expected_context
        )
        precision = await context_precision_metric.ascore(
            user_input=question,
            retrieved_contexts=[retrieved_context],  # expects a list
            reference=expected_context
        )
        context_metrics = {
            "context_recall": recall.value,
            "context_precision": precision.value,
        }
    else:
        context_metrics = {
            "context_recall": None,
            "context_precision": None,
        }

    return {
        **row,
        "score": score.value,
        **context_metrics
    }

# ---------- Main ----------

async def main():
    global source_text_dict
    source_text_dict = load_json_source("my_dataset.json")
    rag_dataset = load_rag_csv_as_dataset("hsf_multistage_result.csv")
    
    print(f"Loaded {len(rag_dataset)} RAG rows and {len(source_text_dict)} JSON entries.")

    # Run experiment using .arun() with the Dataset object
    results = await run_experiment.arun(rag_dataset)

    # Save results to CSV
    output_dir = Path("my_output")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "hsf_multistages.csv"
    
    if results:
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            fieldnames = list(results[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print(f"Experiment completed! Results saved to {output_file.resolve()}")

# ---------- Entry point ----------

if __name__ == "__main__":
    asyncio.run(main())