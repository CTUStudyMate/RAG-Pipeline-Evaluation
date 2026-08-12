import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import csv
import json
import asyncio
import re
from rapidfuzz import fuzz
from openai import AsyncOpenAI  # Use AsyncOpenAI for async compatibility
from ragas import experiment, Dataset  # Corrected Dataset import
from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric
from ragas.metrics.collections import ContextRecall, ContextPrecision, Faithfulness, AnswerCorrectness
from difflib import get_close_matches, SequenceMatcher

load_dotenv()

TARGET_QUESTION = "What is a precursor?"

INPUT_DATAJSON = "notebooklm_generated_dataset.json"
INPUT_DATASET = (
    "final_exp_for_report/answers/stage1__chunk_strategies/"
    "notebooklm/fixedsize_normal_notebooklm.csv"
)
RESULT_DATASET = Path(
    "final_exp_for_report/eval_results/stage1__chunk_strategies/"
    "notebooklm/stage1_fixedsize_normal_600_2000_notebooklm.csv"
)
DATASET_NAME = "stage1_fixedsize_normal_600_2000_notebooklm_missing_record"

def fuzzy_lookup(question, source_dict, cutoff=0.9):
    """
    Tìm question gần giống nhất trong source_dict, trả về dict {"source_text":..., "grading_notes":...}
    Nếu không tìm thấy match nào đủ gần, trả về dict trống.
    """
    # Lấy list keys của source_dict
    keys = list(source_dict.keys())
    # Tìm gần giống nhất
    matches = get_close_matches(question, keys, n=1, cutoff=cutoff)
    if matches:
        return source_dict[matches[0]]
    else:
        return {"source_text": "", "grading_notes": ""}

# Add current dir to sys.path
sys.path.insert(0, str(Path(__file__).parent))

# ---------- LLM setup ----------
# For async context, we use AsyncOpenAI
openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
llm = llm_factory(
    "gpt-5-nano",
    client=openai_client,
    max_tokens=8192,
    reasoning_effort="minimal",
    seed=42,
)
embeddings = embedding_factory("openai", model="text-embedding-3-small", client=openai_client)
# ollama_client = AsyncOpenAI(
#     base_url="http://localhost:11434/v1",
#     api_key="ollama"  # dummy
# )
# llm = llm_factory("qwen3.5:cloud", client=ollama_client)

# ---------- Dataset loading ----------

TOP_LEVEL_CHUNK_PATTERN = re.compile(
    r"(?:\A|\r?\n(?:\r?\n)+)\[(\d+)\][ \t]*\r?\n"
)


def parse_retrieved_contexts(raw_context):
    """
    Parse a concatenated context into top-level retrieved chunks.

    Top-level chunks are expected to be separated by at least one
    blank line before a numbered marker:

        [0]
        first chunk

        [1]
        second chunk

    Nested markers such as:

        [FIGURE_DESCRIPTIONS]
        [0] description

    are preserved because they are not preceded by a blank line.
    """
    if raw_context is None:
        return []

    if isinstance(raw_context, list):
        return [
            str(chunk).strip()
            for chunk in raw_context
            if str(chunk).strip()
        ]

    text = str(raw_context).strip()

    if not text:
        return []

    # Also support contexts saved as a JSON list in future files.
    try:
        parsed = json.loads(text)

        if isinstance(parsed, list):
            return [
                str(chunk).strip()
                for chunk in parsed
                if str(chunk).strip()
            ]
    except (json.JSONDecodeError, TypeError):
        pass

    markers = list(TOP_LEVEL_CHUNK_PATTERN.finditer(text))

    if not markers:
        return [text]

    marker_numbers = [
        int(marker.group(1))
        for marker in markers
    ]

    # The serializer numbers top-level chunks as 0, 1, 2, ...
    expected_numbers = list(range(len(marker_numbers)))

    if marker_numbers != expected_numbers:
        raise ValueError(
            "Unexpected top-level context numbering. "
            f"Found {marker_numbers}, expected {expected_numbers}."
        )

    chunks = []

    for index, marker in enumerate(markers):
        content_start = marker.end()

        if index + 1 < len(markers):
            content_end = markers[index + 1].start()
        else:
            content_end = len(text)

        chunk = text[content_start:content_end].strip()

        if chunk:
            chunks.append(chunk)

    return chunks

def load_json_source(json_path=INPUT_DATAJSON):
    """
    Load JSON with question, optional source_text and answer_guidelines
    Returns dict: question -> {"source_text": ..., "grading_notes": ...}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        row.get("question", f"question_{i}"): {
            "source_text": row.get("source_text", ""),        
            "ground_truth": row.get("ground_truth", "") 
        }
        for i, row in enumerate(data)
    }

def load_rag_csv_as_dataset(csv_path="rag_output.csv"):
    """
    Load CSV from RAG output and return as a Dataset
    """
    # Use "local/csv" backend for standard Ragas v0.4 compliance
    dataset = Dataset(
        name=DATASET_NAME,
        backend="local/csv", 
        root_dir="my_output"
    )

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rag_rows = [
            row for row in reader
            if row.get("question", "") == TARGET_QUESTION
        ]

    if len(rag_rows) != 1:
        raise ValueError(
            f"Expected exactly one record for {TARGET_QUESTION!r}, "
            f"found {len(rag_rows)}."
        )

    for row in rag_rows:
        question = row.get("question", "")
        dataset.append({
            "question": question,
            "response": row.get("answer", ""),
            "retrieved_context": row.get("context", ""),
            "retrieve_time_sec": row.get("retrieve_time_sec", ""),
            "generate_time_sec": row.get("generate_time_sec", "")
        })

    dataset.save()  # Ragas datasets generally need to be saved before execution
    return dataset

ctx_rec_scorer = ContextRecall(llm=llm)
ctx_prec_scorer = ContextPrecision(llm=llm)
cr_scorer = AnswerCorrectness(llm=llm, embeddings=embeddings)
faithfulness_scorer = Faithfulness(llm=llm)

# ---------- Experiment ----------

semaphore = asyncio.Semaphore(3) 

@experiment()
async def run_experiment(row):
    async with semaphore:  
        question = row.get("question", "")
        response = row.get("response", "")
        retrieved_context = row.get("retrieved_context", "")
        retrieved_contexts = parse_retrieved_contexts(retrieved_context)

        print("\n\nProcessing question: ")
        print(question[:120])
        # Fuzzy lookup
        matched = fuzzy_lookup(question, source_text_dict, cutoff=0.87)
        reference = matched.get("source_text", "")
        ground_truth = matched.get("ground_truth", "")
        
        is_abstain_groundtruth = bool(get_close_matches(
            ground_truth,
            ["The chatbot can't answer this question. Please try again with another question."],
            n=1,
            cutoff=0.8
        ))
        abstain_score = fuzz.partial_ratio_alignment("The chatbot can't answer this question. Please try again with another question.", response)
        if abstain_score.score > 85:
            is_abstain_response = True
        else:
            is_abstain_response = False
        
        
        # correctness & true abstention
        true_abstention_val = None
        if ground_truth:
            if is_abstain_groundtruth:
                correct_val = 1 if is_abstain_response else 0
                true_abstention_val = int(is_abstain_response)
            else:
                correct = await cr_scorer.ascore(
                    user_input=question,
                    response=response,
                    reference=ground_truth
                )
                correct_val = correct.value
                true_abstention_val = int(not is_abstain_response)
        else:
            correct_val = "N/A"

        # precision, recall, f1
        if not retrieved_contexts:
            prec = rec = f1 = 0.0
        elif reference:
            precision = await ctx_prec_scorer.ascore(
                user_input=question,
                reference=reference,
                retrieved_contexts=retrieved_contexts
            )
            recall = await ctx_rec_scorer.ascore(
                user_input=question,
                retrieved_contexts=retrieved_contexts,
                reference=reference
            )
            prec = precision.value
            rec = recall.value
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0
        else:
            prec = rec = f1 = "N/A"
        
        # faithfulness
        if not retrieved_contexts:
            faithfulness_score = 1.0 if is_abstain_response else 0.0
        elif is_abstain_response:
            faithfulness_score = "N/A"
        else:
            faithfulness = await faithfulness_scorer.ascore(
                user_input=question,
                response=response,
                retrieved_contexts=retrieved_contexts
            )
            faithfulness_score = faithfulness.value

        await asyncio.sleep(0.1) 
        
        return {
            **row,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "correctness": correct_val,
            "faithfulness": faithfulness_score,
            "true_abstention": true_abstention_val
        }

# ---------- Main ----------

async def main():
    global source_text_dict
    source_text_dict = load_json_source(INPUT_DATAJSON)
    with open("source_text_dict.json", "w", encoding="utf-8") as f:
        json.dump(source_text_dict, f, indent=2)
    rag_dataset = load_rag_csv_as_dataset(INPUT_DATASET) #file csv của mình
    
    print(f"Loaded {len(rag_dataset)} RAG rows and {len(source_text_dict)} JSON entries.")

    # Run experiment using .arun() with the Dataset object
    results = await run_experiment.arun(rag_dataset)

    if not RESULT_DATASET.exists():
        raise FileNotFoundError(f"Missing result file: {RESULT_DATASET}")

    with open(RESULT_DATASET, newline="", encoding="utf-8") as f:
        existing_questions = {row["question"] for row in csv.DictReader(f)}

    rows_to_append = [
        row for row in results
        if row["question"] not in existing_questions
    ]

    if rows_to_append:
        with open(RESULT_DATASET, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_to_append[0].keys()))
            writer.writerows(rows_to_append)
        print(f"Appended {len(rows_to_append)} record(s) to {RESULT_DATASET.resolve()}")
    else:
        print(f"No records appended; question already exists in {RESULT_DATASET.resolve()}")

# ---------- Entry point ----------

if __name__ == "__main__":
    asyncio.run(main())
