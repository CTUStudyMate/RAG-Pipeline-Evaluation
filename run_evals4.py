import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import csv
import json
import asyncio

from openai import AsyncOpenAI  # Use AsyncOpenAI for async compatibility
from ragas import experiment, Dataset  # Corrected Dataset import
from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric
from ragas.metrics.collections import ContextRecall, ContextPrecision, Faithfulness, AnswerCorrectness
from difflib import get_close_matches, SequenceMatcher

load_dotenv()

INPUT_DATAJSON = "dataset_v2.json"
INPUT_DATASET = "EXPs/v1/normal_v2.csv"
OUTPUT_FILE = "normal_v2.csv"
DATASET_NAME = "normal_v2"

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
llm = llm_factory("gpt-4o-mini", client=openai_client, max_tokens=8192)
embeddings = embedding_factory("openai", model="text-embedding-3-small", client=openai_client)
# ollama_client = AsyncOpenAI(
#     base_url="http://localhost:11434/v1",
#     api_key="ollama"  # dummy
# )
# llm = llm_factory("qwen3.5:cloud", client=ollama_client)

# ---------- Dataset loading ----------

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
        is_abstain_response = bool(get_close_matches(
            response,
            ["The chatbot can't answer this question. Please try again with another question."],
            n=1,
            cutoff=0.8
        ))
        
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
                true_abstention_val = int(is_abstain_response)
        else:
            correct_val = "N/A"

        # precision, recall, f1
        if reference:
            precision = await ctx_prec_scorer.ascore(
                user_input=question,
                reference=reference,
                retrieved_contexts=[retrieved_context]
            )
            recall = await ctx_rec_scorer.ascore(
                user_input=question,
                retrieved_contexts=[retrieved_context],
                reference=reference
            )
            prec = precision.value
            rec = recall.value
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0
        else:
            prec = rec = f1 = "N/A"
        
        # faithfulness    
        if retrieved_context:
            if get_close_matches(
                response,
                ["The chatbot can't answer this question. Please try again with another question."],
                n=1,
                cutoff=0.8
            ):
                faithfulness_score = "N/A"
                    
            else:
                faithfulness = await faithfulness_scorer.ascore(
                    user_input=question,
                    response=response,
                    retrieved_contexts=[retrieved_context]
                )
                faithfulness_score = faithfulness.value
        else: 
            faithfulness_score = "N/A"        

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

    # Save results to CSV
    output_dir = Path(f"EXPs/v1/run_result")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{OUTPUT_FILE}"
    
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