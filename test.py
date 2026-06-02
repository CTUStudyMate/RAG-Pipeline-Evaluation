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
from difflib import get_close_matches

load_dotenv()



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

openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
llm = llm_factory("gpt-4o-mini", client=openai_client)
embeddings = embedding_factory("openai", model="text-embedding-3-small", client=openai_client)

cr_scorer = AnswerCorrectness(llm=llm, embeddings=embeddings)
async def test_metric():
    result = await cr_scorer.ascore(
        user_input="When was the first super bowl?",
        response="I don't know",
        reference="I don't know"
    )
    print(f"Answer Correctness Score: {result.value}")

asyncio.run(test_metric())