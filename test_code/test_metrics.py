from ragas.metrics import DiscreteMetric
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory

import os
from dotenv import load_dotenv
import asyncio
from ragas.metrics.collections import ContextRecall, ContextPrecision

from testmodel import parse_retrieved_contexts


load_dotenv()

# ollama_client = AsyncOpenAI(
#     base_url="http://localhost:11434/v1",
#     api_key="ollama"  # dummy, không quan trọng
# )
# llm = llm_factory("qwen3.5:cloud", client=ollama_client, max_tokens=12000)
openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# llm = llm_factory(
#     "gpt-5-nano",
#     client=openai_client,
#     max_tokens=8192,
#     reasoning_effort="minimal",
# )
# llm = llm_factory(
#     "gpt-5-nano",
#     client=openai_client,
#     max_tokens=8192,
#     reasoning_effort="minimal",
# )
embeddings = embedding_factory("openai", model="text-embedding-3-small", client=openai_client)
# correctness = DiscreteMetric(
#     name="correctness",
#     prompt=(
#         "Check if the response contains points mentioned from the grading notes "
#         "and return 'pass' or 'fail'.\nResponse: {response} Grading Notes: {grading_notes}"
#     ),
#     allowed_values=["pass", "fail"],
# )

# ctx_rec_scorer = ContextRecall(llm=llm)
# ctx_prec_scorer = ContextPrecision(llm=llm)

# ---------- Experiment ----------

async def run_experiment(row):
    """
    row: dict with 'question', 'response', 'retrieved_context', 'time_sec'
    """
    question = row.get("question", "")
    response = row.get("response", "")
    retrieved_context = row.get("retrieved_context", "")
    retrieved_context = parse_retrieved_contexts(retrieved_context)
    
    grading_notes = mark.get(question, {}).get("grading_notes", "")
    reference = mark.get(question, {}).get("source_text","")


    # correctness
    # if grading_notes:
    #     correct = await correctness.ascore(
    #         llm=llm,
    #         response=response,
    #         grading_notes=grading_notes
    #     )
    #     correct_val = correct.value
    # else:
    #     correct_val = "N/A"
    
    # prec, rec, f1
    if reference:
        precision = await ctx_prec_scorer.ascore(
            user_input=question,
            reference=reference,
            retrieved_contexts=retrieved_context
        )
        recall = await ctx_rec_scorer.ascore(
            user_input=question,
            retrieved_contexts=retrieved_context,
            reference=reference
        )
        prec = precision.value
        rec = recall.value
        f1 = (2*prec*rec)/(prec+rec) if (prec+rec) else 0
    else:
        prec = rec = f1 = "N/A"
    
    x = {
        **row,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        # "correctness": correct_val,
    }
    print("Precision: ",x["precision"])
    print("Recall: ", x["recall"])
    print("F1: ",x["f1"])
    
        
    return 


# 1. Cấu trúc lại biến mark thành dictionary 2 lớp (lấy câu hỏi làm key)
global mark

mark = {
    "When visually mapping requirements versus specifications, what domain sits exactly at the intersection where the Environment domain overlaps with the System domain?": {
        "ground_truth": "The Shared Interface sits exactly at the intersection where the Environment domain overlaps with the System domain.",
        "source_text": "Figure 4.3 illustrates a Venn diagram representing the relationship between Requirements and Specification. The diagram consists of two overlapping circles labeled Environment and System. The area exactly where the Environment circle overlaps with the System circle is explicitly labeled as the Shared Interface."
    }
}

# 2. obj chứa câu hỏi và ngữ cảnh mà LLM đã trả lời
obj = {
    "question": "When visually mapping requirements versus specifications, what domain sits exactly at the intersection where the Environment domain overlaps with the System domain?",
    "response": """
Environmental assumptions (assumptions about what inputs the system will receive, or about how the environment will react to outputs).
""",
    "retrieved_context": """
    [0]
y refer only to those real-world objects (states, events, actions) that are sensed or actuated by the proposed system:

1. In  documenting  the  system's  interface, we  describe  all  inputs  and  outputs  in detail, including  the  sources  of  inputs, the  destinations  of  outputs, the  value ranges and data formats of input and output data, protocols governing the order in  which  certain  inputs  and  outputs  must  be  exchanged, window  formats  and organization, and any timing constraints. Note that the user interface is rarely the only system interface; the system may interact with other software components (e.g., a database), special-purpose hardware, the Internet, and so on.
2. Next, we restate the required functionality in terms of the interfaces' inputs and outputs. We may use a functional notation or data-flow diagrams to map inputs to outputs, or use logic to document functions' pre-conditions and post-conditions.We may use state machines or event traces to illustrate exact sequences of operations

## SIDEBAR 4.7 HIDDEN ASSUMPTIONS

Z ave and Jackson (1997) have looked carefully at problems in software requirements and specification, including undocumented assumptions about how the real world behaves.

There are actually two types of environmental behavior of interest: desired behavior to be  realized  by  the  proposed  system  (i.e., the  requirements)  and  existing  behavior  that  is unchanged by the proposed system.The latter type of behavior is often called assumptions or domain knowledge . Most requirements writers consider assumptions to be simply the conditions under which the system is guaranteed to operate correctly.While necessary, these conditions are not the only assumptions. We also make assumptions about how the environment will behave in response to the system's outputs.

Consider a railroad-crossing  gate  at  the  intersection  of  a  road  and  a  set  of  railroad tracks. Our requirement is that trains and cars do not collide in the intersection. However, the trains and cars are outside the control of our system; all our system can do is lower the crossing gate


[1]
e that if we implement a system that meets the specification, then that system will satisfy the customer's requirements. Most often, this is simply a check of traceability, where we ensure that each requirement in the definition document is traceable to the specification.

However, for critical systems, we may want to do more, and actually demonstrate that  the  specification  fulfills  the  requirements. This  is  a  more  substantial  effort, in which we prove that the specification realizes every function, event, activity, and constraint in the requirements. The specification by itself is rarely enough to make this kind of argument, because the specification is written in terms of actions performed at the system's interface, such as force applied to an unlocked turnstile, and we may want  to  prove  something  about  the  environment  away  from  the  interface, such  as about the number of entries into the zoo. To bridge this gap, we need to make use of our assumptions  about  how  the  environment  behaves-assumptions  about  what inputs the  system  will  receive, or  about  how  the  environment  will  react  to  outputs (e.g., that if an unlocked turnstile is pushed with sufficient force, it will rotate a halfturn, nudging  the  pusher  into  the  zoo). Mathematically, the  specification  (S)  plus our environmental assumptions (A) must be sufficient to prove that the requirements (R) hold:

<!-- formula-not-decoded -->

For example, to show that a thermostat and furnace will control air temperature, we have to assume that air temperature changes continuously rather than abruptly, although the sensors may detect discrete value changes, and that an operating furnace will raise the air temperature.These assumptions may seem obvious, but if a building is sufficiently porous and the outside temperature is sufficiently cold, then our second assumption will not hold. In such a case, it would be prudent to set some boundaries on the requirement: as long as the outside temperature is above -100ºC, the thermostat and furnace will control the air temperature.

This use of environmental assumptions get


[2]
do not stray into the solution space is to describe requirements and specifications in terms of environmental phenomena.
- There are a variety of sources and means for eliciting requirements. There are both functional and quality requirements to keep in mind.The functional requirements explain what the system will do, and the quality requirements constrain solutions in terms of safety, reliability, budget, schedule, and so on.
- There are many different types of definition and specification techniques. Some are descriptive, such as entity-relationship diagrams and logic, while others are behavioral, such as event traces, data-flow diagrams, and functions. Some have graphical notations, and some are based on mathematics. Each emphasizes a different  view  of  the  problem, and  suggests  different  criteria  for  decomposing  a problem into subproblems. It is often desirable  to  use  a  combination  of  techniques to specify the different aspects of a system.
- The specification techniques also differ in terms of their tool support, maturity, understandability, ease of use, and mathematical formality. Each one should be judged for the project at hand, as there is no best universal technique.
- Requirements questions can be answered using models or prototypes.In either case, the goal is to focus on the subproblem that is at the heart of the question, rather than necessarily modeling or prototyping the entire problem. If prototyping, you need to decide ahead of time whether the resulting software will be kept or thrown away.
- Requirements must be validated to ensure that they accurately reflect the customer's expectations.The requirements should also be checked for completeness,

""",
    "time_sec": 4.7
}


obj2 = {
    "question": "When visually mapping requirements versus specifications, what domain sits exactly at the intersection where the Environment domain overlaps with the System domain?",
    "response": """
The Shared Interface lies at the intersection of the Environment domain and the System domain.
""",
    "retrieved_context": """
    [0]
[SECTION]: Capturing the Requirements > 4.3 Types Of Requirements > Two Kinds of Requirements Documents
[CONTENT]: (footnote: 1 A more intuitive expression of this second requirement, that anyone who pays should be allowed to enter the zoo, is not implementable. There is no way for the system to prevent external factors from keeping the paid visitor from entering the zoo: another visitor may push through the unlocked turnstile before the paid visitor, the zoo may close before the paid visitor enters the turnstile, the paid visitor may decide to leave, and so on (Jackson and Zave 1995).)
[FIGURE 4.3 Requirements vs. Specification.] 
[FIGURE_DESCRIPTIONS]
[0] The diagram presents a layered systems engineering perspective with overlapping ovals representing environments, shared interfaces, system boundaries, and requirements/specifications. A large left ellipse labeled Environment intersects a central oval labeled Shared Interface, which in turn overlaps a smaller inner region labeled Specifications. The outermost gray shape on the right appears to denote the System and includes an overlap with the central Shared Interface. Text labels include Requirements near the top-left inside the large white area, Specifications near the inner overlap, and Environment, Shared Interface, and System along the bottom demarcating system boundaries. The diagram emphasizes the relationships between environmental context, interfaces, and formal requirements/specifications within a system engineering workflow.



[1]
[SECTION]: Capturing the Requirements > 4.8 Requirements Documentation > Requirements Specification
[CONTENT]: The requirements specification covers exactly the same ground as the requirements definition, but from the perspective of the developers. Where the requirements definition  is  written  in  terms  of  the  customer's  vocabulary, referring  to  objects, states, events, and activities in the customer's world, the requirements specification is written in terms of the system's interface.We accomplish this by rewriting the requirements so that they refer only to those real-world objects (states, events, actions) that are sensed or actuated by the proposed system:
 - In  documenting  the  system's  interface, we  describe  all  inputs  and  outputs  in detail, including  the  sources  of  inputs, the  destinations  of  outputs, the  value ranges and data formats of input and output data, protocols governing the order in  which  certain  inputs  and  outputs  must  be  exchanged, window  formats  and organization, and any timing constraints. Note that the user interface is rarely the only system interface; the system may interact with other software components (e.g., a database), special-purpose hardware, the Internet, and so on.
 - Next, we restate the required functionality in terms of the interfaces' inputs and outputs. We may use a functional notation or data-flow diagrams to map inputs to outputs, or use logic to document functions' pre-conditions and post-conditions.We may use state machines or event traces to illustrate exact sequences of operations


[2]
[SECTION]: Capturing the Requirements > 4.8 Requirements Documentation > SIDEBAR 4.7 HIDDEN ASSUMPTIONS
[CONTENT]: Z ave and Jackson (1997) have looked carefully at problems in software requirements and specification, including undocumented assumptions about how the real world behaves.
There are actually two types of environmental behavior of interest: desired behavior to be  realized  by  the  proposed  system  (i.e., the  requirements)  and  existing  behavior  that  is unchanged by the proposed system.The latter type of behavior is often called assumptions or domain knowledge . Most requirements writers consider assumptions to be simply the conditions under which the system is guaranteed to operate correctly.While necessary, these conditions are not the only assumptions. We also make assumptions about how the environment will behave in response to the system's outputs.
Consider a railroad-crossing  gate  at  the  intersection  of  a  road  and  a  set  of  railroad tracks. Our requirement is that trains and cars do not collide in the intersection. However, the trains and cars are outside the control of our system; all our system can do is lower the crossing gate upon the arrival of a train and lift the gate after the train passes. The only way our crossing gate will prevent collisions is if trains and cars follow certain rules. For one thing, we have to assume that the trains travel at some maximum speed, so that we know how early to lower the crossing gate to ensure that the gate is down well before a sensed train reaches the intersection. But we also have to make assumptions about how car drivers will react to the crossing gate being lowered: we have to assume that cars will not stay in or enter the intersection when the gate is down.
or exact orderings of inputs and outputs. We may use an entity-relationship diagram to collect related activities and operations into classes. In the end, the specification  should  be  complete, meaning  that  it  should  specify  an  output  for  any feasible sequence of inputs.Thus, we include validity checks on inputs and system responses to exceptional situations, such as violated pre-conditions.


[3]
[SECTION]: Capturing the Requirements > 4.9 Validation And Verification
[CONTENT]: For example, to show that a thermostat and furnace will control air temperature, we have to assume that air temperature changes continuously rather than abruptly, although the sensors may detect discrete value changes, and that an operating furnace will raise the air temperature.These assumptions may seem obvious, but if a building is sufficiently porous and the outside temperature is sufficiently cold, then our second assumption will not hold. In such a case, it would be prudent to set some boundaries on the requirement: as long as the outside temperature is above -100ºC, the thermostat and furnace will control the air temperature.
This use of environmental assumptions gets at the heart of why the documentation  is  so  important: we  rely  on  the  environment  to  help  us  satisfy  the  customer's requirements, and if our assumptions about how the environment behaves are wrong, then our system may not work as the customer expects. If we cannot prove that our specification  and  our  assumptions fulfill  the  customer's  requirements, then  we  need either to change our specification, strengthen our assumptions about the environment, or  weaken  the  requirements  we  are  trying  to  achieve. Sidebar  4.9  discusses  some techniques for automating these proofs.

""",
    "time_sec": 4.7
}
async def run_three_times(label, scorer_object):
    print(label)

    for run_number in range(1, 4):
        print(f"Run {run_number}")
        await run_experiment(scorer_object)


async def main():
    global llm, ctx_rec_scorer, ctx_prec_scorer

    # Nano minimal
    llm = llm_factory(
        "gpt-5-nano",
        client=openai_client,
        max_tokens=8192,
        reasoning_effort="minimal",
        seed=42,
    )

    ctx_rec_scorer = ContextRecall(llm=llm)
    ctx_prec_scorer = ContextPrecision(llm=llm)

    print("Model: gpt-5-nano")
    print("Reasoning effort: minimal")

    await run_three_times("Fixed size:", obj)
    await run_three_times("HSF:", obj2)

    print("------------------")

    # Nano medium
    llm = llm_factory(
        "gpt-5-nano",
        client=openai_client,
        max_tokens=8192,
        reasoning_effort="medium",
        seed=42,
    )

    ctx_rec_scorer = ContextRecall(llm=llm)
    ctx_prec_scorer = ContextPrecision(llm=llm)

    print("Model: gpt-5-nano")
    print("Reasoning effort: medium")

    await run_three_times("Fixed size:", obj)
    await run_three_times("HSF:", obj2)


asyncio.run(main())



