from ragas.metrics import DiscreteMetric
from openai import AsyncOpenAI
from ragas.llms import llm_factory

import os
from dotenv import load_dotenv
import asyncio
from ragas.metrics.collections import ContextRecall, ContextPrecision


load_dotenv()

ollama_client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # dummy, không quan trọng
)
llm = llm_factory("qwen3.5:cloud", client=ollama_client, max_tokens=12000)

# correctness = DiscreteMetric(
#     name="correctness",
#     prompt=(
#         "Check if the response contains points mentioned from the grading notes "
#         "and return 'pass' or 'fail'.\nResponse: {response} Grading Notes: {grading_notes}"
#     ),
#     allowed_values=["pass", "fail"],
# )

ctx_rec_scorer = ContextRecall(llm=llm)
ctx_prec_scorer = ContextPrecision(llm=llm)

# ---------- Experiment ----------

async def run_experiment(row):
    """
    row: dict with 'question', 'response', 'retrieved_context', 'time_sec'
    """
    question = row.get("question", "")
    response = row.get("response", "")
    retrieved_context = row.get("retrieved_context", "")
    
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
            retrieved_contexts=[retrieved_context]
        )
        recall = await ctx_rec_scorer.ascore(
            user_input=question,
            retrieved_contexts=[retrieved_context],
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
    print(x)
        
    return 


# 1. Cấu trúc lại biến mark thành dictionary 2 lớp (lấy câu hỏi làm key)
global mark

mark = {
    "The “Millenium Bug” or “Y2K problem” is perhaps the most infamous software maintenance problem. Find a discussion of the Y2K problem written for nonscientists. How many of the maintenance problems listed in section 11.3 are accurately presented in the article?": {
        "grading_notes": "Many of the issues in section 11.3 relate in some way to the Y2K problem. Some examples: The limits of human understanding are certainly applicable. There is a definite limit to how quickly maintainers can approach a system that is unfamiliar to them and understand enough about it to make the correct changes for a maintenance problem. That difficulty is compounded when the system being maintained is old and the chances of missing documentation or even source code have increased. Management priorities have been a major contribution to the problem. Since Y2K maintenance does not result in a new product but rather keeps an old product running, management in many cases did not assign a high priority to maintenance in general and Y2K maintenance in particular. As a result, Y2K was often not a high priority until very close to the year 2000, when the problem was no longer avoidable. Morale has been a problem in some cases, in which software practitioners were assigned part-time to handle the Y2K problem in addition to their other duties. This type of situation tends to reinforce the belief that Y2K maintenance is not an important or interesting task.",
        "ground_truth": "The textbook explicitly cites the \"year 2000 problem\" as a primary example of how \"simple but narrow design decisions can have a major effect on maintenance\" [1]. Most popular, non-scientist articles discussing Y2K successfully captured the technical difficulty of the problem, aligning with Section 11.3's concept of \"Limited Understanding\"\u2014specifically that there is a limit to the rate at which maintainers can study documentation and extract material to solve a problem in an old, unfamiliar legacy system [2, 3]. However, popular articles often missed the systemic organizational issues that contributed to the crisis. For example, they rarely discussed \"Management Priorities,\" where managers often override technical priorities to focus on new business applications, delaying necessary repairs (like Y2K fixes) until they become unavoidable [4]. Furthermore, popular media typically ignored the problem of \"Morale.\" Maintainers are often pulled in many directions and suffer from low morale because maintenance is sometimes viewed as a second-class status compared to new development [5, 6]. These management and personnel issues severely compounded the Y2K technical problem.",
        "source_text": "The \"year 2000 problem,\" where many developers represented the year as only two characters, is a good example of how simple but narrow design decisions can have a major effect on maintenance. [1]\nThere is a limit to the rate at which a person can study documentation and extract material relevant to the problem being solved. [2] Parikh and Zvegintzov (1983) report that 47% of software maintenance effort is devoted to understanding the software to be modified. [3]\nManagement priorities often override technical ones; managers sometimes view maintaining and enhancing as more important than building new applications. In other words, companies must sometimes focus on business as usual, instead of investigating new alternatives. [4]\nThe Lientz and Swanson studies (1981) indicate that 11.9% of the problems during maintenance result from low morale and productivity. A major reason for low morale is the second-class status often accorded the maintenance team. [5]"
    }
}

# 2. obj chứa câu hỏi và ngữ cảnh mà LLM đã trả lời
obj = {
    "question": "The “Millenium Bug” or “Y2K problem” is perhaps the most infamous software maintenance problem. Find a discussion of the Y2K problem written for nonscientists. How many of the maintenance problems listed in section 11.3 are accurately presented in the article?",
    "response": """
    "The “Millennium Bug” or “Y2K problem” was a widely publicized software maintenance issue related to how dates were represented in computer systems. Many older programs stored years using only the last two digits (e.g., ""99"" for 1999), which caused concerns that when the year rolled over to 2000 (""00""), systems would misinterpret it as 1900, potentially leading to errors in calculations, data processing, and system operations.

A good discussion of the Y2K problem written for nonscientists can be found in popular science articles, technology magazines, or reputable websites such as:

- The BBC’s “The Y2K Bug: What Was It All About?”  
- The Smithsonian Magazine’s article “The Y2K Bug: How the World Prepared for the Millennium Bug”  
- The Computer History Museum’s overview of Y2K

These articles explain the problem in accessible language, describing the potential risks, the global efforts to fix the issue, and the eventual minimal impact due to extensive maintenance and testing.

Regarding the maintenance problems listed in section 11.3 (assuming this refers to a typical software engineering textbook section on software maintenance issues), common maintenance problems include:

1. **Corrective Maintenance** – fixing bugs and errors found after deployment.  
2. **Adaptive Maintenance** – modifying software to work in a changed environment.  
3. **Perfective Maintenance** – improving performance or maintainability.  
4. **Preventive Maintenance** – updating software to prevent future problems.

The Y2K problem primarily involved **corrective** and **adaptive maintenance**:

- **Corrective**, because many date-related bugs had to be fixed.  
- **Adaptive**, because software had to be updated to handle the new century date format and comply with new standards.

Additionally, the Y2K effort involved **preventive maintenance**, as organizations proactively updated systems to avoid future failures.

In the nonscientific articles, these maintenance types are often described in terms of ""fixing bugs,"" ""updating systems,"" and ""preventing errors,"" which correspond well to the maintenance categories in section 11.3.

Therefore, the maintenance problems related to Y2K as presented in these articles accurately reflect the categories of maintenance problems listed in section 11.3, though the articles may not use the formal terminology. They effectively communicate the nature of the maintenance work involved in addressing the Y2K problem.
    """,
    "retrieved_context": """
    [Chunk 1]
programmer's time result  in  conflicting  priorities. During  maintenance, 8%  of  the  problems  result from  a  programmer's  being  pulled  in  too  many  directions  at  once  and  thus  being unable to concentrate on one problem long enough to solve it.

## Technical Problems

Technical problems also affect maintenance productivity. Sometimes, they are a legacy of what developers and maintainers have done before.At other times, they result from particular paradigms or processes that have been adopted for the implementation.

Artifacts and Paradigms. If the design's logic is not obvious, the team may not easily determine whether the design can handle proposed changes.A flawed or inflexible design can require extra time for understanding, changing, and testing. For instance, developers may have included a component for input and output that handles only tape; major modifications must be made for disk access, because the disk is not constrained by the tape's sequential access. Similarly, the developers may not have anticipated changes; field and table sizes may be fixed, making them difficult to modify.The 'year 2000 problem,' where many developers represented the year as only two characters, is a good example of how simple but narrow design decisions can have a major effect on maintenance.

Maintaining  object-oriented  programs  can  be  problematic, too, because  the design often involves components that are highly interconnected by complex inheritance schemes. Incremental changes must be made with great care, since modifications can result in long chains of classes that hide others or that redefine objects in conflicting ways. Sidebar 11.2 describes more of the particular design trade-offs involved when maintaining object-oriented systems.

I

[Chunk 2]
given organization depends on many things, including whether the system is an S-, P-, or E-system, and how quickly business needs change.

## 11.3 MAINTENANCE PROBLEMS

Maintaining a system is difficult. Because the system is already operational, the maintenance team balances the need for change with the need for keeping a system accessible to users. For example, upgrading a system may require it to be unavailable to users for several hours. However, if the system is critical to the users' business or operation, there may not be a window of several hours when users can give up the system. For instance, a life-support system cannot be disconnected from a patient so that maintenance can be performed on the software. The maintenance team must find a way to implement changes without inconveniencing users.

## Staff Problems

There are many staff and organizational reasons that make maintenance difficult. The staff must act as an intermediary between the problem and its solution, tinkering and tailoring the software to ensure that the solution follows the course of the problem as it changes.

Limited Understanding. In addition to balancing user needs with software and hardware needs, the  maintenance  team  deals  with  the  limitations  of  human  understanding. There is a limit to the rate at which a person can study documentation and extract material relevant to the problem being solved. Furthermore, we usually look for more clues than are really necessary for solving a problem.Adding the daily office distractions, we have a prescription for limited productivity.

Parikh and Zvegintzov (1983) report that 47% of software maintenance effort is devoted to understanding the software to be modified. This high figure is understandable when we consider the nu
""",
    "time_sec": 4.7
}

# 3. Chạy hàm
asyncio.run(run_experiment(obj))



