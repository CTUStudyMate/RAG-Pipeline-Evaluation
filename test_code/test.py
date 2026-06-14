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

# cr_scorer = AnswerCorrectness(llm=llm, embeddings=embeddings)
ctx_rec_scorer = ContextRecall(llm=llm)
async def test_metric():
    # result = await cr_scorer.ascore(
    #     user_input="When was the first super bowl?",
    #     response="I don't know",
    #     reference="I don't know"
    # )
    
    question = "Look through several issues of software magazines (IEEE Computer and IEEE Software are good choices) from the 1970's, 1980's and recent issues.  Compare the types of problems and solutions described in the older issues with those described in the more  recent issues."
    retrieved_context = """
    
[0]
[SECTION]: Testing the System > 9.16 Key References
[CONTENT]: There have been several special issues of IEEE Software focused on the topics covered in this chapter. The June 1992 and May 1995 issues looked at reliability, the March 1991 issue focused on testing, and the May 2007 issue addressed test-driven development.
The proceedings of the annual International Conference on Software Engineering usually has good papers on the latest in testing theory. For example, Frankl et al. (1997) examine the difference between testing to improve reliability and testing to find faults. Good reference books on testing include Beizer (1990);Kaner,Falk,and Nguyen (1993); and Kit (1995). Each provides a realistic perspective based on industrial experience.
There  are  several  companies  that  evaluate  software  testing  tools  and  publish summaries of their capabilities. For example, Grove Consultants in England, Software Quality Engineering in Florida, and both Cigital and Satisfice in Virginia do regular analyses of testing techniques and tools. You can find these and other resources on the Web to help with requirements analysis and validation, planning and management,simulation, test development, test execution, coverage analysis, source code analysis, and test case generation.
Software dependability and safety-critical systems are receiving more and more attention, and there are many good articles and books about the key issues, including Leveson (1996, 1997). In addition, the Dependable Computing Systems Centre in the Department of Computer Science, University of York, UK, is developing techniques and tools for assessing software dependability. You can get more information from its director, John McDermid, at jam@minster.york.ac.uk.
Usability testing is very important; a system that is correct and reliable but difficult to use may in fact be worse than an easy-to-use but unreliable system. Usability tests and more general usability issues are covered in depth in Hix and Hartson (1993).


[1]
[SECTION]: Testing the Programs > 8.15 Key References
[CONTENT]: Testing has been the subject of several special issues of journals and magazines, including the March 1991 issue of IEEE Software and the June 1988 issue of Communications of  the  ACM . The  September  1994  issue  of Communications  of  the  ACM discusses special  considerations  in  testing  object-oriented  systems. And  the  January/February 2000 issue of IEEE Software addresses why testing is so hard. In addition, IEEE Transactions on Software Engineering often has articles that compare different testing techniques in terms of the kinds of faults they find.
There are several good books that describe testing in great detail. Myers (1979) is the classic text, describing the philosophy of testing as well as several specific techniques. Beizer (1990) offers a good overview of testing considerations and techniques, with many references to key papers in the field. His 1995 book focuses particularly on black-box  testing. Hetzel  (1984)  is  also  a  useful  reference, as  are  Perry  (1995), Kit (1995), and Kaner, Falk, and Nguyen (1993). Binder (2000) is a comprehensive guide to testing object-oriented systems.
There are many good papers describing the use of inspections, including Weller (1993 and 1994) and Grady and van Slack (1994). Gilb and Graham's book (1993) on inspections  is  a  good, comprehensive, and  practical  guide. Researchers  continue  to refine  inspection  techniques, and  to  expand  them  to  other  process  artifacts  such  as requirements. For examples of this work, see Porter et al. (1998) and Shull, Rus, and Basili (2000).


[2]
[SECTION]: Maintaining the System > 11.13 Key References
[CONTENT]: There are few up-to-date textbooks on software maintenance; most information is best sought in journals and conference proceedings. IEEE Software 's  January 1990 issue had maintenance, reverse engineering, and design recovery as its theme; the January 1995 issue focused on legacy systems, and the January 1993 issue has a good article by Wilde, Matthews, and Huitt on the special maintenance problems of object-oriented systems. The  May  1994  issue  of Communications  of  the ACM is  a  special  issue  on reverse engineering. Software Maintenance: Research and Practice is a journal devoted entirely to maintenance issues.
The IEEE Computer Society Press offers some good tutorials on maintenancerelated topics, including one on software reengineering by Arnold (1993), and another on impact analysis by Arnold and Bohner (1996).
Samuelson (1990) explores the legal implications of reverse engineering, asking whether such a practice is equivalent to stealing someone's ideas.
The International Conference on Software Maintenance is held every year, sponsored by the IEEE and ACM. You can order past proceedings from the IEEE Computer Society Press and look at information about the next maintenance conference by viewing the Computer Society Web site.


[3]
[SECTION]: The Future of Software Engineering > 14.6 Key References
[CONTENT]: The  social  science  literature  is  rich  with  information  about  group  decisionmaking. Klein (1998) is a very readable introduction to the recognition-primed decision model. The January/February 2000 issue of IEEE Software addresses the more general question of what we can learn from other disciplines.
For information about the Delphi technique, see Linstone and Turoff (1975), and Turoff and Hiltz (1995).
Discussion  of  software  engineering  licensing  is  addressed  in  the  November/ December 1999 issue of IEEE Software and the May 2000 issue of IEEE Computer . Kaner and Pels (1998) consider who is responsible when software fails, and what you can do as a consumer when your software product does not work properly.
Other excellent references related to software certification and licensing include Allen and Hawthorn (1999), Canadian Engineering Quality Board (2001), Knight and Leveson (1986), Notkin (2000), Parnas (2000), and Shaw (1990). As you will see, many of the guidelines are in draft form, because licensing and accreditation issues are still being discussed in the software engineering community. The Professional Engineers Act, one  of  the  statutes  of  the  province  of  Ontario, can  be  found  at  http://www. e-laws.gov.on.ca/html/statutes/english/elaws_statutes_90p28_e.htm. A  description  of the  University  of  Waterloo  software  engineering  program  is  at  http://www.softeng. uwaterloo.ca An excellent discussion of ethics and the role of personal responsibility can be found in Baase (2002).
    """
   
    reference =  """Wasserman (1995) points out that the changes since the 1970s have been dramatic. For example, early applications were intended to run on a single processor, usually a mainframe. The input was linear, usually a deck of cards or an input tape, and the output was alphanumeric. The system was designed in one of two basic ways: as a transformation, where input was converted to output, or as a transaction, where input determined which function would be performed. Today’s software-based systems are far different and more complex... \n\nIn his Stevens lecture, Wasserman (1996) summarized these changes by identifying seven key factors that have altered software engineering practice... \n1. criticality of time-to-market for commercial products \n2. shifts in the economics of computing: lower hardware costs and greater development and maintenance costs \n3. availability of powerful desktop computing \n4. extensive local- and wide-area networking \n5. availability and adoption of object-oriented technology \n6. graphical user interfaces using windows, icons, menus, and pointers \n7. unpredictability of the waterfall model of software development [5, 6]"""
    recall = await ctx_rec_scorer.ascore(
                user_input=question,
                retrieved_contexts=[retrieved_context],
                reference=reference
            )
    print(f"Recall: {recall.value}")


asyncio.run(test_metric())