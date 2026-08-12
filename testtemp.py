import re
import json
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

context = """
[0]
= . shelf, 5 = X. shelf, 6 = X. shelf, 7 = X. shelf, 8 = . Reject event, 1 = . Reject event, 2 = X. Reject event, 3 = X. Reject event, 4 = . Reject event, 5 = . Reject event, 6 = . Reject event, 7 = . Reject event, 8 = 

## Example: Decision Tables

A decision table (Hurley 1983) is a tabular representation of a functional specification that maps events and conditions to appropriate responses or actions. We say that the specification  is informal because  the  inputs  (events  and  conditions)  and  outputs (actions) may be expressed in natural language, as mathematical expressions, or both.

Figure 4.16 shows a decision table for the library functions borrow , return , reserve , and unreserve . All of the possible input events (i.e., function invocations), conditions, and actions are listed along the left side of the table, with the input events and conditions listed above the horizontal line and the actions listed below the line. Each  column  represents  a  rule  that  maps  a  set  of  conditions  to  its  corresponding result(s). An entry of 'T' in a cell means that the row's input condition is true, 'F' means that the input condition is false, and a dash indicates that the value of the condition does not matter. An entry of 'X' at the bottom of the table means that the row's action should be performed whenever its corresponding input conditions hold. Thus, column 1 represents the situation where a library patron wants to borrow a book, the book is not already out on loan, and the patron has no outstanding fine; in this situation, the loan is approved and a due date is calculated. Similarly, column 7 illustrates the case where there is a request to put a book on reserve but the book is currently out on loan; in this case, the book is recalled and the due date is recalculated to reflect the recall.

This kind of representation can result in very large tables, because the number of conditions to consider is equal to the number of combinations of input conditions.That is, if there are n input conditions, there are possible combinations of conditions. Fortunately, many combinations map to the same


[1]
oan and needs to be recalled; this behavior cannot be modeled as a transition from onloan to reserveloan , because state reserveloan has a transition cancel (used to disallow a loan request if the Patron has outstanding fines) that would be inappropriate in this situation.This special case is modeled in Figure 4.9 by testing on entry (keyword entry is explained below) to state reserve whether the concurrent submachine is In state onloan and issuing a recall event if it is.

FIGURE 4.10 Messy UML statechart diagram for Publication class.

State transitions are labeled with their enabling events and conditions and with their side effects.Transition labels have syntax

```
event(args) [condition] /action* ^Object.event(args)*
```

where  the  triggering event is  a  message  that  may  carry  parameters. The  enabling condition , delimited by square brackets, is a predicate on the object's attribute values. If the transition is taken, its actions , each prefaced with a slash (/), specify assignments made to the object's attributes; the asterisk indicates that a transition may have arbitrarily  many  actions. If  the  transition  is  taken, it  may  generate  arbitrarily  many output events , /^Object.event , each prefaced with a caret an output event may carry parameters and is either designated for a target Object or is broadcast to all objects. For example, in the messy Publication statechart (Figure 4.10), the transition to state recall is enabled if the publication is in state onloan when a request to put the item on reserve is received.When the transition is taken, it sends an event to the Loan object, which in turn will notify the borrower that the item must be returned to the library sooner than the loan's due date. Each of the transition-label elements is optional. For example, a transition need not be enabled by an input event; it could be enabled  only  by  a  condition  or  by  nothing, in  which  case  the  transition  is  always enabled. 1 ¿ 2 ; ' '*''

The UML statechart diagram for the Loan association class in Figure 4.11 illustrates how states can be annotated with local variables (e.g


[2]
serve (event) unreserve, 5 = F. (event) borrow (event) return (event) reserve (event) unreserve, 6 = F. (event) borrow (event) return (event) reserve (event) unreserve, 7 = F. (event) borrow (event) return (event) reserve (event) unreserve, 8 = F. , 1 = F. , 2 = F. , 3 = F. , 4 = T. , 5 = T. , 6 = F. , 7 = F. , 8 = F. , 1 = F. , 2 = F. , 3 = F. , 4 = F. , 5 = F. , 6 = T. , 7 = T. , 8 = F. , 1 = F. , 2 = F. , 3 = F. , 4 = F. , 5 = F. , 6 = F. , 7 = F. , 8 = T. item out on loan, 1 = F. item out on loan, 2 = T. item out on loan, 3 = -. item out on loan, 4 = -. item out on loan, 5 = -. item out on loan, 6 = F. item out on loan, 7 = T. item out on loan, 8 = F. item on reserve, 1 = -. item on reserve, 2 = -. item on reserve, 3 = -. item on reserve, 4 = F. item on reserve, 5 = T. item on reserve, 6 = -. item on reserve, 7 = -. item on reserve, 8 = -. patron.fines > $0.00, 1 = F. patron.fines > $0.00, 2 = -. patron.fines > $0.00, 3 = T. patron.fines > $0.00, 4 = -. patron.fines > $0.00, 5 = -. patron.fines > $0.00, 6 = -. patron.fines > $0.00, 7 = -. patron.fines > $0.00, 8 = -. (Re-)Calculate due date Put item in stacks Put item on reserve Send recall notice, 1 = X. (Re-)Calculate due date Put item in stacks Put item on reserve Send recall notice, 2 = . (Re-)Calculate due date Put item in stacks Put item on reserve Send recall notice, 3 = . (Re-)Calculate due date Put item in stacks Put item on reserve Send recall notice, 4 = X. (Re-)Calculate due date Put item in stacks Put item on reserve Send recall notice, 5 = . (Re-)Calculate due date Put item in stacks Put item on reserve Send recall notice, 6 = . (Re-)Calculate due date Put item in stacks Put item on reserve Send recall notice, 7 = X. (Re-)Calculate due date Put item in stacks Put item on reserve Send recall notice, 8 = X. shelf, 1 = . shelf, 2 = . shelf, 3 = . shelf, 4 = . shelf, 5 = X. shelf, 6 = X. shelf, 7 = X. shelf, 8 = . Reject event, 1 = . Reject event, 2 = X. Reject event, 3 = X. Reject event, 4 = . Reject event, 5 = . Reject event, 6 = . Reject event, 7 = . Reject event, 8 = 

## Example: Decision Tables

A decision table (Hurley 1983) is
"""

contexts = parse_retrieved_contexts(context)
print(len(contexts))
# print(contexts)
