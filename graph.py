from typing import TypedDict, List, Optional
import os
import json
from anthropic import Anthropic
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

load_dotenv()
client = Anthropic(timeout=30.0)

tools = [
    {
        "type": "web_search_20250305",
        "name": "web_search"
    }
]

class ResearchState(TypedDict):
    question: str
    sub_questions: List[str]
    findings: List[dict]
    needs_retry: bool
    retry_index: Optional[int]
    critic_feedback: Optional[str]
    retry_count: int
    final_report: Optional[str]
    total_input_tokens: int
    total_output_tokens: int


def planner_node(state: ResearchState) -> ResearchState:
    print("\n[Planner] Breaking down the question...")

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=500,
        system="""You are a research planner. Break down a complex ML/AI question into exactly 3 focused sub-questions.

Return ONLY a JSON array of sub-questions, nothing else. No explanation, no markdown, no backticks.
Example: ["What is X?", "How does Y work?", "What are the tradeoffs of Z?"]""",
        messages=[
            {"role": "user", "content": f"Break down this question into exactly 3 sub-questions:\n\n{state['question']}"}
        ]
    )

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    raw = response.content[0].text.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        sub_questions = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[Planner] JSON parse failed. Raw: {raw[:100]}")
        sub_questions = [state["question"]]

    print(f"[Planner] Generated {len(sub_questions)} sub-questions:")
    for i, q in enumerate(sub_questions):
        print(f"  {i+1}. {q}")

    return {
        **state,
        "sub_questions": sub_questions,
        "findings": [],
        "needs_retry": False,
        "retry_index": None,
        "critic_feedback": None,
        "retry_count": 0,
        "final_report": None,
        "total_input_tokens": state["total_input_tokens"] + input_tokens,
        "total_output_tokens": state["total_output_tokens"] + output_tokens
    }


def researcher_node(state: ResearchState) -> ResearchState:
    findings = list(state["findings"])

    if state["needs_retry"] and state["retry_index"] is not None:
        index = state["retry_index"]
        sub_question = state["sub_questions"][index]
        print(f"\n[Researcher {index+1}] RETRY — {sub_question}")
        user_message = f"""Research this question: {sub_question}

The previous answer was flagged with this feedback:
{state['critic_feedback']}

Search again with this feedback in mind."""
    else:
        index = len(findings)
        sub_question = state["sub_questions"][index]
        print(f"\n[Researcher {index+1}] Researching — {sub_question}")
        user_message = f"Research this question and give 3-5 concise findings with specific facts: {sub_question}"

    messages = [{"role": "user", "content": user_message}]
    input_tokens = 0
    output_tokens = 0
    final_text = "Research incomplete due to timeout."

    try:
        while True:
            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=800,
                system="""You are a research specialist in ML and AI.
Always use web search to find current information.
Be concise — 3-5 key findings with specific facts and numbers only.
No lengthy explanations. The writer agent will expand on your findings.""",
                tools=tools,
                messages=messages
            )

            input_tokens += response.usage.input_tokens
            output_tokens += response.usage.output_tokens

            if response.stop_reason == "end_turn":
                final_text = " ".join(
                    block.text for block in response.content
                    if hasattr(block, "text")
                )
                break

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": next(
                                block.id for block in response.content
                                if block.type == "server_tool_use"
                            ),
                            "content": ""
                        }
                    ]
                })

    except Exception as e:
        print(f"[Researcher {index+1}] Error: {e}, using partial results.")

    finding = {
        "index": index,
        "sub_question": sub_question,
        "answer": final_text
    }

    if state["needs_retry"] and state["retry_index"] is not None:
        findings[index] = finding
        print(f"[Researcher {index+1}] Retry complete, finding updated.")
    else:
        findings.append(finding)
        print(f"[Researcher {index+1}] Done.")

    return {
        **state,
        "findings": findings,
        "needs_retry": False,
        "retry_index": None,
        "total_input_tokens": state["total_input_tokens"] + input_tokens,
        "total_output_tokens": state["total_output_tokens"] + output_tokens
    }


def critic_node(state: ResearchState) -> ResearchState:
    print("\n[Critic] Reviewing all findings...")

    findings_text = ""
    for finding in state["findings"]:
        findings_text += f"\nSub-question: {finding['sub_question']}\n"
        findings_text += f"Answer: {finding['answer']}\n"
        findings_text += "-" * 40 + "\n"

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=500,
        system="""You are a critical reviewer of ML/AI research findings.
Review the findings and identify the single most important problem if one exists.

Problems to look for:
- Contradictions between findings
- Vague answers with no specific facts or numbers
- Outdated information (pre-2023)
- A finding that completely missed the point of its sub-question

Respond ONLY in JSON. No explanation, no markdown, no backticks.

If findings are acceptable:
{"needs_retry": false, "retry_index": null, "feedback": null}

If one finding needs improvement:
{"needs_retry": true, "retry_index": 0, "feedback": "specific explanation of what was wrong and what to look for instead"}

retry_index is the zero-based index of the finding that needs to be redone.""",
        messages=[
            {
                "role": "user",
                "content": f"Original question: {state['question']}\n\nFindings to review:\n{findings_text}"
            }
        ]
    )

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    raw = response.content[0].text.strip()

# Strip markdown backticks if Haiku added them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        verdict = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[Critic] JSON parse failed, defaulting to no retry. Raw: {raw[:100]}")
        verdict = {"needs_retry": False, "retry_index": None, "feedback": None}

    if verdict["needs_retry"] and state["retry_count"] < 1:
        print(f"[Critic] Flagged finding {verdict['retry_index'] + 1}: {verdict['feedback']}")
        return {
            **state,
            "needs_retry": True,
            "retry_index": verdict["retry_index"],
            "critic_feedback": verdict["feedback"],
            "retry_count": state["retry_count"] + 1,
            "total_input_tokens": state["total_input_tokens"] + input_tokens,
            "total_output_tokens": state["total_output_tokens"] + output_tokens
        }
    else:
        print("[Critic] Findings look good, proceeding to Writer.")
        return {
            **state,
            "needs_retry": False,
            "retry_index": None,
            "critic_feedback": None,
            "total_input_tokens": state["total_input_tokens"] + input_tokens,
            "total_output_tokens": state["total_output_tokens"] + output_tokens
        }


def writer_node(state: ResearchState) -> ResearchState:
    print("\n[Writer] Synthesizing final report...")

    findings_text = ""
    for finding in state["findings"]:
        findings_text += f"\nSub-question: {finding['sub_question']}\n"
        findings_text += f"Answer: {finding['answer']}\n"
        findings_text += "-" * 40 + "\n"

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=2000,
        system="""You are a technical writer specializing in ML and AI research.
Synthesize research findings into a structured report with these sections:
1. Executive Summary (2-3 sentences)
2. Key Findings (one section per sub-question)
3. Open Questions
4. Recommendations (concrete and actionable)

Write for a technical ML engineer audience. Be specific, not generic.""",
        messages=[
            {
                "role": "user",
                "content": f"Original question: {state['question']}\n\nResearch findings:\n{findings_text}\n\nWrite the final report."
            }
        ]
    )

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    final_report = response.content[0].text.strip()
    print("[Writer] Report complete.")

    return {
        **state,
        "final_report": final_report,
        "total_input_tokens": state["total_input_tokens"] + input_tokens,
        "total_output_tokens": state["total_output_tokens"] + output_tokens
    }


def should_continue_researching(state: ResearchState) -> str:
    if len(state["findings"]) < len(state["sub_questions"]):
        return "researcher"
    return "critic"


def should_retry(state: ResearchState) -> str:
    if state["needs_retry"]:
        return "researcher"
    return "writer"


builder = StateGraph(ResearchState)

builder.add_node("planner", planner_node)
builder.add_node("researcher", researcher_node)
builder.add_node("critic", critic_node)
builder.add_node("writer", writer_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "researcher")

builder.add_conditional_edges(
    "researcher",
    should_continue_researching,
    {
        "researcher": "researcher",
        "critic": "critic"
    }
)

builder.add_conditional_edges(
    "critic",
    should_retry,
    {
        "researcher": "researcher",
        "writer": "writer"
    }
)

builder.add_edge("writer", END)

graph = builder.compile()


if __name__ == "__main__":
    initial_state = {
        "question": "What is LoRA and when should you use it?",
        "sub_questions": [],
        "findings": [],
        "needs_retry": False,
        "retry_index": None,
        "critic_feedback": None,
        "retry_count": 0,
        "final_report": None,
        "total_input_tokens": 0,
        "total_output_tokens": 0
    }

    final_state = graph.invoke(initial_state)

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(final_state["final_report"])

    input_cost = (final_state["total_input_tokens"] / 1_000_000) * 1.0
    output_cost = (final_state["total_output_tokens"] / 1_000_000) * 5.0
    total_cost = input_cost + output_cost

    print("\n" + "=" * 60)
    print(f"Total tokens — Input: {final_state['total_input_tokens']}, Output: {final_state['total_output_tokens']}")
    print(f"Estimated cost: ${total_cost:.6f}")