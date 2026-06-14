# ML Research Agent

A multi-agent AI system that takes any ML/AI research question and produces a structured, sourced report with contradiction checking, source verification, and concrete recommendations.

**Live demo**: [link goes here after Phase 5 deployment]

---

## What it does

You type a question like:

> "What are the tradeoffs between LoRA and full fine-tuning?"

The system:

1. Breaks the question into focused sub-questions
2. Searches the web for current information on each sub-question
3. Reviews findings for contradictions, gaps, and outdated information
4. Sends weak findings back for a targeted retry
5. Synthesizes everything into a structured report with recommendations

The output is a downloadable PDF report containing:

* Executive Summary
* Key Findings
* Contradictions & Risks
* Open Questions
* Actionable Recommendations
* Relevant Code Examples (when applicable)

---

## Architecture

```text
User Question
     │
     ▼
  Planner (Haiku)
  Breaks question into 3 focused sub-questions
     │
     ▼
  Researcher x3 (Haiku + Web Search)
  Each researcher investigates one sub-question
     │
     ▼
  Critic (Haiku)
  Reviews all findings together
  Flags contradictions, weak evidence,
  and missing information
  Can trigger one targeted retry
     │
     ▼
  Writer (Haiku)
  Produces final structured report
     │
     ▼
  PDF Report + Streamlit UI
```

The orchestration layer is built using LangGraph. Each agent is represented as a node, shared state is passed through the graph, and conditional routing allows the Critic to either:

* Send a researcher back for one retry
* Approve findings and move to report generation

---

## Design Decisions

### Why a Critic Agent?

Most multi-agent research systems move directly from Researchers to a Writer.

This project introduces a Critic layer that validates findings before report generation.

Benefits:

* Detects contradictory claims
* Flags weak or unsupported evidence
* Identifies missing information
* Improves reliability of final reports

In testing, the Critic frequently identified conflicting findings that would otherwise have appeared in the final report.

---

### Why Claude Haiku for Every Agent?

The primary goal was achieving strong quality while maintaining cost efficiency.

Approximate run cost:

* Haiku pipeline: ~$0.15–0.20
* Equivalent Sonnet pipeline: ~$1.00+

For structured tasks such as:

* Planning
* Summarization
* JSON generation
* Critique
* Report writing

Haiku provides sufficient quality at a fraction of the cost.

Higher-end models can be swapped in later if required.

---

### Why Build the Tool-Use Loop Manually First?

LangGraph abstracts tool execution.

Before using the abstraction, Phase 1 implemented the complete loop manually:

1. Model requests a tool
2. Tool executes
3. Result is returned
4. Model continues reasoning

Understanding this flow makes debugging significantly easier when working inside agent frameworks.

---

### Why Limit Retries to One?

Unlimited retries create:

* Higher costs
* Longer latency
* Potential infinite loops

One retry provides a practical balance between:

* Research quality
* Runtime
* Cost control

---

### Why LangGraph?

The system requires:

* Shared state
* Conditional routing
* Retry loops
* Agent coordination

A simple sequential pipeline cannot express this behavior cleanly.

LangGraph provides:

* State management
* Node-based execution
* Conditional edges
* Scalable agent orchestration

without requiring custom workflow infrastructure.

---

## Tech Stack

| Component       | Technology                  |
| --------------- | --------------------------- |
| LLM             | Claude Haiku 4.5            |
| Agent Framework | LangGraph                   |
| Web Search      | Anthropic Native Web Search |
| Frontend        | Streamlit                   |
| PDF Generation  | ReportLab                   |
| Environment     | Python 3.10+                |

---

## Estimated Cost

| Component         | Model          | Approx Cost     |
| ----------------- | -------------- | --------------- |
| Planner           | Haiku          | ~$0.01          |
| Researchers ×3    | Haiku + Search | ~$0.08          |
| Critic            | Haiku          | ~$0.02          |
| Retry (if needed) | Haiku + Search | ~$0.03          |
| Writer            | Haiku          | ~$0.02          |
| **Total**         |                | **~$0.15–0.20** |

Additional web search charges may apply depending on API pricing.

---

## Running Locally

### Prerequisites

* Python 3.10+
* Anthropic API Key
* Web Search enabled

### Clone Repository

```bash
git clone https://github.com/yourusername/ml-research-agent
cd ml-research-agent
```

### Create Virtual Environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux / Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file:

```env
ANTHROPIC_API_KEY=your_api_key_here
```

### Run Application

```bash
streamlit run app.py
```

The Streamlit interface will open in your browser.

---

## Known Limitations

### Search Reliability

Occasionally web search requests may timeout or return incomplete results.

The application attempts graceful degradation and returns partial findings rather than failing entirely.

### Critic Accuracy

The Critic currently uses Haiku.

While effective, it may miss nuanced contradictions that larger models could identify.

Upgrading only the Critic node to Sonnet is a simple improvement path.

### No Caching

Every query is processed from scratch.

This improves freshness but increases token usage for repeated questions.

### API Timeouts

Long-running searches may be interrupted under poor network conditions.

### PDF Formatting

The PDF focuses on readability rather than preserving full markdown styling.

The Streamlit report view generally provides richer formatting.

---

## Project Structure

```text
ml-research-agent/
│
├── agent.py
│   └── Phase 1 manual tool-use loop
│
├── graph.py
│   └── LangGraph workflow and agent nodes
│
├── app.py
│   └── Streamlit user interface
│
├── requirements.txt
│
├── .env
│   └── API key (never commit)
│
├── .gitignore
│
└── README.md
```

---

## Future Improvements

* Multi-source verification scoring
* Citation confidence metrics
* Research history and caching
* Parallel web search optimization
* Multiple report export formats
* Sonnet-powered Critic mode
* User feedback driven report refinement

---

## License

MIT License
