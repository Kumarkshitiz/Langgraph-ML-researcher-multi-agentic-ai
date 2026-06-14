import io
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from graph import graph

st.set_page_config(
    page_title="ML Research Agent",
    page_icon="🔬",
    layout="wide"
)

st.title("ML Research Agent")
st.caption("Multi-agent AI system for ML/AI research questions")

question = st.text_input(
    "Enter your research question",
    placeholder="e.g. What are the tradeoffs between LoRA and full fine-tuning in 2025?"
)

run_button = st.button("Run Research", type="primary")

if run_button and question.strip():

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Agent Activity")
        activity_container = st.empty()

    with col2:
        st.subheader("Cost Tracker")
        cost_container = st.empty()

    activity_log = []
    last_state = {}

    initial_state = {
        "question": question,
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

    try:
        with st.spinner("Research in progress..."):
            for event in graph.stream(initial_state, stream_mode="updates"):
                node_name = list(event.keys())[0]
                node_state = event[node_name]
                last_state.update(node_state)

                if node_name == "planner":
                    subs = node_state.get("sub_questions", [])
                    activity_log.append(f"✅ **Planner** — Generated {len(subs)} sub-questions")
                    for i, q in enumerate(subs):
                        activity_log.append(f"&nbsp;&nbsp;&nbsp;&nbsp;{i+1}. {q}")

                elif node_name == "researcher":
                    findings = node_state.get("findings", [])
                    idx = len(findings)
                    activity_log.append(f"✅ **Researcher {idx}** — Done")

                elif node_name == "critic":
                    needs_retry = node_state.get("needs_retry", False)
                    if needs_retry:
                        feedback = node_state.get("critic_feedback", "")
                        activity_log.append(f"⚠️ **Critic** — Flagged a finding, retrying")
                        activity_log.append(f"&nbsp;&nbsp;&nbsp;&nbsp;_{feedback[:120]}..._")
                    else:
                        activity_log.append("✅ **Critic** — All findings approved")

                elif node_name == "writer":
                    activity_log.append("✅ **Writer** — Report complete")

                activity_container.markdown("\n\n".join(activity_log))

                input_tokens = last_state.get("total_input_tokens", 0)
                output_tokens = last_state.get("total_output_tokens", 0)
                if input_tokens > 0:
                    input_cost = (input_tokens / 1_000_000) * 1.0
                    output_cost = (output_tokens / 1_000_000) * 5.0
                    total_cost = input_cost + output_cost
                    cost_container.metric(
                        "Estimated Cost",
                        f"${total_cost:.4f}",
                        f"{input_tokens + output_tokens:,} tokens"
                    )

    except TimeoutError:
        st.error("A researcher timed out. Try a narrower question or run again.")
    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")
        st.stop()

    st.divider()
    st.subheader("Final Report")

    final_report = last_state.get("final_report")
    if final_report:
        st.markdown(final_report)

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        for line in final_report.split("\n"):
            if line.strip():
                story.append(Paragraph(line.replace("#", "").strip(), styles["Normal"]))
                story.append(Spacer(1, 6))

        doc.build(story)
        buffer.seek(0)

        st.download_button(
            label="Download Report (PDF)",
            data=buffer,
            file_name="research_report.pdf",
            mime="application/pdf"
        )

elif run_button and not question.strip():
    st.warning("Please enter a research question.")