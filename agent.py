import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(timeout=15.0)

# This is the tool definition — you're telling the model what tools exist,
# what they do, and what inputs they expect. The model reads this and decides
# when to call it.
tools = [
    {
        "type": "web_search_20250305",
        "name": "web_search"
    }
]

def run_agent(question: str):
    print(f"\nQuestion: {question}")
    print("-" * 50)

    messages = [
        {"role": "user", "content": question}
    ]

    input_tokens_total = 0
    output_tokens_total = 0

    # This is the loop. You keep calling the API until the model
    # stops asking for tools and gives you a final text response.
    while True:
        response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=4000,
        system="You are a research assistant specializing in ML and AI. Always use the web_search tool to find current, up-to-date information before answering. Never answer from memory alone — the field moves fast and your training data may be outdated.",
        tools=tools,
        messages=messages
        )
        print(f"Response content blocks: {[block.type for block in response.content]}")
        # Track tokens every turn
        input_tokens_total += response.usage.input_tokens
        output_tokens_total += response.usage.output_tokens

        # stop_reason tells you why the model stopped.
        # "tool_use" means it wants to call a tool.
        # "end_turn" means it's done and giving you a final answer.
        print(f"Stop reason: {response.stop_reason}")

        if response.stop_reason == "end_turn":
            final_text = " ".join(
            block.text for block in response.content
            if hasattr(block, "text")
            )
            print(f"\nAnswer:\n{final_text}")
            break
        if response.stop_reason == "tool_use":
            # The model wants to call a tool. Find the tool_use block.
            tool_use_block = next(
                block for block in response.content
                if block.type == "tool_use"
            )

            print(f"Model is calling tool: {tool_use_block.name}")
            print(f"With input: {tool_use_block.input}")

            # You MUST add the assistant's response (including the tool_use block)
            # back into messages before sending the tool result.
            # This is the part most people miss — the conversation history
            # must stay intact or the API throws an error.
            messages.append({"role": "assistant", "content": response.content})

            # Now add the tool result as a user message.
            # The web_search tool is handled server-side by Anthropic —
            # you don't actually execute the search yourself.
            # You just tell the model "here's what the tool returned"
            # and Anthropic fills in the actual search result automatically.
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_block.id,
                        "content": ""
                    }
                ]
            })

    # Cost calculation
    # Haiku 4.5: $1 per million input tokens, $5 per million output tokens
    input_cost = (input_tokens_total / 1_000_000) * 1.0
    output_cost = (output_tokens_total / 1_000_000) * 5.0
    total_cost = input_cost + output_cost

    print(f"\nTokens used — Input: {input_tokens_total}, Output: {output_tokens_total}")
    print(f"Estimated cost: ${total_cost:.6f}")


if __name__ == "__main__":
    run_agent("What is the current best practice for RAG chunking strategies in 2025?")