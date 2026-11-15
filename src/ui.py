import gradio as gr
import requests

from generate import answer

EXAMPLES = [
    "What was the best F1 in the sarcasm paper?",
    "Which model achieved the best F1 on DailyDialog?",
    "What data sources were used to predict Bitcoin price direction?",
]


def format_reply(result):
    if not result["sources"]:
        return result["answer"]
    sources = "\n".join(f"- {s['paper']}, page {s['page']}" for s in result["sources"])
    return f"{result['answer']}\n\n**Sources**\n{sources}"


def chat(message, history):
    try:
        return format_reply(answer(message))
    except requests.RequestException as error:
        return f"The language model is not reachable: {error}"


demo = gr.ChatInterface(
    chat,
    title="PaperRAG",
    description="Ask questions about my three research papers. Every answer cites the paper and page it came from.",
    examples=EXAMPLES,
    analytics_enabled=False,
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1")
