import gradio as gr
from rag import context_retrieval, hgf_generator, openrouter_generator
import time
import os
import shutil


def query_local_hgf_llm(history):
    """
    Processes a user question by retrieving relevant context and generating a response.

    This function first retrieves context related to the user's question using the
    `context_retrieval` function, then generates a response using the `hgf_generator`
    function. Both context retrieval and response generation are handled locally.

    Args:
        user_question (str): The user's input question.

    Returns:
        str: The generated response from the assistant.
    """
    user_question = history[-1]["content"] if history else ""
    print("User question: ", user_question)
    context = context_retrieval(user_question)
    bot_response = hgf_generator(user_question, context)
    history.append({"role": "assistant", "content": ""})
    for character in bot_response:
        history[-1]["content"] += character
        time.sleep(0.01)
        yield history
    return bot_response


def query_openrouter_llm(history: list):
    """
    Processes a user question by retrieving relevant context and generating a response using OpenRouter.

    This function first retrieves context related to the user's question using the
    `context_retrieval` function, then generates a response using the `openrouter_generator`
    function. Both context retrieval and response generation are handled via OpenRouter API.

    Args:
        user_question (str): The user's input question.

    Returns:
        str: The generated response from the assistant.
    """
    user_question = history[-1]["content"] if history else ""

    print("User question: ", user_question)
    context = context_retrieval(user_question)
    bot_response = openrouter_generator(user_question, context)
    history.append({"role": "assistant", "content": ""})
    for character in bot_response:
        history[-1]["content"] += character
        time.sleep(0.01)
        yield history
    return bot_response


def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)


UPLOAD_DIR = "./uploads"


def save_uploaded_file(temp_path):
    """Save a temporary uploaded file to a permanent location and return the new path."""
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    filename = os.path.basename(temp_path)
    dest_path = os.path.join(UPLOAD_DIR, filename)
    # Avoid overwriting files with the same name
    base, ext = os.path.splitext(filename)
    counter = 1
    while os.path.exists(dest_path):
        dest_path = os.path.join(UPLOAD_DIR, f"{base}_{counter}{ext}")
        counter += 1
    shutil.copy(temp_path, dest_path)
    return dest_path


def add_message(history, message):
    # Save each uploaded file and update the path in the message
    for temp_path in message["files"]:
        saved_path = save_uploaded_file(temp_path)
        history.append({"role": "user", "content": {"path": saved_path}})
    if message["text"] is not None:
        history.append({"role": "user", "content": message["text"]})
    return history, gr.MultimodalTextbox(value=None, interactive=False)


def interactive_smart_assistant(fn=query_openrouter_llm):
    with gr.Blocks(theme="earneleh/paris") as pki_expert:
        chatbot = gr.Chatbot(elem_id="chatbot", type="messages")

        chat_input = gr.MultimodalTextbox(
            interactive=True,
            file_count="multiple",
            placeholder="Enter message or upload file...",
            show_label=False,
            sources=["microphone", "upload"],
        )
        chat_msg = chat_input.submit(
            add_message, [chatbot, chat_input], [chatbot, chat_input]
        )
        bot_msg = chat_msg.then(fn, chatbot, chatbot, api_name="bot_response")
        bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

        chatbot.like(print_like_dislike, None, None, like_user_message=True)

    pki_expert.launch()


if __name__ == "__main__":
    interactive_smart_assistant(fn=query_openrouter_llm)  # or fn=query_local_hgf_llm
