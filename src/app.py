import gradio as gr
from rag import context_retrieval, hgf_generator, openrouter_generator


def query_local_hgf_llm(user_question: str, history):
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
    # print(f"Retrieving context for question: {user_question}")

    context = context_retrieval(user_question)
    
    return hgf_generator(user_question, context)


def query_openrouter_llm(user_question: str, history):
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
    print(f"Retrieving context for question: {user_question}")

    context = context_retrieval(user_question)
    
    return openrouter_generator(user_question, context)


def launch_hgf_gradio_app_history():
    gr.ChatInterface(
        fn=query_local_hgf_llm,
        type="messages",
        chatbot=gr.Chatbot(type="messages"),
        title="PKI Expert",
        description="Ask any question, and the assistant will answer using retrieved context from the database.",
        theme="default",
        save_history=True,
    ).launch()

def launch_openrouter_gradio_app_history():
    gr.ChatInterface(
        fn=query_openrouter_llm,
        type="messages",
        chatbot=gr.Chatbot(type="messages"),
        title="PKI Expert",
        description="Ask any question, and the assistant will answer using retrieved context from the database.",
        theme="default",
        save_history=True,
    ).launch()

if __name__ == "__main__":
    # Launch the Gradio app for local Hugging Face LLM
    # launch_hgf_gradio_app_history()
    
    # Launch the Gradio app for openrouter LLM
    launch_openrouter_gradio_app_history()