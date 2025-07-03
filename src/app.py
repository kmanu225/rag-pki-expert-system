import gradio as gr
from rag import context_retrieval, generate_response


def query_local_hg_face(user_question: str):
    """
    Processes a user question by retrieving relevant context and generating a response.

    This function first retrieves context related to the user's question using the
    `context_retrieval` function, then generates a response using the `generate_response`
    function. Both context retrieval and response generation are handled locally.

    Args:
        user_question (str): The user's input question.

    Returns:
        str: The generated response from the assistant.
    """
    context = context_retrieval(user_question)
    return generate_response(user_question, context)


# Gradio wrapper
def gradio_interface(user_question):
    return query_local_hg_face(user_question)


def launch_gradio_app_history():
    gr.ChatInterface(
        fn=query_local_hg_face,
        type="messages",
        chatbot=gr.Chatbot(type="messages"),
        title="PKI Expert",
        description="Ask any question, and the assistant will answer using retrieved context from the database.",
        theme="default",
        save_history=True,
    ).launch()



if __name__ == "__main__":
    launch_gradio_app_history()