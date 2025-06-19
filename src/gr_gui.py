import gradio as gr
from rag import collection, llm_pipeline
from dialogue_mngr import query


# Gradio wrapper
def gradio_interface(user_question):
    return query(user_question, collection, llm_pipeline)


# Launch the Gradio UI
def launch_gradio_app():
    global query  # required if query refers to globals

    iface = gr.Interface(
        fn=gradio_interface,
        inputs=gr.Textbox(
            lines=2, placeholder="Ask a question...", label="Your Question"
        ),
        outputs=gr.Textbox(label="Assistant Response"),
        title="PKI Expert",
        description="Ask any question, and the assistant will answer using retrieved context from the database.",
        theme="default",
        allow_flagging="never",
    )
    iface.launch()
