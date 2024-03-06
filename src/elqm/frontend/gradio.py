import textwrap

import gradio as gr

from elqm import ELQMPipeline


def launch_gradio_frontend(elqm_pipeline: ELQMPipeline, share: bool = False) -> None:
    """
    Launch a gradio frontend for the chatbot.

    Parameters
    ----------
    elqm_pipeline :
        The ELQM pipeline to use for answering questions.
    share : bool, optional
        Whether to share the frontend, by default False
    """
    CSS = textwrap.dedent("""
    #component-1 { height:  90vh !important; }
    """)

    clear_btn = gr.Button(value="Clear")

    with gr.Blocks(css=CSS) as frontend:
        gr.ChatInterface(
            fn=elqm_pipeline.answer,
            clear_btn=clear_btn,
            title="ELQM: Energy Law Query Master")

        clear_btn.click(fn=elqm_pipeline.clear_chat_history)

    frontend.launch(share=share)
