import gradio as gr

from completions import create_completion
from embeddings import potential_contexts_by_query_similarity, load_embeddings, read_csv
from main import EMBEDDINGS_CSV_PATH, CSV_PATH


def question_answer(question):
    # Read the data
    df = read_csv(CSV_PATH, header=0)
    df = df.set_index(['title', 'description'])
    df = df.sort_index()

    # Load the document embeddings
    document_embeddings = load_embeddings(EMBEDDINGS_CSV_PATH)

    # Read the query from the user input
    potential_contexts = potential_contexts_by_query_similarity(question, df, document_embeddings)

    # Get the completion for a query
    answer = ''
    response = create_completion(question, potential_contexts)
    for chunk in response:
        chunk_message = chunk['choices'][0]['delta']
        if 'content' in chunk_message:
            message = chunk_message['content']
            answer += message
            yield answer


css = """
* {
    background-color: #202634;
}

.gradio-container {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 100%;
    font-family: 'Inter', sans-serif;
    padding: 0 !important;
}

.main {
    width: 100%;
}

#input_box {
    padding: 0;
    background-color: white;
    border-radius: 0.375rem;
    outline: none;
    transition: all 0.15s ease-in-out;
    font-size: 1rem;
    width: 100%;
}

#input_box:focus {
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.5);
}

#output_box {
    margin-top: 1rem;
    padding: 0.75rem;
    font-size: 1rem;
}

#output_box p {
    color: white !important;
}

#output_box ol {
    color: white !important;
}

#output_box code {
    color: black !important;
}

#output_box li {
    color: white !important;
}    

#output_box li strong {
    color: white !important;
}

#submit_button {
    background: #4BA570 !important;
    width: 10%;
    margin: auto;
    color: white;
    padding-left: 1.5rem;
    padding-right: 1.5rem;
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
    border-radius: 4px !important;
    transition: background-color 0.15s ease-in-out;
    font-size: 1rem;
    font-weight: 500;
    border: none;
}

#submit_button:hover {
    background: #32734E !important;
}

.prose h2 {
    text-align: center;
    color: white !important;
}

#title {
    margin-top: 50px;
}

"""

with gr.Blocks(css=css, elem_id='gradio_container') as iface:
    gr.Markdown("## SectionGPT", elem_id='title')
    input_text = gr.Textbox(elem_id='input_box', show_label=False)
    search_button = styled_button = gr.Button(value="SEARCH",
                                              elem_id='submit_button')
    output_markdown = gr.Markdown(elem_id='output_box')

    args = {
        'fn': question_answer,
        'inputs': input_text,
        'outputs': output_markdown
    }

    input_text.submit(**args)
    search_button.click(**args)

iface.queue()

try:
    iface.launch(server_name="0.0.0.0", server_port=80)
except KeyboardInterrupt:
    # Code to handle the keyboard interrupt
    gr.close_all()
    print("KeyboardInterrupt detected. Exiting...")
