import gradio as gr

from completions import create_completion
from embeddings import potential_contexts_by_query_similarity, load_embeddings, read_csv
from main import EMBEDDINGS_CSV_PATH, CSV_PATH


def question_answer(question):
    # Compute the document embeddings
    df = read_csv(CSV_PATH, header=0)
    df = df.set_index(['title', 'description'])
    # print(f"{len(df)} rows in the data")
    # document_embeddings = compute_doc_embeddings(df)
    # save_embeddings(document_embeddings, EMBEDDINGS_CSV_PATH)

    # Load the document embeddings
    document_embeddings = load_embeddings(EMBEDDINGS_CSV_PATH)

    # example_entry = list(document_embeddings.items())[0]
    # print(f"{example_entry[0]} : {example_entry[1][:5]}... ({len(example_entry[1])} entries)")

    # Read the query from the user input
    potential_contexts = potential_contexts_by_query_similarity(question, df, document_embeddings)

    # Get the completion for a query
    return create_completion(question, potential_contexts)


gr.Interface(fn=question_answer, inputs=["text"], outputs=["markdown"], title="SectionGPT",
             description="Instant answers to Section docs").launch()
