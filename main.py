from data_collection import prepare_csv_data
from embeddings import *
from completions import *

DOCUMENTS_PATH = "/Users/manibatra/code/section/docs/docs"
CSV_PATH = "./data/data-concise.csv"
EMBEDDINGS_CSV_PATH = "./data/embeddings-concise.csv"

if __name__ == '__main__':
    # Organise the data into a CSV file
    # prepare_csv_data(DOCUMENTS_PATH, CSV_PATH)

    # Compute the document embeddings
    df = read_csv(CSV_PATH, header=0)
    df = df.set_index(['title', 'description'])
    df = df.sort_index()

    # Test
    # print(f"{len(df)} rows in the data")
    # doc_index = ('Kubernetes and Section ', 'Kubernetes API and Section')
    # print(df.loc[doc_index].content.values)
    #
    # # Create and save the embeddings
    document_embeddings = compute_doc_embeddings(df)
    save_embeddings(document_embeddings, EMBEDDINGS_CSV_PATH)

    # # Load the document embeddings
    # document_embeddings = load_embeddings(EMBEDDINGS_CSV_PATH)
    #
    # # Test
    # # example_entry = list(document_embeddings.items())[0]
    # # print(f"{example_entry[0]} : {example_entry[1][:5]}... ({len(example_entry[1])} entries)")
    #
    # # Read the query from the user input
    # query = input("Enter your query: ")
    # potential_contexts = potential_contexts_by_query_similarity(query, df, document_embeddings)
    # # print('Sending to GPT:' + ''.join(potential_contexts[:5]))
    #
    # # Get the completion for a query
    # completion_generator = create_completion(query, potential_contexts)
    # for chunk in completion_generator:
    #     chunk_message = chunk['choices'][0]['delta']
    #     if 'content' in chunk_message:
    #         message = chunk_message['content']
    #         print(message, end='')
