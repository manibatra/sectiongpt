from data_collection import prepare_csv_data
from embeddings import *
from completions import *

DOCUMENTS_PATH = "/Users/manibatra/code/section/docs/docs"
CSV_PATH = "./data/data.csv"
EMBEDDINGS_CSV_PATH = "./data/embeddings.csv"

if __name__ == '__main__':
    # Organise the data into a CSV file
    # prepare_csv_data(DOCUMENTS_PATH, CSV_PATH)

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
    query = input("Enter your query: ")
    potential_contexts = potential_contexts_by_query_similarity(query, df, document_embeddings)

    # Get the completion for a query
    print(create_completion(query, potential_contexts))
