from data_collection import prepare_csv_data
from embeddings import *

DOCUMENTS_PATH = "/Users/manibatra/code/section/docs/docs"
CSV_PATH = "/Users/manibatra/data.csv"
EMBEDDINGS_CSV_PATH = "/Users/manibatra/embeddings.csv"

if __name__ == '__main__':
    # Organise the data into a CSV file
    # prepare_csv_data(DOCUMENTS_PATH, CSV_PATH)

    # Compute the document embeddings
    # df = read_csv(CSV_PATH, header=0)
    # df = df.set_index(['title', 'description'])
    # print(f"{len(df)} rows in the data")
    # document_embeddings = compute_doc_embeddings(df)
    # save_embeddings(document_embeddings, EMBEDDINGS_CSV_PATH)

    # Load the document embeddings
    document_embeddings = load_embeddings(EMBEDDINGS_CSV_PATH)
    print(order_document_sections_by_query_similarity("How to add a domain?", document_embeddings)[:5])

    # example_entry = list(document_embeddings.items())[0]
    # print(f"{example_entry[0]} : {example_entry[1][:5]}... ({len(example_entry[1])} entries)")
