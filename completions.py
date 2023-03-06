import numpy as np
import openai
import tiktoken

from embeddings import order_document_sections_by_query_similarity

MAX_TOKEN_LENGTH = 2048
MODEL = "gpt-3.5-turbo"
SYSTEM_SETUP_MESSAGE = 'You are a helpful assistant that works for Section and answers question based on the ' \
                       'provided context. Be concise, format the answer in lists/steps where possible, supply code ' \
                       'examples where ' \
                       'possible  and if you\'re unsure of the ' \
                       'answer, say "Sorry, I don\'t ' \
                       'know."'


def construct_messages(query: str, contexts: list[str]) -> list[dict[str, str]]:
    messages = []
    num_tokens = 0

    encoding = tiktoken.encoding_for_model(MODEL)

    # add the system setup message
    messages.append({"role": "system", "content": SYSTEM_SETUP_MESSAGE})
    num_tokens += len(encoding.encode(SYSTEM_SETUP_MESSAGE))
    num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>

    # count the query tokens but add it after the contexts
    formatted_query = '\nQ: ' + query + '\nA: '
    num_tokens += len(encoding.encode(formatted_query))

    # add the context but add it after finalising the context
    num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>
    num_tokens += len(encoding.encode('Context: '))

    # send at least 1 context
    final_context = contexts.pop(0)
    num_tokens += len(encoding.encode(final_context))

    # for shorter context send multiple up to MAX_TOKEN_LENGTH
    for context in contexts:
        potential_num_tokens = num_tokens + len(encoding.encode(context))
        if potential_num_tokens > MAX_TOKEN_LENGTH:
            break
        else:
            final_context += '\n' + context
            num_tokens = potential_num_tokens

    messages.append({"role": "user", "content": final_context + formatted_query})

    # print(messages)
    return messages


def create_completion(query: str, contexts: list[str]) -> str:
    """
    Create a completion given a query and a context.
    """

    messages = construct_messages(query, contexts)

    # create the completion
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=0
    )

    return response['choices'][0]['message']['content']
