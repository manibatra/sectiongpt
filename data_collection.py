import json
import os
import re

import pandas as pd
import tiktoken

"""
NOTES
 - ignore index.md files
 - the tutorials should be the entire context where possible
 - the reference can be subdivided into sections
 - the guides can be subdivided into sections
 - the explanations can be subdivided into sections
 - the about can be subdivided into sections
"""

# create a dictionary of constant key value pairs
KEYWORDS = {'guides': 'how to use',
            'explanations': 'what is',
            'about': 'tell me about',
            'tutorials': 'how to setup',
            'reference': 'how to integrate',
            'get-started': 'how to'
            }

IGNORE_LIST = ['terms-and-conditions.md', 'data-processing.md', 'index.md']


# function to recursively traverse a directory and find all md files
def find_md_files(path: str):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".md"):
                yield os.path.join(root, file)


def get_md_files(path: str):
    md_files = find_md_files(path)
    return md_files


def generate_jsonl_pass1(md_files: list):
    for file in md_files:
        # remove '/Users/manibatra/code/section/docs' from the path
        file_temp = file.replace('/Users/manibatra/code/section/docs/docs', '')

        # check if the base directory is in the dictionary
        print(file_temp.split('/')[1])
        if file_temp.split('/')[1] in KEYWORDS:
            prompt = KEYWORDS[file_temp.split('/')[1]]

            # read the file
            with open(file, 'r') as f:
                lines = f.readlines()
                # write the lines to a JSONL document in the following format
                # {"prompt": "<prompt>", "completion": "<lines>"}
                # {"prompt": "<prompt>", "completion": "<lines>"}
                # {"prompt": "<prompt>", "completion": "<lines>"}
                with open('/Users/manibatra/data.jsonl', 'a') as result:
                    topic = file_temp.split('/')[-1].split('.')[0]
                    lines = [line.strip() for line in lines]

                    entry = {
                        "prompt": prompt + ' ' + topic + '\n\n###\n\n',
                        "completion": '. '.join(lines).replace('---. title:', '') + '--####--'
                    }
                    json.dump(entry, result)
                    result.write('\n')
    for file in md_files:
        # remove '/Users/manibatra/code/section/docs' from the path
        file_temp = file.replace('/Users/manibatra/code/section/docs/docs', '')

        # check if the base directory is in the dictionary
        print(file_temp.split('/')[1])
        if file_temp.split('/')[1] in KEYWORDS:
            prompt = KEYWORDS[file_temp.split('/')[1]]

            # read the file
            with open(file, 'r') as f:
                lines = f.readlines()
                # write the lines to a JSONL document in the following format
                # {"prompt": "<prompt>", "completion": "<lines>"}
                # {"prompt": "<prompt>", "completion": "<lines>"}
                # {"prompt": "<prompt>", "completion": "<lines>"}
                with open('/Users/manibatra/data.jsonl', 'a') as result:
                    topic = file_temp.split('/')[-1].split('.')[0]
                    lines = [line.strip() for line in lines]

                    entry = {
                        "prompt": prompt + ' ' + topic + '\n\n###\n\n',
                        "completion": '. '.join(lines).replace('---. title:', '') + '--####--'
                    }
                    json.dump(entry, result)
                    result.write('\n')


def count_tokens(content: str):
    model = "gpt-3.5-turbo-0301"
    encoding = tiktoken.encoding_for_model(model)
    tokens = len(encoding.encode(content))
    return tokens


def extract_title_description(content: str) -> tuple:
    """
    extract the title and description from the content. The format is
    ---
    title: "CI/CD Pipeline Deployment"
    description: Learn how to create a pipeline that deploys your code to Section automatically
    ---
    """
    title_match = re.search(r'title: \"?([^\"]+)\"?\n[d|D]escription:', content)
    description_match = re.search(r'[d|D]escription: (.+)', content)
    return title_match.group(1), description_match.group(1)


def remove_title_description(content: str) -> str:
    """
    remove the title and description from the content. The format is
    ---
    title: "CI/CD Pipeline Deployment"
    description: Learn how to create a pipeline that deploys your code to Section automatically
    ---
    """
    content = re.sub(r'---\ntitle: \"?([^\"]+)\"?\n[d|D]escription: (.+)\n---\n', '',
                     content)
    content = re.sub(r'---\ntitle: \"?([^\"]+)\"?\n[d|D]escription: (.+)\nsidebar_position: (.+)\n---\n', '', content)
    return content


def read_content(file: str) -> str:
    with open(file, 'r') as f:
        lines = f.readlines()
        lines = ''.join(lines)
        return lines


def extract_heading_sections(title, description, content: str) -> list:
    sections = []
    lines = content.splitlines()
    section_content = []
    section_heading = None
    in_code_block = False
    has_heading = False
    all_section_headings = []

    for line in lines:
        if line.strip() == '```':
            in_code_block = not in_code_block

        if not in_code_block:
            if line.startswith('#'):
                has_heading = True
                if section_heading:
                    # add title and description as prefix to improve context matching
                    all_section_headings.append(section_heading)
                    section_content = section_content
                    section_text = '\n'.join(section_content).strip()
                    sections.append((section_heading, section_text))
                section_heading = line.strip('#').strip()
                section_content = []
            else:
                section_content.append(line)
        else:
            section_content.append(line)

    if section_heading:
        # add title and description as prefix to improve context matching
        # all_section_headings.append(section_heading)
        # section_content = [title, description] + section_content
        section_text = '\n'.join(section_content).strip()
        sections.append((section_heading, section_text))

    if not has_heading:
        section_heading = title + ' ' + description
        section_text = '\n'.join(lines).strip()
        sections.append((section_heading, section_text))

    for idx, (heading, section_text) in enumerate(sections):
        # section_content = [title, description] + all_section_headings + section_text.splitlines()
        section_content = section_text.splitlines()
        updated_section_text = '\n'.join(section_content).strip()
        sections[idx] = (heading, updated_section_text)

    return sections


def generate_elaborate_data(file: str) -> tuple:
    content = read_content(file)
    title, description = extract_title_description(content)
    content = remove_title_description(content)
    tokens = count_tokens(content)
    return title, description, content, tokens


def generate_concise_data(file: str) -> list:
    content = read_content(file)
    title, description = extract_title_description(content)
    content = remove_title_description(content)
    sections = extract_heading_sections(title, description, content)

    data = []
    for section_heading, section_content in sections:
        tokens = count_tokens(section_content)
        if section_content and section_content.strip() != '':
            data.append((title, section_heading, section_content, tokens))

    return data


def create_dataframes(res: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(res, columns=['title', 'description', 'content', 'tokens'])


def create_csv(df: pd.DataFrame, file: str):
    if os.path.exists(file):
        os.remove(file)
    df.to_csv(file, index=False)


def prepare_csv_data(file: str, csv_file: str):
    md_files = get_md_files(file)
    res = []
    for file in md_files:
        if file.split('/')[-1] not in IGNORE_LIST:
            res.extend(generate_concise_data(file))
            res.append(generate_elaborate_data(file))

    print(len(res))
    df = create_dataframes(res)
    pd.set_option('display.max_columns', None)
    create_csv(df, csv_file)
