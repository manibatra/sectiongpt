from data_collection import *

if __name__ == '__main__':
    md_files = get_md_files("/Users/manibatra/code/section/docs/docs")
    res = []
    for file in md_files:
        if file not in IGNORE_LIST:
            res.append(generate_data(file))

    df = create_dataframes(res)
    pd.set_option('display.max_columns', None)
    create_csv(df, '/Users/manibatra/data.csv')
