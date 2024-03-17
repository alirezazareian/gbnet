from pandas import merge as pd_merge

def compare(df_1, df_2):
    return pd_merge(df_1, df_2, on=list(['e1', 'e2', 'r']), how='outer', indicator='Exist')

def modify_edges(df, conditions, column_index, to):
    # if column_index == 1: column_index = 'e1'
    # if column_index == 2: column_index = 'e2'
    # if column_index == 3: column_index = 'r'
    e1, e2, r = conditions
    df.loc[(df['e1'] == e1) & (df['e2'] == e2) & (df['r'] == r), column_index] = to


def delete_edge(df, conditions):
    e1, e2, r = conditions
    print(df.loc[(df['e1'] == e1) & (df['e2'] == e2) & (df['r'] == r)])
    df.drop(df.index[(df['e1'] == e1) & (df['e2'] == e2) & (df['r'] == r)], axis=0, inplace=True)


def switch(listy):
    listy[0], listy[1] = listy[1], listy[0]
    return listy
