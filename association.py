import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from apriori_hash_tree import apriori_student
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

def time_wrap(func):
    def wrap(*args, **kwargs):
        t_start = time.time()
        result = func(*args, **kwargs)
        t_end = time.time()
        time_diff = t_end-t_start
        print("Fuction {} execution time is {:.2f} s".format(func.__name__, time_diff))
        return result, int(time_diff/60)
    return wrap

def read_data(file_path):
    with open(file_path, 'r') as file:
        dataset = []
        lines = file.readlines()
        for line in lines[:-1]:
            user, *items = line.strip().split(',')
            dataset.append(items)
    return dataset

def create_setstr(itemsets):
    text = ''
    for itemset in itemsets:
        text += '{'
        for item in list(itemset)[:-1]:
            text += item+', '
        text += list(itemset)[-1]+'}\n'
    return text

def post_process(df, n_items=3, confi_thres=0.5, top=10):
    """
    post process frequent itemsets dataframe to compute number of itemsets containing
    at least "n_items" items and top N longest itemsets

    params:
    n_items: number of items an itemset must contain
    confi_thres: threshold for rule generation
    top: top N longest itemsets you want to return
    """
    df["length"] = df['itemsets'].apply(lambda x:len(x))
    n_freq_itemsets = df.loc[df["length"]>=n_items].shape[0]

    sort_df = df.copy()
    sort_df = df.sort_values(by=["length", "itemsets"], ascending=[False, True])
    top_long = sort_df.head(top)
    top_long_txt = create_setstr(top_long["itemsets"])

    rules = association_rules(freq_itemsets, min_threshold=0.5)
    top_conf = rules.sort_values(by=["confidence", "antecedents"], ascending=[False, True]).head(10)
    top_conf_txt = ''
    ant_texts = []
    con_texts = []
    confs = []
    for rule in top_conf[["antecedents", "consequents", "confidence"]].values:
        ant_txt = '{'
        for ant in list(rule[0])[:-1]:
            ant_txt += ant+', '
        ant_txt += list(rule[0])[-1]+'}'
        
        con_txt = '{'
        for con in list(rule[1])[:-1]:
            con_txt += con+', '
        con_txt += list(rule[1])[-1]+'}'

        ant_texts.append(ant_txt)
        con_texts.append(con_txt)
        confs.append(rule[2])
    ant_max_len = max([len(text) for text in ant_texts])
    con_max_len = max([len(text) for text in con_texts])

    top_conf_txt = "{:>{x}}      {:>{y}}   {:>10}\n".format("antecedents", "consequents", "confidence", x=ant_max_len+4, y=con_max_len+4)
    str_format = "{:>{x}}  =>  {:>{y}}   {:>10}\n"
    for ant, con, conf in zip(ant_texts, con_texts, confs):
        top_conf_txt += str_format.format(ant, con, conf, x=ant_max_len+4, y=con_max_len+4)
    return n_freq_itemsets, top_long_txt, top_conf_txt

@time_wrap
def apriori_t(df, min_support=0.0009, use_colnames=True, low_memory=True):
    freq_itemsets = apriori(df, min_support=min_support, use_colnames=use_colnames, low_memory=low_memory)
    return freq_itemsets

@time_wrap
def fpgrowth_t(df, min_support=0.0009, use_colnames=True):
    freq_itemsets = fpgrowth(df, min_support=min_support, use_colnames=use_colnames)
    return freq_itemsets

@time_wrap
def apriori_stu_t(df, dataset, te_array, min_support=0.0009):
    freq_itemsets = apriori_student(df, dataset, te_array, min_support=min_support)
    return freq_itemsets

def plot_time(t1, t2, t3, min_sups, save_path):
    fig = plt.figure(figsize=(12, 8))
    plt.plot([0,1,2], t1, markersize=12, marker='o', color='b', label="Apriori_API")
    plt.plot([0,1,2], t2, markersize=12, marker='v', color='g', label="Fpgrowth_API")
    plt.plot([0,1,2], t3, markersize=12, marker='s', color='r', label="Apriori_Student")
    
    plt.xlabel("Support")
    plt.ylabel("Execution Time (min)")
    plt.xticks([0, 1, 2], ["sup={}".format(s) for s in min_sups])
    plt.legend(loc="upper left")
    plt.savefig(save_path)

if __name__=='__main__':
    file_path = "music.txt"
    save_path = "Q3.jpg"
    dataset = read_data(file_path)
    te = TransactionEncoder()
    te_array = te.fit_transform(dataset, sparse=True)
    sparse_df = pd.DataFrame.sparse.from_spmatrix(te_array, columns=te.columns_)

    # Q1
    print("========== Q1 ==========")
    freq_itemsets, _ = apriori_t(sparse_df, min_support=0.0009, use_colnames=True, low_memory=True)
    amount, top_long, top_conf = post_process(freq_itemsets, n_items=3, top=10)
    print("Amount of frequent itemsets having at least 3 items: {}".format(amount))
    print("Top 10 longest frqeuent itemsets:")
    print(top_long)
    # Q2
    print("========== Q2 ==========")
    print("Top 10 greatest confidence")
    print(top_conf)
    # Q3
    print("========== Q3 ==========")
    t_apriori_api = []
    t_fpgrowth_api = []
    t_apriori_stu = []
    min_sups = [0.0009, 0.0006, 0.0003]
    for min_sup in min_sups:
        print("Start support counting: {}".format(min_sup))
        _, t_apr = apriori_t(sparse_df, min_support=min_sup, use_colnames=True, low_memory=True)
        _, t_fp = fpgrowth_t(sparse_df, min_support=min_sup, use_colnames=True)
        _, t_apr_stu = apriori_stu_t(sparse_df, dataset, te_array, min_support=min_sup)
        t_apriori_api.append(t_apr)
        t_fpgrowth_api.append(t_fp)
        t_apriori_stu.append(t_apr_stu)
    print("Complete support counting")
    plot_time(t_apriori_api, t_fpgrowth_api, t_apriori_stu, min_sups, save_path)