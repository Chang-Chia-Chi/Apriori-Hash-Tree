import time
import numpy as np
import pandas as pd
from copy import deepcopy
from itertools import combinations, product
from collections import defaultdict, Counter
from mlxtend.preprocessing import TransactionEncoder

class TreeNode:
    """
    Class to build tree node instance
    """
    def __init__(self):
        self.children = defaultdict(TreeNode)
        self.bucket = defaultdict(lambda : 0)
        self.isleaf = True
        self.idx = 0 # idx to decide which element to be used for hashing

class HashTree:
    """
    Class to build and store tree for apriori algorithm

    params:
    max_leaf_size --> maximum number of itemsets in a childe node
    hash_num --> number for mod computation
    """
    def __init__(self, max_leaf_size=3, hash_num=3):
        self.root = TreeNode()
        self.root.isleaf = False
        self.added = set()

        self.max_leaf_size = max_leaf_size
        self.hash_num = hash_num
    
    def insert(self, node, itemset, cnt):
        """
        Insert new itemset recursively

        params:
        node --> TreeNode instance
        itemset --> itemset to be inserted
        cnt --> support count for itemset
        """
        if not node.isleaf:
            key = self.hash(itemset[node.idx])
            self.insert(node.children[key], itemset, cnt)
        else:
            node.bucket[itemset] += cnt
            if len(node.bucket) > self.max_leaf_size:
                node.idx += 1
                for old_itemset, old_cnt in node.bucket.items():
                    key = self.hash(old_itemset[node.idx])
                    node.children[key].idx = min(node.idx, len(old_itemset)-1)
                    self.insert(node.children[key], old_itemset, old_cnt)
                node.bucket = defaultdict(lambda : 0)
                node.isleaf = False

    def build_tree(self, itemsets):
        """
        Build HashTree

        params:
        itemsets --> itemsets used to initialze the tree
        """
        for itemset in itemsets:
            self.insert(self.root, itemset, 0)
    
    def freq_itemsets(self, node, support, result_list, count_list):
        """
        Function to get frequent itemsets by depth-first-search

        params:
        node --> TreeNode instance
        support --> threshold of support count
        result_list --> list to store frequent itemsets
        """
        if node.isleaf:
            for itemset, cnt in node.bucket.items():
                if cnt >= support:
                    result_list.append(itemset)
                    count_list.append(cnt)
            return
        else:
            for child in node.children.values():
                self.freq_itemsets(child, support, result_list, count_list)
        
        return result_list, count_list

    def add_count(self, node, pick, rest, idx, k):
        """
        Function to add support count of itemset

        params:
        node --> TreeNode
        pick --> items have been picked
        rest --> items haven't been picked
        idx --> index of which transaction is being counted
        k --> length of itemset
        """
        if node.isleaf:
            superset = pick+rest
            for itemset in node.bucket:
                if itemset in self.added:
                    continue
                # if last element is different or smallest element of itemset is smaller than first element in tmp_superset
                # itemset must not be subset of the superset
                tmp_superset = [item for item in superset if item >= itemset[0] and item <= itemset[-1]]
                if len(tmp_superset)==0:
                    continue
                if itemset[-1] != tmp_superset[-1] or itemset[0] != tmp_superset[0]:
                    continue
                if all([item in tmp_superset for item in itemset]):
                    node.bucket[itemset] += 1
                    self.added.add(itemset)
        else:
            n_pick = len(pick)
            n_rest = len(rest)
            n_rest_min = k - (n_pick+1)
            if n_rest_min < 0:
                return
            n_iter = n_rest - n_rest_min
            # print(n_iter, n_rest_min, pick, rest)
            # print([child.isleaf for child in node.children.values()])
            # if not all([child.isleaf for child in node.children.values()]):
            for i in range(n_iter):
                curr_pick = pick + [rest[i]]
                curr_rest = rest[i+1:]
                key = self.hash(curr_pick[node.idx])
                if key in node.children:
                    self.add_count(node.children[key], curr_pick, curr_rest, idx, k)
            # else:
            #     all_item = pick + rest
            #     key = self.hash(all_item[node.idx])
            #     for child in node.children:
            #         self.add_count(node.children[key], pick, rest, count, idx, k)

    def hash(self, num):
        """
        Simple hash function using mod computation

        params:
        num --> number of mod base
        """
        return num % self.hash_num

def ismerge(itemset_1, itemset_2):
    if itemset_1[:-1] == itemset_2[:-1]:
        return True
    else:
        return False

def get_new_candidates(all_frequent, k):
    if k == 1:
        new_candidates = list(map(lambda x: list(x), combinations(all_frequent[1][0], k+1)))
    elif k == 2:
        F_1 = all_frequent[1][0]
        new_candidates = [fk1_itemset+[f1_item] for fk1_itemset in all_frequent[2][0] for f1_item in F_1 if f1_item not in fk1_itemset]
    else:
        Fk_1 = all_frequent[k][0]
        Fk_1_compare = set([tuple(itemset) for itemset in Fk_1]) # create for hash comparing
        new_candidates = []
        for i in range(len(Fk_1)):
            for j in range(i+1, len(Fk_1)):
                if ismerge(Fk_1[i], Fk_1[j]):
                    new = Fk_1[i][:-1] + [Fk_1[i][-1]] + [Fk_1[j][-1]]

                    # candidate prunning
                    add = True
                    for comb in list(combinations(new, k)):
                        if comb not in Fk_1_compare:
                            add = False
                            break
                    if add:
                        new_candidates.append(new)
        
    return new_candidates

def dict2df(all_frequent, df_rows, num2id_mapping):
    """
    Convert frequent dictionary back to dataframe as same format as mlxend API

    params:
    all_frequent --> frequent dictionary
    df_size --> number of rows of dataframe
    num2id_mapping --> encode to productId mapping dictionary
    """
    itemsets, supports = list(zip(*[(itemset, support/df_rows) for itemsets, supports in all_frequent.values() 
                              for itemset, support in zip(itemsets, supports)]))
    convert_itemsets = []
    for itemset in itemsets:
        if isinstance(itemset, np.int32):
            new_itemset = [num2id_mapping[itemset]]
        else:
            new_itemset = [num2id_mapping[item] for item in itemset]
        convert_itemsets.append(tuple(new_itemset))

    tmp_dict = {
        "support":supports,
        "itemsets":convert_itemsets
    }
    df = pd.DataFrame.from_dict(tmp_dict)
    return df

def num2Id(datasets, te_array):
    """
    Create mapping dictionary with key: number, value: productId

    params:
    datasets --> datasets to be converted
    te_array --> array after encoding
    """
    p_start = 0
    mapping = {}
    visited = set()
    for idx, p_end in enumerate(te_array.indptr[1:]):
        dataset = datasets[idx]
        encode = te_array.indices[p_start:p_end]
        for data, code in zip(dataset, encode):
            if data not in visited:
                mapping[code] = data
        p_start = p_end
    return mapping

def apriori_student(df, dataset, te_array, min_support=0.001):
    """
    Apriori algorithm with hash tree and candidates prunning

    params:
    df --> sparse dataframe
    te_array --> encoded array for mapping conversion
    min_support --> minimum support
    """
    T = df.sparse.to_coo().tocsr()
    support = int(df.shape[0]*min_support)

    # convert all transactions to a list
    p_start = 0
    transactions = []
    for p_end in T.indptr[1:]:
        trans = T.indices[p_start:p_end]
        transactions.append(list(trans))
        p_start = p_end
    # sorter by length for intermediate break when looping to add support
    transactions = sorted(transactions, key=lambda x:len(x), reverse=True)

    # initializing frequent 1-itemsets
    all_record = T.indices
    p_idx, p_cnt = np.unique(all_record, return_counts=True)
    mask = np.where(p_cnt >= support)
    freq_itemsets = list(p_idx[mask])
    count_itemsets = list(p_cnt[mask])
    
    # repeat until Fk is empty
    k = 1
    all_frequent = {}
    all_frequent[k] = (freq_itemsets.copy(), count_itemsets.copy())
    while len(freq_itemsets) > 0:
        candidates = [tuple(candidate) for candidate in get_new_candidates(all_frequent, k)]
        n_can = len(candidates)

        k += 1
        htree = HashTree(max_leaf_size=n_can, hash_num=n_can)
        htree.build_tree(candidates)
        for idx, tran in enumerate(transactions):
            if len(tran) < k:
                break
            htree.add_count(htree.root, [], tran, idx, k)
            htree.added = set()

        freq_itemsets, count_itemsets = htree.freq_itemsets(htree.root, support, [], [])
        # sort to ensure correct order of combinations results
        # if freq_itemsets is not empty, add new frequent sets to dict
        if freq_itemsets and count_itemsets:
            sort_itemsets = sorted(zip(freq_itemsets, count_itemsets), key=lambda x:x[0])
            freq_itemsets, count_itemsets = (list(x) for x in zip(*sort_itemsets))
            freq_itemsets = [list(itemset) for itemset in freq_itemsets]
            all_frequent[k] = (freq_itemsets, count_itemsets)

    mapping = num2Id(dataset, te_array)
    freq_itemsets = dict2df(all_frequent, df.shape[0], mapping)
    return freq_itemsets

if __name__=='__main__':
    # ======== Code for Test =========
    candidate = [(1,4,5),(1,2,4),(4,5,7),(1,2,5),(4,5,8),(1,5,9),(1,3,6),(2,3,4),(5,6,7),(3,4,5),(3,5,6),(3,5,7),(6,8,9),(3,6,7),(3,6,8)]
    htree = HashTree(3, 3)
    htree.build_tree(candidate)
    print(htree.freq_itemsets(htree.root, 0, [], []))

    def read_data(file_path):
        with open(file_path, 'r') as file:
            dataset = []
            lines = file.readlines()
            for line in lines[:10000]:
                user, *items = line.strip().split(',')
                dataset.append(items)
        return dataset

    file_path = "music.txt"
    dataset = read_data(file_path)
    te = TransactionEncoder()
    te_array = te.fit_transform(dataset, sparse=True)
    sparse_df = pd.DataFrame.sparse.from_spmatrix(te_array, columns=te.columns_)
    t1 = time.time()
    frequent_itemsets = apriori_student(sparse_df, dataset, te_array)
    print(time.time()-t1)
    print(frequent_itemsets)