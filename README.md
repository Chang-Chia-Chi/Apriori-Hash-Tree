# Apriori-Hash-Tree

## Introduction
**Association rule** is a very useful tool to explore relationships between different items and itemsets.     
How to find these rules efficiently is critic when number of products and transaction database are large.     
In Data Mining, **apriori principle** is one of the method to quickly search **frequent itemsets** with different     
itemsets size. Besides, we also need a data structure-**hash tree** to reduce number of comparison between    
transactions and possible candidates.

## About code
This project implement **apriori algorithm** with **hash tree** used for large dataset. Data will be stored in **sparse matrix**   
The execution result and time are compared with python package `mlxtend`. Not only apriori, but also fpgrowth is used.    

## Result
![image](https://github.com/Chang-Chia-Chi/Apriori-Hash-Tree/blob/main/pic/Execution_time.jpg)    

Apriori with hash tree is slower and execution time increases rapidly when support reduced because   
naive for loop to scan the whole database and generate new candidates is used. It’s hard to comparable     
to Apriori_API, which utilizes vectorization, especially for large dataset.   

FP-growth uses a specific data structure that compresses frequent itemsets into a FP-tree (Frequent Pattern Tree).    
By doing so, FP-growth doesn’t need to scan database multiple times and large amount of candidates generation     
is avoided. This is the reason why execution time of FP-growth is almost constant when support becomes smaller and smaller.
