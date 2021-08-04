import sys
from pyspark import SparkContext
from operator import add
from collections import defaultdict
import math
import time


support=sys.argv[1]
input_file=sys.argv[2]
output_file=sys.argv[3]

support = int(support)


start_time =time.time()
sc = SparkContext(appName="task2")

#preprocess
lines = sc.textFile(input_file).filter(lambda x: x != ('user_id, business_id'))\
            .distinct().map(lambda x: x.split(','))
#if case_num == 1:
#    baskets_ = lines.map(lambda x: (str(x[0]),str(x[1])))\
 #               .distinct().groupByKey().mapValues(set).map(lambda x: x[1])
#if case_num == 2:
baskets_ = lines.map(lambda x: (str(x[0]),str(x[1])))\
                .distinct().groupByKey().mapValues(set).map(lambda x: x[1])

baskets_count = baskets_.count()

#check frequent
def check_frequent(unknown_candidate, min_support, baskets):
    count = defaultdict(int)
    for candidate in unknown_candidate:
        for basket in baskets:
            if set(candidate).issubset(basket):
                count[tuple(sorted(candidate))] += 1
    #count(candidate, count)
    frequent_count = list(filter(lambda x: x[1] >= min_support, list(count.items())))
    frequent_pair = [i[0] for i in frequent_count]
    return frequent_pair

#get candidate
def get_candidate(tmp_candidate, N):
    candidate = []
    for i in range(len(tmp_candidate)-1):
        for j in range(i+1, len(tmp_candidate)):
            if len(tmp_candidate[i] | tmp_candidate[j]) == N:
                candidate.append(tuple(sorted(tmp_candidate[i] | tmp_candidate[j])))
    return candidate


#A_priori
def a_priori(iterator):
    baskets_a = [set(i) for i in list(iterator)]
    baskets_a_cnt = len(baskets_a)
    min_support = baskets_a_cnt * support / baskets_count

    x = baskets_a[0]
    for basket in baskets_a[1:]:
        x = x|basket
    
    each_basket = list(set(x))
    Apripri_candidate = []
    unknown_candidate = [{i} for i in each_basket]
    #check the frequent
    frequent_pair = check_frequent(unknown_candidate, min_support, baskets_a)
    Apripri_candidate.extend(frequent_pair)

    tmp_candidate = [{i[0]} for i in frequent_pair]

    #get candidate
    for N in range(1, len(each_basket)+1):
        if len(tmp_candidate) != 0 :
            unknown_candidate_ = get_candidate(tmp_candidate, N+1)
            unknown_candidate_ = [set(i) for i in set(unknown_candidate_)]
            frequent_pair_ = check_frequent(unknown_candidate_, min_support, baskets_a)
            tmp_candidate = [set(i) for i in frequent_pair_]
            Apripri_candidate.extend(tmp_candidate)
        else:
            break
    Apripri_candidate = [tuple(sorted(list(i))) for i in Apripri_candidate]
    return Apripri_candidate

#baskets_count = baskets_.count()

#SON
#Phase 1
map_1 = baskets_.mapPartitions(a_priori).map(lambda x: (x,1))
reduce_1 = map_1.reduceByKey(lambda x,y: (1)).keys()
candidate_all = sorted(reduce_1.collect(), key = lambda x: (len(x),x))

#Phase 2
def check_count(total_basket):
    baskets = [set(i) for i in list(total_basket)]
    count_dict = defaultdict(int)
    for each in candidate_all:
        for basket in baskets:
            if set(each).issubset(basket):
                count_dict[tuple(sorted(each))] += 1
    return list(count_dict.items())

#check the count
map_2 = baskets_.mapPartitions(check_count)
reduce_2 = map_2.reduceByKey(add).filter(lambda x:x[1] >= support).keys()
frequent_all = sorted(sorted(reduce_2.collect()), key = lambda x: (len(x),x))


#output file
with open(output_file,'w') as f:
    #output 1
    output_1 = 'Candidates:\n'
    tmp = 1
    for x in candidate_all:
        if len(x) == tmp:
            output_1 += str(x).replace(",)",")") + ','
        else:
            output_1 = output_1[:-1] + '\n\n'
            tmp = len(x)
            output_1 += str(x) + ','        
    output_1 = output_1[:-1]
    f.write(output_1 + '\n')
    
    #output 2
    output_2 = '\n Frequent Itemsets:\n'
    tmp_2 = 1
    for x in frequent_all:
        if len(x) == tmp_2:
            output_2 += str(x).replace(",)",")") + ','
        else:
            output_2 = output_2[:-1] + '\n\n'
            tmp_2 = len(x)
            output_2 += str(x)+','
    output_2 = output_2[:-1]
    f.write(output_2)


print("Duration: %s seconds." % (time.time() - start_time))




