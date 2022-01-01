import re
import os

def transform(filename):  
    f = open(filename,"r+")
    strings = f.read()
    pattern = re.compile(r'Query(:[a-z]+,[0-9])+')
    matches = pattern.finditer(strings)
    for x in matches:
        result = (x.group().replace("Query",""))
        n_result = re.sub(',[0-9]+:'," ",result)
        n_result = n_result[1:-2]
        print(n_result)
        # newf = open('/share/msmarco-passage/queries.dev.tsv')
        # newstring = newf.read()
        # for words in n_result:
        #     new_matches = re.findall(r'\b'+words+r'\b',newstring)
        #     print(new_matches)
    f.close()


for filename in os.listdir("/home/carl/SIGIR2021-ShortPaper-WindowNavigation/QueryIndex/results"):
    transform(filename)