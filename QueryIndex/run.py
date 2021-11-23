import subprocess
import re
from shutil import copyfile

method = [" bmw"," tp"]
topk = [" 10"," 50", " 100", " 200", " 500", " 1000"]
blocksize = [" 64"," 128"," 256", " 512"," 1024"]
score = [" BM25", " DeepImpact"]

command = "./query_binary_index /share/msmarco/msmarco_passage_v1.normlen /share/msmarco/bm25.data /share/msmarco/bm25.info /share/msmarco-passage/queries.dev.tsv"
command2 = "./query_binary_index /share/msmarco/msmarco_passage_v1.normlen /share/msmarco/deepimpact.data /share/msmarco/deepimpact.info /share/msmarco-passage/queries.dev.tsv"


for i in method:
    for b in blocksize:
        for p in topk:
            for s in score:
                f = open("/home/carl/SIGIR2021-ShortPaper-WindowNavigation/QueryIndex/results/" + i.strip() + "_"+ b.strip() + "_"  + p.strip() + "_" + s.strip() + ".txt","w+")
                if(score == " BM25"):
                    subprocess.run(command+i+b+p+s,shell=True,stdout=f)
                else:
                    subprocess.run(command2+i+b+p+s, shell=True,stdout=f)

