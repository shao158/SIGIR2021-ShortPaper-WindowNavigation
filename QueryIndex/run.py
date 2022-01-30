import subprocess
from shutil import copyfile\

method = [" bmw"] #" tp",
topk = [" 1000"] #" 10", " 100",
blocksize = [" 512"," 1024"] #1024 #variable_bmw_size " 128"," 256"
score = [" BM25"] #" BM25"
# is_random = [" R", " NR"]
# is_variable = [" V"," NV"]
random_bmw_size = " /share/msmarco-passage/variable-bmw/deepimpact.sizes"
random_bmw_size2 = " /share/msmarco-passage/variable-bmw/regular/deeplayered.sizes"
random_bmw_size3 = " /share/msmarco-passage/variable-bmw/bm25.sizes"

command = "./query_binary_index /share/msmarco/msmarco_passage_v1.normlen /share/msmarco-passage/bm25_count.data /share/msmarco-passage/bm25_count.info /share/msmarco-passage/queries.dev.tsv"
command2 = "./query_binary_index /share/msmarco/msmarco_passage_v1.normlen /share/msmarco-passage/deepimpact.data /share/msmarco-passage/deepimpact.info /share/msmarco-passage/queries.dev.tsv"
command3 = "./query_binary_index /share/msmarco/msmarco_passage_v1.normlen /share/msmarco-passage/variable-bmw/regular/deeplayered.data /share/msmarco-passage/variable-bmw/regular/deeplayered.info /home/carl/SIGIR2021-ShortPaper-WindowNavigation/QueryIndex/queries_layered.dev.tsv"
command4 = "./query_binary_index /share/msmarco/msmarco_passage_v1.normlen /share/msmarco-passage/variable-bmw/bm25L/bm25L.data  /share/msmarco-passage/variable-bmw/bm25L/info /home/carl/SIGIR2021-ShortPaper-WindowNavigation/QueryIndex/bm25L.queries.dev.tsv"



for i in method:
    for p in topk:
        for s in score:
            for b in blocksize:
                # for x in is_variable:
                f = open("/home/carl/SIGIR2021-ShortPaper-WindowNavigation/QueryIndex/variable_bmw_result/" + i.strip() + "_"+ b.strip() + "_"  + p.strip() + "_" + s.strip() + ".txt","w+")
                if(score == " BM25"):
                    # if(x == " NV"):
                    subprocess.run(command+i+b+p+s+" invalidate_address",shell=True,stdout=f)
                    # else:
                    #     subprocess.run(command+i+b+p+s+random_bmw_size,shell=True,stdout=f)
                # else:
                    # if(x == " NV"):
                    # subprocess.run(command2+i+b+p+s+" invalidiate_address", shell=True,stdout=f)
                    # else:
                    # subprocess.run(command3+i+b+p+s+" invalidate_address", shell=True,stdout=f)
            # f_1 = open("/home/carl/SIGIR2021-ShortPaper-WindowNavigation/QueryIndex/variable_bmw_result/" + i.strip() + "_"+ "0" + "_"  + p.strip() + "_" + s.strip() + ".txt","w+")
            # subprocess.run(command+i+" 0"+p+s+random_bmw_size3, shell=True,stdout=f_1)

