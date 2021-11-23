import re
import matplotlib
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

# style.use('dark_background')

def read_avg(retrieval_method,score_method):
    block_size = [64, 128, 256, 512, 1024]
    top_k = [10, 50, 100, 200, 500, 1000]
    # block_size = [64]
    # top_k = [10]
    query_block = np.array([])
    query_k = np.array([])
    query_avg = np.array([])
    for block in block_size:
        for k in top_k:
            query_block = np.append(query_block, block)
            query_k = np.append(query_k, k)
            filename = "./results/"+ retrieval_method + "_" + str(block) + "_" + str(k) + "_"+score_method + ".txt"
            # file name : ./results/<retrieval_method>_<block_size>_<top_k>.txt
            f=open(filename,"r") 
            strings = f.read()
            pattern = re.compile(r'Time cost: [0-9]*')
            matches = pattern.finditer(strings)
            i = 0
            sum = 0
            outputfile = ""
            for x in matches:
                time = int(x.group().replace("Time cost: ", ""))
                print(time)
                sum += time
                # outputfile += x.group()
                i += 1
            print("average is:", sum / i)
            query_avg = np.append(query_avg, sum / i)
            f.close()
    return query_block, query_k, query_avg

# bmw_block, bmw_k, bmw_avg = read_avg("bmw","BM25")
# bmw_block_dp, bmw_k_dp, bmw_avg_dp = read_avg("bmw","DeepImpact")
tp_block, tp_k, tp_avg = read_avg("tp","BM25")
tp_block_dp, tp_k_dp, tp_avg_dp = read_avg("tp","DeepImpact")


def format_block(n):
    numbers = {
        64: 1,
        128: 2,
        256: 3,
        512: 4,
        1024: 5,
    }
    return numbers.get(n, None)

def format_k(n):
    numbers = {
        10: 1,
        50: 2,
        100: 3,
        200: 4,
        500: 5,
        1000: 6,
    }
    return numbers.get(n, None)


for i in np.arange(len(tp_avg)):
    print("bmw: block size: %d, top k: %d, average time: %d." %(tp_block[i], tp_k[i], tp_avg[i]))
    tp_block[i] = format_block(tp_block[i])
    tp_k[i] = format_k(tp_k[i])
    print("tp: block size: %d, top k: %d, average time: %d." %(tp_block_dp[i], tp_k_dp[i], tp_avg_dp[i]))
    tp_block_dp[i] = format_block(tp_block_dp[i])
    tp_k_dp[i] = format_k(tp_k_dp[i])

colors = ['blue', 'red']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xticks([0, 1, 2, 3, 4, 5])
ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
ax.set_xticklabels((0, 64, 128, 256, 512, 1024))
ax.set_yticklabels((0, 10, 50, 100, 200, 500, 1000))

ax.axes.set_xlim3d(left = 0, right = 5) 
ax.axes.set_ylim3d(bottom = 0, top = 6)  

scatter1 = ax.scatter(tp_block, tp_k, tp_avg, c = colors[0], marker = 'o')
scatter2 = ax.scatter(tp_block_dp, tp_k_dp, tp_avg_dp, c = colors[1], marker = 'v')

scatter1_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[0], marker = 'o')
scatter2_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[1], marker = 'v')
ax.legend([scatter1_proxy, scatter2_proxy], ['tp', 'tp with DP'], numpoints = 1)
ax.set_xlabel('block size')
ax.set_ylabel('topK')
ax.set_zlabel('avg time')

plt.savefig('graph.png')
