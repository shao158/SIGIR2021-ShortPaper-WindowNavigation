import re
import matplotlib
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

# style.use('dark_background')

def read_avg(retrieval_method,score_method):
    block_size = [0, 512, 1024]
    top_k = [1000]
    is_random = ["R","NR"]
    # block_size = [64]
    # top_k = [10]
    query_block = np.array([])
    query_k = np.array([])
    query_avg = np.array([])
    for block in block_size:
        for k in top_k:
            query_block = np.append(query_block, block)
            query_k = np.append(query_k, k)
            filename = "./variable_bmw_result/"+ retrieval_method + "_" + str(block) + "_" + str(k) + "_"+score_method + ".txt"
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
                # print(time)
                sum += time
                # outputfile += x.group()
                i += 1
            print("average is:", sum / i)
            query_avg = np.append(query_avg, sum / i)
            f.close()
    return query_block, query_k, query_avg

# bmw_block, bmw_k, bmw_avg = read_avg("bmw","BM25","NR")
bmw_block_dp, bmw_k_dp, bmw_avg_dp = read_avg("bmw","DeepImpact")
bmw_block_dl, bmw_k_dl, bmw_avg_dl = read_avg("bmw","DeepLayered")
# bmw_block_r, bmw_k_r,bmw_avg_r = read_avg("bmw","BM25","R")
# bmw_block_dp_v, bmw_k_dp_v,bmw_avg_dp_v = read_avg("bmw","DeepImpact")
# tp_block, tp_k, tp_avg = ("tp","BM25")
# tp_block_dp, tp_k_dp, tp_avg_dp = read_avg("tp","DeepImpact")


def format_block(n):
    numbers = {
        # 64: 1,
        0: 1,
        512: 2,
        1024: 3,
        # 1024: 5,
    }
    return numbers.get(n, None)

def format_k(n):
    numbers = {
        # 10: 1,
        # 50: 2,
        # 100: 2,
        # 200: 4,
        # 500: 5,
        1000: 1,
    }
    return numbers.get(n, None)


for i in np.arange(len(bmw_avg_dp)):
    # print("bmw: block size: %d, top k: %d, average time: %d." %(bmw_block[i], bmw_k[i], bmw_avg[i]))
    # bmw_block[i] = format_block(bmw_block[i])
    # bmw_k[i] = format_k(bmw_k[i])
    print("bmw_dp: block size: %d, top k: %d, average time: %f." %(bmw_block_dp[i], bmw_k_dp[i], bmw_avg_dp[i]))
    bmw_block_dp[i] = format_block(bmw_block_dp[i])
    bmw_k_dp[i] = format_k(bmw_k_dp[i])
    print("bmw_dl: block size: %d, top k: %d, average time: %f." %(bmw_block_dl[i], bmw_k_dl[i], bmw_avg_dl[i]))
    bmw_block_dl[i] = format_block(bmw_block_dl[i])
    bmw_k_dl[i] = format_k(bmw_k_dl[i])
    # print("bmw_r: block size: %d, top k: %d, average time: %d." %(bmw_block_r[i], bmw_k_r[i], bmw_avg_r[i]))
    # bmw_block_r[i] = format_block(bmw_block_r[i])
    # bmw_k_r[i] = format_k(bmw_k_r[i])
    # print("bmw_dp_r: block size: %d, top k: %d, average time: %d." %(bmw_block_dp_r[i], bmw_k_dp_r[i], bmw_avg_dp_r[i]))
    # bmw_block_dp_r[i] = format_block(bmw_block_dp_r[i])
    # bmw_k_dp_r[i] = format_k(bmw_k_dp_r[i])
    # print("bmw: block size: %d, top k: %d, average time: %d." %(tp_block[i], tp_k[i], tp_avg[i]))
    # tp_block[i] = format_block(tp_block[i])
    # tp_k[i] = format_k(tp_k[i])
    # print("tp: block size: %d, top k: %d, average time: %d." %(tp_block_dp[i], tp_k_dp[i], tp_avg_dp[i]))
    # tp_block_dp[i] = format_block(tp_block_dp[i])
    # tp_k_dp[i] = format_k(tp_k_dp[i])

colors = ['blue','red'] #'red','orange','green'
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xticks([0, 1, 2, 3])
ax.set_yticks([0, 1])
ax.set_xticklabels((0, -1, 512, 1024))
ax.set_yticklabels((0, 1000))

ax.axes.set_xlim3d(left = 0, right = 3) 
ax.axes.set_ylim3d(bottom = 0, top = 1)  

# scatter1 = ax.scatter(tp_block, tp_k, tp_avg, c = colors[0], marker = 'o')
# scatter2 = ax.scatter(tp_block_dp, tp_k_dp, tp_avg_dp, c = colors[1], marker = 'v')
# scatter1 = ax.scatter(bmw_block, bmw_k, bmw_avg, c = colors[0], marker = 's')
scatter1 = ax.scatter(bmw_block_dp, bmw_k_dp, bmw_avg_dp, c = colors[0], marker = 'p')
scatter2 = ax.scatter(bmw_block_dl, bmw_k_dl, bmw_avg_dl, c = colors[1], marker = 's')
# scatter2 = ax.scatter(bmw_block_dp_r, bmw_k_dp_r, bmw_avg_dp_r, c = colors[1], marker = 'p')

scatter1_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[0], marker = 'p')
scatter2_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[1], marker = 's')
# scatter3_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[2], marker = 's')
# scatter4_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[3], marker = 'p')
ax.legend([scatter1_proxy,scatter2_proxy], ['bmw_deepimpact','bmw_deep_layered'], numpoints = 1)
ax.set_xlabel('block size')
ax.set_ylabel('topK')
ax.set_zlabel('avg time')

plt.savefig('graph.png')
