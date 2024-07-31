import sys
import os
import json

d = sys.argv[1]
pdata = {}
for f in os.listdir(d):
    fname = os.path.join(d, f)
    if fname.endswith('.json'):
        result_file = open(fname, 'r')
        result = json.load(result_file)
        result_file.close()
        for (k, v) in result.items():
            if k in pdata.keys():
                pdata[k][0] += v
                pdata[k][1] += 1
            else:
                pdata[k] = [v, 1]
adata = {}
for (k, v) in pdata.items():
    adata[k] = float(v[0]) / v[1]
with open(os.path.join(d, "results_pooled.json"), 'w') as outfile:
    json.dump(adata, outfile) 
