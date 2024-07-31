import numpy as np
import os
import json
import csv
import sys

outfile = sys.argv[1] if len(sys.argv) > 1 else 'results_table.csv'

headers = []
csvfile = open(outfile, 'w')
csvwriter = csv.writer(csvfile, delimiter=',')
for d in os.listdir():
    if not os.path.isdir(d):
        continue
    print(d)
    result_filename = os.path.join(d, 'results_pooled.json')
    result_file = open(result_filename, 'r')
    result = json.load(result_file)
    print(result)
    result_file.close()
    split_values = d.split('_')
    kT_val = float(split_values[1].split('-')[1]) / 100.0
    vf_val = float(split_values[2].split('-')[1]) / 100.0
    print(kT_val, vf_val)
    print()
    if len(headers) == 0:
        headers = list(result.keys())
        csvwriter.writerow(['kT', 'vf'] + headers)
    csvwriter.writerow([kT_val, vf_val] + [result[k] for k in headers])

csvfile.close()
