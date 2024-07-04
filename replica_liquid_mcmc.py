import numpy as np
import json
import multiprocessing as mp
import argparse
import subprocess
import os
import functools

parser = argparse.ArgumentParser(description='Replica MCMC simulation of liquid droplet morphologies')
parser.add_argument('--kT', type=float, default=10.0, help='temperature')
parser.add_argument('--J', type=float, default=1.0, help='coupling constant')
parser.add_argument('--vf', type=float, default=0.5, help='volume fraction')
parser.add_argument('--nrows', type=int, default=40, help='number of rows')
parser.add_argument('--ncols', type=int, default=40, help='number of cols')
parser.add_argument('--niters', type=int, default=1000000, help='number of iterations')
parser.add_argument('--burnin', type=int, default=200000, help='number of burn-in iterations')
parser.add_argument('--burnin_schedule', type=str, default='[1000, 100, 10, 2, 1]', help='burn-in schedule multipler for kT')
parser.add_argument('--clstat_freq', type=int, default=500, help='number of iterations per cluster computations')
parser.add_argument('--outfreq', type=int, default=10000, help='number of iterations per diagnostic information')
parser.add_argument('--outdir', type=str, default="temp", help='output directory')
parser.add_argument('--do_plots', default=False, action="store_true", help='create plots of microstates and clusters')
parser.add_argument('--np', type=int, default=1, help='number of processes')
parser.add_argument('--nr', type=int, default=1, help='number of replicas')
parser.add_argument('--python_bin', type=str, default='python', help='python command')

replica_args = ['kT', 'J', 'vf', 'nrows', 'ncols', 'niters', 'burnin', 'burnin_schedule', 'clstat_freq', 'outfreq', 'outdir']

def run_replica(replica_id, rargs):
    global replica_args
    process_args = [rargs["python_bin"], "liquid_mcmc.py", "--replica_id", str(replica_id)]
    for replica_arg in replica_args:
        process_args.append("--{}".format(replica_arg))
        process_args.append(str(rargs[replica_arg]))
    if rargs["do_plots"]:
        process_args.append("--do_plots")
    print("Running process: ", process_args)
    subprocess.run(process_args)

namespace_args = parser.parse_args()
rargs = vars(namespace_args)
rargs["burnin_schedule"] = eval(rargs["burnin_schedule"])

if __name__ == '__main__':
    outdir = rargs["outdir"] 
    replica_results = []
    pool = mp.Pool(processes=rargs["np"])
    rr_curry = functools.partial(run_replica, rargs=rargs)
    pool.map(rr_curry, range(rargs["nr"]))
    replica_results = [json.load(open(os.path.join(outdir, "results_rid-{}.json".format(replica_id)), 'r')) for replica_id in range(rargs["nr"])]
    pooled_results = {}
    for k in replica_results[0].keys():
        pooled_results[k] = sum([result[k] for result in replica_results]) / rargs["nr"]
    print()
    print('Thermodynamic averages (pool averages)')
    print('----------------------')
    for (k, v) in pooled_results.items():
        print('    <{}> = {}'.format(k, v))
    np.save(os.path.join(outdir, "results_pooled.npy"), pooled_results)
    with open(os.path.join(outdir, "results_pooled.json"), 'w') as file:
        file.write(json.dumps(pooled_results))

