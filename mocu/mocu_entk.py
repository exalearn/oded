from radical import entk
import os
import argparse, sys, math

import numpy as np
from mocu.utils.utils import *
from mocu.utils.costfunctions import *
from mocu.src.experimentaldesign import *
from mocu.src.mocu_utils import *
from mocu.scripts.example_linearsys import *

class MOCU(object):

    def __init__(self):
        self.set_argparse()
        self._set_rmq()
        self.am = entk.AppManager(hostname=self.rmq_hostname, port=self.rmq_port, username=self.rmq_username, password=self.rmq_password)
        self.p = entk.Pipeline()
        self.s1 = entk.Stage()
        self.s2 = entk.Stage()

    def _set_rmq(self):
        self.rmq_port = int(os.environ.get('RMQ_PORT', 5672))
        self.rmq_hostname = os.environ.get('RMQ_HOSTNAME', '129.114.17.185')
        self.rmq_username = os.environ.get('RMQ_USERNAME', 'litan')
        self.rmq_password = os.environ.get('RMQ_PASSWORD', 'sccDg7PxE3UjhA5L')

    def set_resource(self, res_desc):
        res_desc["schema"] = "local"
        self.am.resource_desc = res_desc

    def set_argparse(self):
        parser = argparse.ArgumentParser(description="MOCU_EnTK")
        parser.add_argument("--num_run", "-n", help="number of OED runs")
        parser.add_argument("--theta", "-t", help="value of theta")
        parser.add_argument("--psi", "-p", help="value of Psi")
        parser.add_argument("--s", "-s", help="value of s")
        args = parser.parse_args()
        self.args = args
        if args.num_run is None or args.theta is None or args.psi is None or args.s is None:
            parser.print_help()
            sys.exit(-1)

    def example_linearsys_py(self, num_run, theta, psi, s):

        #os.system("mkdir ../MOCU_data")

        for i in range(int(num_run)):#range(1, int(num_exp) + 1):
            t = entk.Task()
            t.pre_exec = [
                "export INPUT=/gpfs/alpine/csc299/scratch/litan/MOCU/new/mocu/scripts",
                #"cd ../MOCU_data; rm -f Ji_*",
                "export OMP_NUM_THREADS=1"
                ]
            t.executable = '/ccs/home/litan/miniconda3/envs/mocu/bin/python3.6'
            t.arguments = ['$INPUT/example_linearsys_stage1.py', '-n{}'.format(num_run), '-i{}'.format(i), '-t{}'.format(theta), '-p{}'.format(psi), '-s{}'.format(s)]
            #t.post_exec = ["mv Ji_{} ../MOCU_data".format(i+1)]
            t.post_exec = ["mv Ji_{}.npy ..".format(i+1)]
            t.cpu_reqs = {
                'processes': 1,
                'process_type': None,
                'threads_per_process': 4,
                'thread_type': 'OpenMP'
            }
            '''t.gpu_reqs = {
                'processes': 1,
                'process_type': None,
                'threads_per_process': 1,
                'thread_type': 'CUDA'
            }'''
            self.s1.add_tasks(t)
        self.p.add_stages(self.s1)

        t = entk.Task()
        t.pre_exec = [
            "export INPUT=/gpfs/alpine/csc299/scratch/litan/MOCU/new/mocu/scripts",
            "cd ..",
            "export OMP_NUM_THREADS=1"
            ]
        t.executable = '/ccs/home/litan/miniconda3/envs/mocu/bin/python3.6'
        t.arguments = ['$INPUT/example_linearsys_stage2.py', '-n{}'.format(num_run), '-t{}'.format(theta), '-p{}'.format(psi), '-s{}'.format(s)]
        t.post_exec = ["rm -f Ji_*.npy"]
        t.cpu_reqs = {
                'processes': 1,
                'process_type': None,
                'threads_per_process': 4,
                'thread_type': 'OpenMP'
        }
        '''t.gpu_reqs = {
                'processes': 1,
                'process_type': None,
                'threads_per_process': 1,
                'thread_type': 'CUDA'
        }'''
        self.s2.add_tasks(t)
        self.p.add_stages(self.s2)

    def run(self):
        self.am.workflow = [self.p]
        self.am.run()


if __name__ == "__main__":

    mocu = MOCU()
    n_nodes = math.ceil(float(int(mocu.args.num_run)/41))
    mocu.set_resource(res_desc = {
        'resource': 'ornl.summit',
        'queue'   : 'batch',
        'walltime': 120, #MIN
        'cpus'    : 168 * n_nodes,
        'gpus'    : 6 * n_nodes,
        'project' : 'MED110'
        })
    mocu.example_linearsys_py(num_run=mocu.args.num_run, theta=mocu.args.theta, psi=mocu.args.psi, s=mocu.args.s)
    mocu.run()
