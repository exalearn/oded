from radical import entk
import os
import argparse, sys, math

import numpy as np

class MOCU(object):

    def __init__(self):
        self.set_argparse()
        self._set_rmq()
        self.am = entk.AppManager(hostname=self.rmq_hostname, port=self.rmq_port, username=self.rmq_username, password=self.rmq_password)
        self.p = entk.Pipeline()
        self.s = entk.Stage()

    def _set_rmq(self):
        '''self.rmq_port = int(os.environ.get('RMQ_PORT', 5672))
        self.rmq_hostname = os.environ.get('RMQ_HOSTNAME', '129.114.17.185')
        self.rmq_username = os.environ.get('RMQ_USERNAME', 'litan')
        self.rmq_password = os.environ.get('RMQ_PASSWORD', 'sccDg7PxE3UjhA5L')'''
        self.rmq_port = int(os.environ.get('RMQ_PORT', 31848))
        self.rmq_hostname = os.environ.get('RMQ_HOSTNAME', 'apps.marble.ccs.ornl.gov')
        self.rmq_username = os.environ.get('RMQ_USERNAME', 'admin')
        self.rmq_password = os.environ.get('RMQ_PASSWORD', 'password')

    def set_resource(self, res_desc):
        res_desc["schema"] = "local"
        self.am.resource_desc = res_desc

    def set_argparse(self):
        parser = argparse.ArgumentParser(description="MOCU_EnTK")
        parser.add_argument("--num_sim", "-n", help="number of simulations")
        args = parser.parse_args()
        self.args = args
        if args.num_sim is None:
            parser.print_help()
            sys.exit(-1)

    def runMainForPerformanceMeasure_py(self, num_sim):

        for i in range(int(num_sim)):#range(1, int(num_sim) + 1):
            t = entk.Task()
            t.pre_exec = [
                "export INPUT=/gpfs/alpine/csc299/scratch/litan/MOCU/Byung-Jun/KuramotoModel/N9ForShare",
                ". /ccs/home/litan/miniconda3/etc/profile.d/conda.sh",
                "conda activate mocu",
                "module load gcc/7.4.0 cuda/10.1.243",
                "export PYCUDA_CACHE_DIR=/gpfs/alpine/csc299/scratch/litan/.cache",
                "rm -f $INPUT/results/*",
                "export OMP_NUM_THREADS=1"
                ]
            t.executable = '/ccs/home/litan/miniconda3/envs/mocu/bin/python3.6'
            t.arguments = ['$INPUT/runMainForPerformanceMeasure.py', '-n{}'.format(num_sim), '-i{}'.format(i)]
            t.post_exec = ["export TASK_ID={}".format(t.uid),"echo $TASK_ID | cut -d \".\" -f 2"]
            t.cpu_reqs = {
                'processes': 1,
                'process_type': None,
                'threads_per_process': 4,
                'thread_type': 'OpenMP'
            }
            t.gpu_reqs = {
                'processes': 1,
                'process_type': None,
                'threads_per_process': 1,
                'thread_type': 'CUDA'
            }
            self.s.add_tasks(t)
        self.p.add_stages(self.s)

    def run(self):
        self.am.workflow = [self.p]
        self.am.run()


if __name__ == "__main__":

    mocu = MOCU()
    n_nodes = math.ceil(float(int(mocu.args.num_sim)/6))
    mocu.set_resource(res_desc = {
        'resource': 'ornl.summit',
        'queue'   : 'batch',
        'walltime': 1440, #MIN
        'cpus'    : 168 * n_nodes,
        'gpus'    : 6 * n_nodes,
        'project' : 'MED110'
        })
    mocu.runMainForPerformanceMeasure_py(num_sim=mocu.args.num_sim)
    mocu.run()
