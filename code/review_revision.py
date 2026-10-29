"""Entry point for the independent review audits. Historical fits are read only."""
import os,argparse,importlib
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='1'
from threadpoolctl import threadpool_limits

TASKS=['mapping_coordinate','mapping','risks','exact','public','factorial','artifacts','verify','detail']
def run(task):
    if task in ['mapping','mapping_coordinate']:
        import review_mapping
        review_mapping.main(strategy='exact' if task=='mapping' else 'coordinate')
    else:
        module={'artifacts':'make_review_artifacts','verify':'verify_review_revision','detail':'verify_detail_revision'}.get(task,'review_'+task)
        importlib.import_module(module).main()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=['all']+TASKS,default='verify')
    args=p.parse_args()
    with threadpool_limits(limits=1):
        for task in TASKS if args.task=='all' else [args.task]:run(task)
