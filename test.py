import argparse
from PC2_eval.eval import pc2_eval_all

parser = argparse.ArgumentParser()
parser.add_argument('--prd_dir', type=str, default="demo/pred",
                    help=" predicted .xyz dir")
parser.add_argument('--gt_dir', type=str, default="demo/gt",
                    help=" output ground truth .xyz file")
parser.add_argument('--mesh_dir', type=str, default="demo/mesh",
                    help=" mesh file dir")
parser.add_argument('--csv_dir', type=str, default="demo",
                    help=" output .csv dir")
args = parser.parse_args()

returns = pc2_eval_all(prd_dir=args.prd_dir,
                       gt_dir=args.gt_dir,
                       mesh_dir=args.mesh_dir,
                       csv_dir=args.csv_dir
                       )

print(returns)