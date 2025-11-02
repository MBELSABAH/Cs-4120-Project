"""
train_nn.py
Neural network training for classification and regression.
Entry point: python src/train_nn.py
"""

import argparse, mlflow
from utils import set_seeds
def main(args):
    set_seeds(42)
    with mlflow.start_run(run_name="NN_stub"):
        mlflow.log_param("note", "NN to be implemented for Final per plan in Midpoint.")
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/heart.csv")
    args = p.parse_args()
    main(args)
