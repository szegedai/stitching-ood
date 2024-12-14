import argparse
import sys
import os
from pathlib import Path

from nn_stitching.utils.stitched_layers import get_stitched_layers


def _get_acc_from_summary(summary_file: str) -> float:
    with open(summary_file, "r") as f:
        for line in f.readlines():
            if line.startswith("Clean accuracy: "):
                return float(line.split("Clean accuracy: ")[-1])

    return 0


# TODO: banish to utils
def get_layers_from_summary(summary_file_path: str):
    front_layer = ""
    end_layer = ""

    with open(summary_file_path, "r") as f:
        contents = f.readlines()
        for line in contents:
            if line.startswith("Front layer"):
                front_layer = line.split(" ")[2]
            elif line.startswith("End layer"):
                end_layer = line.split(" ")[2]

    return front_layer, end_layer


def _parse_args(args):
    parser = argparse.ArgumentParser(description="Stitcher result collection")

    parser.add_argument("source_dir", type=str)
    parser.add_argument("front_model_type", type=str)
    parser.add_argument("end_model_type", type=str)

    return parser.parse_args(args)


def run(conf):
    front_layers = get_stitched_layers(conf.front_model_type)
    end_layers = get_stitched_layers(conf.end_model_type)

    acc_mtx = [[0]*len(end_layers) for _ in range(len(front_layers))]

    result_subdirs = [dir for dir in sorted(Path(conf.source_dir).iterdir(),
                                            key=os.path.getmtime)
                                  if os.path.isdir(dir)]

    for dir in result_subdirs:
        summary_file = os.path.join(dir, "summary.txt")
        front_layer, end_layer = get_layers_from_summary(summary_file)
        acc = _get_acc_from_summary(summary_file)

        front_idx = front_layers.index(front_layer)
        end_idx = end_layers.index(end_layer)

        acc_mtx[front_idx][end_idx] = acc

    print(acc_mtx)


if __name__ == "__main__":
    args = sys.argv[1:]
    conf = _parse_args(args)
    run(conf)
