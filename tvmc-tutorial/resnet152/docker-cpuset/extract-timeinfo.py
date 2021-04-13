import os
import sys
import csv
from typing import List


def ectract_time_from_file(filepath: str) -> List[dict]:
    # Check whether file exists
    if not os.path.exists(filepath):
        sys.exit(f"No such file: {filepath}")

    # Extract Docker cpu limit from filename
    # e.g., 60-cpu-execution-time.log
    docker_cpu_limit = filepath.split("-")[0]

    with open(filepath, "r") as f:
        lines = f.readlines()

    ret = []

    for line in lines:
        # Extract model tuned cpu usage from first line of each result block
        # e.g., docker-60cpu-resnet152-module.tar
        if line.startswith("docker"):
            items = line.split("-")
            tuned_cpu_usage = items[1].rstrip("cpu")
            model_name = items[2]
            # Set nextLineGetTime = True when find lines starts with "mean(ms)"
            nextLineGetTime = False

        if line.startswith("mean"):
            # The next line is execution time
            # Mark nextLineGetTime = True and iterate next line
            nextLineGetTime = True
            continue

        if nextLineGetTime:
            mean, max, min, std = line.split()
            d = {
                "model_name": model_name,
                "tune_cores": tuned_cpu_usage,
                "run_cores": docker_cpu_limit,
                "mean(ms)": mean,
                "max(ms)": max,
                "min(ms)": min,
                "std(ms)": std,
            }
            ret.append(d)
            nextLineGetTime = False

    return ret


def list_of_dict_to_csv(l: List[dict], output_filename: str):
    keys = l[0].keys()
    with open(output_filename, "w") as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(l)


if __name__ == '__main__':
    # Test
    args = sys.argv
    if len(args) != 3:
        sys.exit("This script should have 3 arguments. e.g., python3 extract-timeinfo [logfile] [output-csvfile]")
    result = ectract_time_from_file(args[1])
    list_of_dict_to_csv(result, args[2])
