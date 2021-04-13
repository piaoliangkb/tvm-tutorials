import os
import subprocess


TVMC = "python3 -m tvm.driver.tvmc"
COMPILE = f"{TVMC} compile"
TUNE = f"{TVMC} tune"
RUN = f"{TVMC} run"

TIMECOMMAND = "/usr/bin/time -v"

TARGET = "llvm -mcpu=cascadelake"
TARGET_AVX512 = "llvm -mcpu=cascadelake-avx512"

NUM_THREADS = [40, 20, 10, 5, 3, 1]


def tune(num_of_threads: int, model_path: str):
    # TODO: Modify model_name
    model_name = model_path
    tunelog_prefix = f"{num_of_threads}-threads"
    tunelog_output_filename = f"{tunelog_prefix}-{model_name}.json"
    cmd = f"{TIMECOMMAND} {TUNE} --target {TARGET_AVX512} --output {tunelog_output_filename} {model_path}"
    cmd = cmd.split()
    tune_ret = subprocess.run(
        cmd, capture_output=True, encoding="utf-8",
    )
    if tune_ret != 0:
        print(f"tune error: output: {tune_ret.stdout}, err: {tune_ret.stderr}")
    return tune_ret


def evaluate(num_of_threads: int):
    pass


# if __name__ == '__main__':