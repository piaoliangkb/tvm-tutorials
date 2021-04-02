
TVM docs ref: [Getting Started with TVM command line driver - TVMC](https://tvm.apache.org/docs/tutorials/get_started/tvmc_command_line_driver.html)

- resnet50-v2-7.onnx: ResNet-50 V2 model in ONNX format

- tvmc-tutorial.py:
    
    - download and pre-process image, generate model input `imagenet_cat.npz

    - post-process model output `predictions.npz`

- compiled_module.tar: directly compile onnx module using `tvmc compile`

    `python3 -m tvm.driver.tvmc compile --target "llvm -mcpu=cascadelake" --output compiled_module.tar resnet50-v2-7.onnx`

- aftertuned_module.tar: then use records to generate module:

    - tune this model and generate `cascadelake_autotuner_records.json`:
    
    ```
    python3 -m tvm.driver.tvmc tune --target "llvm -mcpu=cascadelake" --output cascadelake_autotuner_records.json resnet5-v2-7.onnx
    ```

    - use `cascadelake_autotuner_records.json` to get a new model:
    
    ```
    python3 -m tvm.driver.tvmc compile --tuning-records cascadelake_autotuner_records.json --target "llvm -mcpu=cascadelake" --output aftertuned_module.tar resnet50-v2-7.onnx
    ```

- load-and-run-onnx.py: load onnx model, use `imagenet_cat.npz` as input data to predict
