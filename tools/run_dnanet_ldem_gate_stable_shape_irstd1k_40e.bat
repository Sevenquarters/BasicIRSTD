@echo off
cd /d "D:\Program Files (x86)\IRSTD\BasicIRSTD"
"D:\Program Files (x86)\IRSTD\BasicIRSTD\venv\Scripts\python.exe" train.py ^
  --model_names DNANet-LDEM-Gate-Stable-Shape ^
  --dataset_names IRSTD-1K ^
  --batchSize 16 ^
  --patchSize 256 ^
  --nEpochs 40 ^
  --threads 1 ^
  --intervals 1 ^
  --eval_intervals 1 ^
  --save_intervals 10 ^
  --save ./log/exp_dnanet_ldem_gate_stable_shape_irstd1k_40e ^
  1> log\exp_dnanet_ldem_gate_stable_shape_irstd1k_40e_stdout.txt ^
  2> log\exp_dnanet_ldem_gate_stable_shape_irstd1k_40e_stderr.txt
