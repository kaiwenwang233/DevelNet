# DevelNet
A detection neural network for Develocorder films

## Run DevelNet prediction
python run.py --data_dir=./Films/197303  --data_list=./Films/197303/197303.csv --mode=pred --ckdir=./log/0828224334 --output_dir=./test_pred --save_result --batch_size=10 --input_length=1100

## Write prediction results to txt file
python3 predpicks.py

## Earthquake examples
Film scans from Rangely experiment(1973). Images size: 252*387 (in ./Films/197303)

## Citation
Kaiwen Wang, William Ellsworth, Gregory C. Beroza, Weiqiang Zhu, Justin L. Rubinstein; DevelNet: Earthquake Detection on Develocorder Films with Deep Learning: Application to the Rangely Earthquake Control Experiment. Seismological Research Letters 2022; doi: https://doi.org/10.1785/0220220066
