Adding task requires the following process

Here are the files you'd need to create/modify:

Task config — create config/task/<dataset_task>.yaml 

Define name, data class, data_dir, label_names, clip_eval_mode
Dataset class 

Load annotations 
If frame extraction is required, add it to extract_test_frames_parallel.py
Include max_samples support (default = 100)
Prompts — add entries in vlmeval/prompts.py for ('<dataset_task>', '<model>') etc.
You need to consider the unique inference properties of each model

Data registry — register the new class in vlmeval/config.py under data_map

Eval logic — add a branch in vlmeval/inference_surg.py eval_data() (~line 588) to handle output parsing.

Bash script — update scripts/run_<dataset_task>_all.sh to include the new tasks


After this, take care of 
- multibypass140 dataset: s3://vlm-testbench/dataset/MultiBypass140/ (download -> unzip)
- /home/ubuntu/datasets/vlm/CholecT50/CholecT50

Same process: 
- find out all the available tasks
- extract frames by looking at https://arxiv.org/pdf/2504.02799 (make sure the disk is not full)
- check prompts - annotation matching sanity 
- check whether the model has the prompts. 
- run evaluation for qwenvl2 to confirm the evaluation result is similar with the paper
- run evaluation background and output log outputs/model_test

Resolve any errors yourself. Do not ask permission