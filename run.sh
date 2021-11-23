CUDA_VISIBLE_DEVICES=0 python -u main.py --input_dir ./sst2 \
                                         --prompt_config_dir ./sst2/prompt_config.json \
                                         --model_class bert \
                                         --model_name_or_path prajjwal1/bert-tiny \
                                         --lr 2e-4
                                         --bs 32 \
                                         --max_seq_length 64 \
                                         --patience 4 \
                                         --accumulation 2 \
                                         --seed 666