# pl_prompt_sst

An example project using [OpenPrompt](https://github.com/thunlp/OpenPrompt) under the framework of [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) for a training prompt-based text classification model on SST2 sentiment analysis dataset. Leveraging the pytorch-lightning features like logging, gradient accumulation and early stopping, etc. Can be used as a template for further development.

## Run

Install requirement

```bash
pip install -r requirements.txt
```

Setup the prompt to use in `sst2/prompt_config.json`

```json
{
    "template_text": "{\"placeholder\": \"text_a\"} In summary, the film was {\"mask\"}.",
    "label_words": [["bad"], ["good"]]
}
```

Adjust the arguments in `run.sh` or the code below for your need, and run it.

```bash
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
```

In my preliminary experiment with the settings above, the model achieve 0.822 F1 compared to 0.820 without prompt.

## Note

Can only be executed after this [fix](https://github.com/blmoistawinde/OpenPrompt/commit/7a3ed2acf73f400a5848564e1914b7b261785b4d) on `state_dict()`