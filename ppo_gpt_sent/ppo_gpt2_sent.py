from IPython import get_ipython
ipython = get_ipython()

if '__IPYTHON__' in globals():
    ipython.magic('load_ext autoreload')
    ipython.magic('autoreload 2')


import torch
from tqdm import tqdm
import pandas as pd
tqdm.pandas()

from transformers import pipeline, AutoTokenizer
from datasets import load_dataset

# use the trainer class from huggingface
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler


import warnings

import wandb
wandb.init() # login from terminal




def build_dataset(config, dataset_name="imdb", input_min_text_length=2, input_max_text_length=8):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])



if __name__ == "__main__":


    # PPO Param
    config = PPOConfig(
        model_name="lvwerra/gpt2-imdb",
        learning_rate=5e-5,
        log_with="wandb",
    )

    sent_kwargs = {
        "return_all_scores": True, 
        "function_to_apply": "none", 
        "batch_size": 32
    }
    
    dataset = build_dataset(config)

    # load the generation LM and the corresponding BERT model as RM
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    tokenizer.pad_token = tokenizer.eos_token


    # init PPO Trainer
    ppo_trainer = PPOTrainer(
        config, 
        model, 
        ref_model, 
        tokenizer, 
        dataset=dataset, 
        data_collator=collator
    )
    
    # device hardware
    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
    sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=device)

    # Generation config
    gen_kwargs = {
        "min_length": -1, 
        "top_k": 0.0, 
        "top_p": 1.0, 
        "do_sample": True, 
        "pad_token_id": tokenizer.eos_token_id
    }
    
    ## training loop
    with warnings.catch_warnings():
    warnings.simplefilter('ignore')

    output_min_length = 4
    output_max_length = 16
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }


    # training loop
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]

        # generation
        response_tensors = []
        # for each of the prompt, grab the response of the LM
        for query in query_tensors:
            gen_len = output_length_sampler()
            generation_kwargs["max_new_tokens"] = gen_len
            response = ppo_trainer.generate(query, **generation_kwargs)
            response_tensors.append(response.squeeze()[-gen_len:])
        # decode the response to fit into the RM
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        # put the prompt and the LM response together, feed into the RM pipeline to obtain scores as reward
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

        # PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
        
        
    
    # eval
    bs = 32
    game_data = dict()
    dataset.set_format("pandas")
    df_batch = dataset[:].sample(bs)
    game_data["query"] = df_batch["query"].tolist()
    query_tensors = df_batch["input_ids"].tolist()

    response_tensors_ref, response_tensors = [], []

    # get two responses before and after PPO
    for i in range(bs):
        gen_len = output_length_sampler()
        output = ref_model.generate(
            torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(device), max_new_tokens=gen_len, **gen_kwargs
        ).squeeze()[-gen_len:]
        response_tensors_ref.append(output)
        output = model.generate(
            torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(device), max_new_tokens=gen_len, **gen_kwargs
        ).squeeze()[-gen_len:]
        response_tensors.append(output)

    #### decode responses
    game_data["response (before)"] = [tokenizer.decode(response_tensors_ref[i]) for i in range(bs)]
    game_data["response (after)"] = [tokenizer.decode(response_tensors[i]) for i in range(bs)]

    #### sentiment analysis of query/response pairs before/after
    texts = [q + r for q, r in zip(game_data["query"], game_data["response (before)"])]
    game_data["rewards (before)"] = [output[1]["score"] for output in sentiment_pipe(texts, **sent_kwargs)]

    texts = [q + r for q, r in zip(game_data["query"], game_data["response (after)"])]
    game_data["rewards (after)"] = [output[1]["score"] for output in sentiment_pipe(texts, **sent_kwargs)]

    # store results in a dataframe
    df_results = pd.DataFrame(game_data)

    
    model.save_pretrained("ppo-gpt2", push_to_hub=True)
    tokenizer.save_pretrained("ppo-gpt2", push_to_hub=True)

