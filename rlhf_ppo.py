import torch
from datasets import load_dataset
from tqdm import tqdm

from transformers import AutoTokenizer, pipeline

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer


dataset = load_dataset("Dahoas/full-hh-rlhf", split="train")
dataset = dataset.rename_column("prompt", "query")
dataset = dataset.remove_columns(["response", "chosen", "rejected"])

config = PPOConfig(
    model_name="agi-css/hh-rlhf-sft",
    learning_rate=1.41e-5,
)


tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)

next(model.parameters()).is_cuda


# rm_tokenizer = AutoTokenizer.from_pretrained("weqweasdas/hh_rlhf_rm_open_llama_3b")

rm_pipe = pipeline(
    "sentiment-analysis",
    model="weqweasdas/hh_rlhf_rm_open_llama_3b",
    device="cuda",
    tokenizer=tokenizer, #rm_tokenizer
    model_kwargs={"torch_dtype": torch.bfloat16}
)

def tokenize(sample):
    sample["input_ids"] = tokenizer.encode(sample["query"])
    return sample


dataset = dataset.map(tokenize, batched=False)



ppo_trainer = PPOTrainer(
    model=model,
    config=config,
    dataset=dataset, # no train_dataset param
    tokenizer=tokenizer,
)
pipe_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 1
}

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    #### Get response from SFTModel
    response_tensors = ppo_trainer.generate(query_tensors, **pipe_kwargs)
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    #### Compute reward score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    # pipe_outputs = reward_model(texts)
    # rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
    pipe_outputs = rm_pipe(texts, **pipe_kwargs)
    rewards = [output[0]["score"] for output in pipe_outputs]

    #### Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

#### Save model
ppo_trainer.save_model("hhrlhf_ppo_model")