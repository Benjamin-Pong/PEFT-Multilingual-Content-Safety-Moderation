# -*- coding: utf-8 -*-
"""Qwen Full_finetuning.ipynb
Refered to unsloth's documentation for finetuning Qwen models: https://docs.unsloth.ai/basics/qwen3-how-to-run-and-fine-tune
info about chat templates for qwen https://qwen.readthedocs.io/en/latest/getting_started/concepts.html
"""

# Load model directly
#!pip install -U datasets
#!pip install hf_xet
#!pip install unsloth
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from unsloth import FastLanguageModel, unsloth_trai
import torch
from transformers import TrainingArguments, Trainer
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from unsloth import is_bfloat16_supported
from huggingface_hub import login
from google.colab import drive
drive.mount('/content/drive')
import json

#load dataset
from datasets import load_dataset
from huggingface_hub import login

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("ToxicityPrompts/PolyGuardMix")
training_data = ds['train']

###### Load model ###############
#def formatting_function(examples):
#checkpoint_path = "Benjaminpwh/qwen2.5-7b-instruct-polyguard-full/checkpoint-3800"
# 1. Load model WITHOUT LoRA/PEFT


model, tokenizer = FastLanguageModel.from_pretrained(
    "Benjaminpwh/qwen2.5-7b-instruct-polyguard-full-re",
    load_in_4bit=False,  #Set to False to allow full-finetuning
    device_map="auto",

    #torch_dtype=torch.float16,
)

### check trainable parameters ####
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
print(f"Number of trainable parameters: {params.item()}")

print(training_data[0].keys())

def formatting_function(data_sample):

    prompt = data_sample['prompt']
    response = data_sample['response']
    if not response:
      response = 'None'
    response_harm_label = data_sample['response_harm_label']
    response_safety_categories = data_sample['response_safety_categories']
    prompt_harm_label = data_sample['prompt_harm_label']
    prompt_safety_categories = data_sample['prompt_safety_categories']
    response_refusal_label = data_sample['response_refusal_label']

    instructions = f'''
      <|im_start|>system You are a classifier for analyzing
      interactions between humans and LLMs.
      Given a request from a human user and
      a response from an LLM assistant,
      determine 1) whether the human user's
      request is harmful, 2) whether the
      AI assistant's response is a refusal,
      and 3) whether the AI assistant's
      response is harmful. In addition,
      identify the policies violated if
      the interaction is unsafe and harmful
      according to the safety policy with
      the below categories..

      <BEGIN UNSAFE CONTENT CATEGORIES>
      S1: Violent Crimes.
      S2: Non-Violent Crimes.
      S3: Sex Crimes.
      S4: Child Exploitation.
      S5: Defamation.
      S6: Specialized Advice.
      S7: Privacy.
      S8: Intellectual Property.
      S9: Indiscriminate Weapons.
      S10: Hate.
      S11: Self-Harm.
      S12: Sexual Content.
      S13: Elections.
      S14: Code Interpreter Abuse.
      <END UNSAFE CONTENT CATEGORIES><|im_end|>

      <|im_start|>user {prompt} <|im_end|>

      <|im_start|>assistant {response}<|im_end|>

      <|im_start|>assistant Harmful request: {prompt_harm_label}
      Request safety violations: {prompt_safety_categories}
      Response refusal: {response_refusal_label}
      Harmful response: {response_harm_label}
      Response safety violations: {response_safety_categories}<|im_end|>
      '''
    return {'text': instructions}

###### process data ########

dataset_mapped = training_data.map(formatting_function)

print(dataset_mapped[0]['text'])

if tokenizer is None:
  print("FATAL ERROR: Tokenizer failed to load. Cannot proceed.")
        # You might want to add a sys.exit() or raise an exception here
        # to stop execution if the tokenizer is essential.
else:
  print("Tokenizer loaded successfully.")






# https://unsloth.ai/blog/gradient 0 fix gradient
torch.cuda.empty_cache()
output_directory = "/content/drive/MyDrive/qwen2.5-7b-instruct-polyguard-full"


## name of your model #
repo_name = "Benjaminpwh/qwen2.5-7b-instruct-polyguard-full-re"

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Important: We're not doing masked LM
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset_mapped,
    #eval_dataset = val_data,
    dataset_text_field = "text",
    #formatting_func = formatting_func,
    #max_seq_length = max_seq_length,
    data_collator = data_collator,

    args = TrainingArguments(
        output_dir = '/content/drive/MyDrive/qwen2.5-7b-instruct-polyguard-full',
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 16 ,  #effective batch size: 4 x 2 #forward to compute activation, background
        learning_rate = 2e-4, #10^-5
        #lr_scheduler_type = "", #10^5 --> 0
        #regularization
        #weight_decay = 0.01
        fp16 = False,
        bf16 = True,
        num_train_epochs=1, #100 samples, 5 , 20 samples

        # Checkpointing and validation
        #eval_strategy="no",               # Updated from evaluation_strategy
        #eval_steps=5,                      # Evaluate every 50 steps
        save_steps=500,                      # Save every 50 steps
        save_total_limit=2,                  # Keep only the 5 most recent checkpoints
        #load_best_model_at_end=True,
        #metric_for_best_model="eval_loss",   # Might need another check
        gradient_checkpointing=True,


        # Logging parameters
        logging_dir="./logs",                # Directory for logs
        logging_steps=100,                    # Log every 5 steps
        logging_first_step=True,             # Log the first step

        # Hugging Face Hub parameters
        push_to_hub=True,                    # Push model to Hugging Face Hub
        hub_model_id=repo_name,
        hub_strategy="every_save",           # Push every time we save a checkpoint
        hub_private_repo=False,              # Make the repository public


    )
)


#trainer.train()
# unsloth_train fixes gradient_accumulation_steps
trainer_stats = unsloth_train(trainer, resume_from_checkpoint= "/content/drive/MyDrive/qwen2.5-7b-instruct-polyguard-full/checkpoint-4000")


