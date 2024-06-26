import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup

from peft import AdaLoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model
import mlflow
import time

def get_num_parameters(model):
  num_params = 0
  for param in model.parameters():
    num_params += param.numel()
  # in million
  num_params /= 10**6
  return num_params


os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = "cuda"
model_name_or_path = "facebook/bart-base"
tokenizer_name_or_path = "facebook/bart-base"

checkpoint_name = "financial_sentiment_analysis_lora_v1.pt"
text_column = "sentence"
label_column = "text_label"
max_length = 128
lr = 1e-3
num_epochs = 8
batch_size = 8


# creating model
peft_config = AdaLoraConfig(
    init_r=12,
    target_r=8,
    beta1=0.85,
    beta2=0.85,
    tinit=200,
    tfinal=1000,
    deltaT=10,
    lora_alpha=32,
    lora_dropout=0.1,
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
if (not mlflow_uri):
    mlflow_uri = "http://127.0.0.1:5001"
    mlflow.set_tracking_uri(mlflow_uri)
# mlflow initial
experiment_id = mlflow.create_experiment('conditional_generation-{}'.format(model_name_or_path))
experiment = mlflow.get_experiment(experiment_id)
mlflow_runner = mlflow.start_run(run_name=model_name_or_path, experiment_id=experiment.experiment_id)

num_params = get_num_parameters(model)
mlflow.log_param('num_params', num_params)

# loading dataset
dataset = load_dataset("financial_phrasebank", "sentences_allagree")
dataset = dataset["train"].train_test_split(test_size=0.1)
dataset["validation"] = dataset["test"]
del dataset["test"]

classes = dataset["train"].features["label"].names
dataset = dataset.map(
    lambda x: {"text_label": [classes[label] for label in x["label"]]},
    batched=True,
    num_proc=1,
)


# data preprocessing
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


def preprocess_function(examples):
    inputs = examples[text_column]
    targets = examples[label_column]
    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(targets, max_length=3, padding="max_length", truncation=True, return_tensors="pt")
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs


processed_datasets = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["validation"]

train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)
eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)


# optimizer and lr scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)
model.base_model.peft_config.total_step = len(train_dataloader) * num_epochs


# training and evaluation
model = model.to(device)
global_step = 0

with mlflow_runner:
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()  # Start time for the epoch
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            # Update the importance of low-rank matrices
            # and allocate the budget accordingly.
            model.base_model.update_and_allocate(global_step)
            optimizer.zero_grad()
            global_step += 1

        end_time = time.time()  # End time for the epoch

        # Calculate metrics
        epoch_runtime = end_time - start_time
        samples_per_second = len(train_dataloader) / epoch_runtime
        steps_per_second = len(train_dataloader) / epoch_runtime

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_dataloader)

        # Log metrics for the epoch
        mlflow.log_metric('loss', avg_loss)
        mlflow.log_metric('total_loss', total_loss)
        mlflow.log_metric('train_runtime', epoch_runtime)
        mlflow.log_metric('train_samples_per_second', samples_per_second)
        mlflow.log_metric('train_steps_per_second', steps_per_second)

        model.eval()
        eval_loss = 0
        eval_preds = []
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            eval_preds.extend(
                tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
            )

        eval_epoch_loss = eval_loss / len(train_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(eval_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")
    mlflow.end_run()


# print accuracy
correct = 0
total = 0
for pred, true in zip(eval_preds, dataset["validation"]["text_label"]):
    if pred.strip() == true.strip():
        correct += 1
    total += 1
accuracy = correct / total * 100
print(f"{accuracy=} % on the evaluation dataset")
print(f"{eval_preds[:10]=}")
print(f"{dataset['validation']['text_label'][:10]=}")


# saving model
peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"
model.save_pretrained(peft_model_id)


ckpt = f"{peft_model_id}/adapter_model.bin"
# get_ipython().system('du -h $ckpt')


peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"

config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_id)


model.eval()
i = 13
inputs = tokenizer(dataset["validation"][text_column][i], return_tensors="pt")
print(dataset["validation"][text_column][i])
print(inputs)

with torch.no_grad():
    outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)
    print(outputs)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))
