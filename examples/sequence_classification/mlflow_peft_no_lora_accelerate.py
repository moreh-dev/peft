import argparse
import evaluate
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from peft import (
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    get_peft_model,
)
from peft.utils.other import fsdp_auto_wrap_policy
import mlflow
import time
import os

logging_steps = 100

def parse_args():
    parser = argparse.ArgumentParser(description="PEFT a transformers model on a sequence classification task")
    parser.add_argument('--log_interval', type=int, default=1, help='log interval.')
    parser.add_argument(
        "--num_virtual_tokens",
        type=int,
        default=20,
        help="num_virtual_tokens if the number of virtual tokens used in prompt/prefix/P tuning.",
    )
    parser.add_argument(
        "--encoder_hidden_size",
        type=int,
        default=128,
        help="encoder_hidden_size if the encoder hidden size used in P tuninig/Prefix tuning.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='bert-base-uncased',
        help="Path to pretrained model or model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=900,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default='outputs', help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--peft_type",
        type=str,
        default="p_tuning",
        help="The PEFT type to use.",
        choices=["p_tuning", "prefix_tuning", "prompt_tuning"],
    )
    parser.add_argument('--dataset_name', type=str, default='glue', help='The name of the Dataset (from the HuggingFace hub) to train on.')
    parser.add_argument('--cache_dir', type=str, default=None, help='Directory to read/write data.')
    parser.add_argument("--amp", type=str, choices=["bf16", "fp16", "no"], default="fp16", help="Choose AMP mode")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Choose the optimization computation method")
    parser.add_argument('--checkpoint_dir', type=str, default= './save_checkpoint', help='Directory to save checkpoints.')
    parser.add_argument('--load_checkpoint', type=str, default="False", help='Load checkpoint or not.')
    parser.add_argument('--save_checkpoint', type=str, default="False", help='Save checkpoint or not.')
    args = parser.parse_args()

    assert args.output_dir is not None, "Need an `output_dir` to store the finetune model and verify."

    return args

def get_num_parameters(model):
  num_params = 0
  for param in model.parameters():
    num_params += param.numel()
  # in million
  num_params /= 10**6
  return num_params

def main():
    args = parse_args()
    
    ddp_scaler = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision=args.amp, kwargs_handlers=[ddp_scaler])

    task = "mrpc"

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if args.peft_type == "p_tuning":
        peft_config = PromptEncoderConfig(
            task_type="SEQ_CLS",
            num_virtual_tokens=args.num_virtual_tokens,
            encoder_hidden_size=args.encoder_hidden_size,
        )
    elif args.peft_type == "prefix_tuning":
        peft_config = PrefixTuningConfig(
            task_type="SEQ_CLS",
            num_virtual_tokens=args.num_virtual_tokens,
            encoder_hidden_size=args.encoder_hidden_size,
        )
    else:
        peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=args.num_virtual_tokens)

    tokenizer_kwargs = {}

    if any(k in args.model_name_or_path for k in ("gpt", "opt", "bloom")):
        tokenizer_kwargs["padding_side"] = "left"
    else:
        tokenizer_kwargs["padding_side"] = "right"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    datasets = load_dataset(args.dataset_name, task, cache_dir=args.cache_dir)
    metric = evaluate.load("glue", task)

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
        return outputs

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    with accelerator.main_process_first():
        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["idx", "sentence1", "sentence2"],
        )

    # We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
    # transformers library
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # Instantiate dataloaders.
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"],
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.per_device_eval_batch_size,
    )

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    if getattr(accelerator.state, "fsdp_plugin", None) is not None:
        accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)
        model = accelerator.prepare(model)

    
    if args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(params=model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(params=model.parameters(), lr=args.learning_rate)

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=(len(train_dataloader) * args.num_train_epochs),
    )

    if args.load_checkpoint == "True":
        if os.path.exists(args.checkpoint_dir):
            checkpoint = torch.load(args.checkpoint_dir)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
        else:
            pass

    if getattr(accelerator.state, "fsdp_plugin", None) is not None:
        train_dataloader, eval_dataloader, optimizer, lr_scheduler = accelerator.prepare(
            train_dataloader, eval_dataloader, optimizer, lr_scheduler
        )
    else:
        model, train_dataloader, eval_dataloader, optimizer, lr_scheduler = accelerator.prepare(
            model, train_dataloader, eval_dataloader, optimizer, lr_scheduler
        )
    mlflow.start_run()
    num_params = get_num_parameters(model)
    mlflow.log_param('num_params', num_params)

    elapsed = 0
    epoch_runtime_list = []
    for epoch in range(args.num_train_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0

        for step, batch in enumerate(tqdm(train_dataloader)):
            start_time_step = time.time()
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            elapsed += time.time() - start_time_step
            total_steps = epoch * len(train_dataloader) + step + 1
            if total_steps % args.log_interval == 0:
                thoughput = total_steps * args.per_device_train_batch_size/ elapsed
                mlflow.log_metric('throughput', thoughput, step=total_steps)
                mlflow.log_metric('loss', loss, step=total_steps)
                mlflow.log_metric('lr', lr_scheduler.get_last_lr()[0], step=total_steps)

        end_time = time.time()  # End time for the epoch

        # Calculate metrics
        epoch_runtime = end_time - start_time
        epoch_runtime_list.append(epoch_runtime)
    avg_epoch_runtime = sum(epoch_runtime_list)/len(epoch_runtime_list)
    avg_throughput = len(train_dataloader) * args.per_device_train_batch_size*args.num_train_epochs/ sum(epoch_runtime_list)
    mlflow.log_metric('epoch_time', avg_epoch_runtime)  
    mlflow.log_metric('avg_throughput', avg_throughput)

    model.eval()
    samples_seen = 0
    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = accelerator.gather((predictions, batch["labels"]))
        # If we are in a multiprocess environment, the last batch has duplicates
        if accelerator.num_processes > 1:
            if step == len(eval_dataloader) - 1:
                predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                references = references[: len(eval_dataloader.dataset) - samples_seen]
            else:
                samples_seen += references.shape[0]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )
        eval_metric = metric.compute()
        
        accelerator.print(f"epoch {epoch}:", eval_metric)
        accuracy = eval_metric.get('accuracy', None)

        mlflow.log_metric('accuracy', accuracy)
    mlflow.end_run()
    
    accelerator.wait_for_everyone()
    if args.save_checkpoint == "True":
        if epoch == args.num_train_epochs - 1:
            os.makedirs(os.path.dirname(args.checkpoint_dir), exist_ok=True)
            torch.save({
            'epoch': args.num_train_epochs - 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'loss': loss,
            }, args.checkpoint_dir)

    # unwrapped_model = accelerator.unwrap_model(model)
    # unwrapped_model.save_pretrained(args.output_dir, state_dict=accelerator.get_state_dict(model))
    # if accelerator.is_main_process:
    #     tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    main()
