import argparse
from typing import Optional

from dataclasses import dataclass, field
import transformers
import torch
import peft

@dataclass
class ModelArguments:
  model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-12b")
  trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."})
  token: Optional[str] = field(default=None, metadata={"help": "Enables using Huggingface auth token from Git Credentials."})

@dataclass
class DataArguments:
  eval_dataset_size: int = field(default=1000, metadata={"help": "Size of validation dataset."})
  max_train_samples: Optional[int] = field(default=None, metadata={"help": "For debugging purposes or quicker training, truncate the number of training examples to this value if set."})
  max_eval_samples: Optional[int] = field(default=None, metadata={"help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set."})
  max_length: int = field(default=1024, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."})
  source_max_len: int = field(default=1024, metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."})
  target_max_len: int = field(default=256, metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."})
  dataset: str = field(default='alpaca', metadata={"help": "Which dataset to finetune on. See datamodule for options."})
  dataset_format: Optional[str] = field(default=None, metadata={"help": "Which dataset format is used. [alpaca|chip2|self-instruct|hh-rlhf]"})

  # CUSTOM ARGS
  validation_dataset: Optional[str] = field(default=None, metadata={"help": "Which dataset to validate on."})

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
  # Train
  do_train: bool = field(default=False, metadata={"help": 'To train or not to train, that is the question?'})
  train_on_source: Optional[bool] = field(default=False, metadata={"help": "Whether to train on the input in addition to the target text."})
  num_train_epochs: int = field(default=1, metadata={"help": 'How many epochs to train for'})
  weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
  optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
  max_steps: int = field(default=-1, metadata={"help": 'How many optimizer update steps to take'})
  adam_beta1: float = field(default=0.9, metadata={"help": 'The beta1 hyperparameter for the Adam optimizer'})
  adam_beta2: float = field(default=0.999, metadata={"help": 'The beta2 hyperparameter for the Adam optimizer'})
  learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
  lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
  max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
  warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
  seed: int = field(default=42, metadata={"help": 'Random seed'})
  
  # QLoRA
  full_finetune: bool = field(default=False, metadata={"help": "Finetune the entire model without adapters."})
  adam8bit: bool = field(default=False, metadata={"help": "Use 8-bit adam."})
  double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
  quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
  bits: int = field(default=4, metadata={"help": "How many bits to use."})
  fp16: bool = field(default=False, metadata={"help": "Use fp16."})
  bf16: bool = field(default=True, metadata={"help": "Use bfloat16."})
  lora_r: int = field(default=64, metadata={"help": "Lora R dimension."})
  lora_alpha: float = field(default=16, metadata={"help": " Lora alpha."})
  lora_dropout: float = field(default=0.1, metadata={"help":"Lora dropout."})
  peft_merge: bool = field(default=False, metadata={"help": "Merge LoRA weights into base model for faster inference (requires checkpoint)."})

  # Env
  cache_dir: Optional[str] = field(default=None)
  output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})

  # GPU Memory
  max_memory_MB: int = field(default=None, metadata={"help": "Free memory per gpu."})
  per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
  gradient_accumulation_steps: int = field(default=1, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
  gradient_checkpointing: bool = field(default=False, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
  
  # Utils
  remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})

  # Logging
  report_to: str = field(default='none', metadata={"help": "To use wandb or something else for reporting."})
  logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
  group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})

  # CUSTOM ARGS
  eval_steps: int = field(default=1000, metadata={"help": 'The frequency of update steps after which to evaluate the model'})
  use_reentrant: Optional[bool] = field(default=False, metadata={"help": "Use reentrant memory for gradient checkpointing."})

  # Save
  save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints'})
  save_steps: int = field(default=500, metadata={"help": 'How often to save a model'})
  specific_save_steps: Optional[str] = field(default=None, metadata={"help": 'Save model at specific steps separated by comma'})
  save_total_limit: int = field(default=20, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})
  # mmlu_split: Optional[str] = field(default='eval', metadata={"help": "The MMLU split to run on"})
  # mmlu_dataset: Optional[str] = field(default='mmlu-fs', metadata={"help": "MMLU dataset to use: options are `mmlu-zs` for zero-shot or `mmlu-fs` for few shot."})
  # do_mmlu_eval: Optional[bool] = field(default=False, metadata={"help": "Whether to run the MMLU evaluation."})
  # max_mmlu_samples: Optional[int] = field(default=None, metadata={"help": "If set, only evaluates on `max_mmlu_samples` of the MMMLU dataset."})
  # mmlu_source_max_len: int = field(default=2048, metadata={"help": "Maximum source sequence length for mmlu."})

@dataclass
class GenerationArguments:
  # For more hyperparameters check:
  # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig

  # Benchmark type
  do_humaneval: Optional[bool] = field(default=False, metadata={"help": "Whether to run the humaneval evaluation."})
  do_quixbugs: Optional[bool] = field(default=False, metadata={"help": "Whether to run the quixbugs evaluation."})
  do_defects4j: Optional[bool] = field(default=False, metadata={"help": "Whether to run the defects4j evaluation."})
  strict_defects4j: Optional[bool] = field(default=False, metadata={"help": "Whether to run the defects4j evaluation strictly."})
  validate_result_split_defects4j: Optional[bool] = field(default=False, metadata={"help": "Whether to split the results for defects4j v1.2 and v2.0."})

  # Patch generation & validation
  do_generate: Optional[bool] = field(default=False, metadata={"help": "Patch code generation form benchmark data."})
  do_validate: Optional[bool] = field(default=False, metadata={"help": "Validate the generated patches."})
  
  # Generation strategy
  do_sample: Optional[bool] = field(default=False)
  num_beams: Optional[int] = field(default=10)
  num_beam_groups: Optional[int] = field(default=1)
  penalty_alpha: Optional[float] = field(default=None)
  use_cache: Optional[bool] = field(default=True)

  # Length arguments
  max_new_tokens: Optional[int] = field(default=64, metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops if predict_with_generate is set."})
  min_new_tokens : Optional[int] = field(default=None, metadata={"help": "Minimum number of new tokens to generate."})

  # Hyperparameters for logit manipulation
  temperature: Optional[float] = field(default=0.7)
  top_k: Optional[int] = field(default=50)
  top_p: Optional[float] = field(default=0.9)
  typical_p: Optional[float] = field(default=1.0)
  diversity_penalty: Optional[float] = field(default=0.0)
  repetition_penalty: Optional[float] = field(default=1.0)
  length_penalty: Optional[float] = field(default=1.0)
  no_repeat_ngram_size: Optional[int] = field(default=0)


def parse_args():
  hfparser = transformers.HfArgumentParser((
    ModelArguments, DataArguments, TrainingArguments, GenerationArguments
  ))
  model_args, data_args, training_args, generation_args, extra_args = \
    hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
  
  # https://github.com/huggingface/transformers/issues/26969#issuecomment-1807831645
  if training_args.gradient_checkpointing:
    # TODO: 🍞
    if training_args.use_reentrant:
      print('📀 Use reentrant memory for gradient checkpointing.')
    training_args.gradient_checkpointing_kwargs = {
      # DeepSeek-Coder-V2 학습간에 오류: torch.utias.checkpoint.ChsckpointError: torch.utils.checkpoint: Recomputed values for the following tensors have different metadata than during the forward pass.
      # https://github.com/huggingface/transformers/issues/28499#issuecomment-2015282622
      'use_reentrant': training_args.use_reentrant,
    }

  training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
  # print(model_args, data_args, training_args, generation_args, extra_args)
  args = argparse.Namespace(
    **vars(model_args), **vars(data_args), **vars(training_args),
  )
  # if report_to is array, convert to string
  if isinstance(args.report_to, list) and len(args.report_to) > 0:
    args.report_to = args.report_to[0]

  return (args, model_args, data_args, training_args, generation_args, extra_args)
