import argparse
import importlib
import os
import packaging
from typing import Dict

import bitsandbytes
import torch
import transformers
import peft

import sg_tools



DEFAULT_PAD_TOKEN = "[PAD]"


def is_ipex_available():
  def get_major_and_minor_from_version(full_version):
    return str(packaging.version.parse(full_version).major) + "." + str(packaging.version.parse(full_version).minor)

  _torch_version = importlib.metadata.version("torch")
  if importlib.util.find_spec("intel_extension_for_pytorch") is None:
    return False
  _ipex_version = "N/A"
  try:
    _ipex_version = importlib.metadata.version("intel_extension_for_pytorch")
  except importlib.metadata.PackageNotFoundError:
    return False
  torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
  ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
  if torch_major_and_minor != ipex_major_and_minor:
    print(
      f"Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*,"
      f" but PyTorch {_torch_version} is found. Please switch to the matching version and run again."
    )
    return False
  return True


def find_all_linear_names(args, model):
  cls = torch.nn.Linear
  if args.bits == 4:
    cls = bitsandbytes.nn.Linear4bit
  elif args.bits == 8:
    cls = bitsandbytes.nn.Linear8bitLt

  lora_module_names = set()
  for name, module in model.named_modules():
    if isinstance(module, cls):
      names = name.split('.')
      lora_module_names.add(names[0] if len(names) == 1 else names[-1])

  if 'lm_head' in lora_module_names: # needed for 16-bit
    lora_module_names.remove('lm_head')
  return list(lora_module_names)


def __smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
  """토크나이저와 임베딩을 리사이징합니다.

  Note: 최적화되지 않은 버전으로, 임베딩 사이즈가 64로 나누어 떨어지지 않을 수 있습니다.
  """
  num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
  model.resize_token_embeddings(len(tokenizer))
  
  if num_new_tokens > 0:
    input_embeddings_data = model.get_input_embeddings().weight.data
    output_embeddings_data = model.get_output_embeddings().weight.data

    input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
    output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

    input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
    output_embeddings_data[-num_new_tokens:] = output_embeddings_avg


def get_model_tokenizer(
  args: argparse.Namespace,
  force_model: str, # 'code_llama'
) -> tuple[(transformers.PreTrainedModel | peft.PeftModel | peft.PeftMixedModel), transformers.PreTrainedTokenizer]:
  # peft_merge 모드 검증
  peft_merge = getattr(args, 'peft_merge', False)
  if peft_merge and args.do_train:
    raise ValueError("peft_merge=True cannot be used with do_train=True. peft_merge is for inference only.")
  if peft_merge and args.full_finetune:
    raise ValueError("peft_merge=True cannot be used with full_finetune=True. peft_merge requires LoRA checkpoints.")

  # 마지막 세팅 불러오기
  checkpoint_dir, completed_training = sg_tools.get_last_checkpoint(args.output_dir)
  if completed_training:
    print('Detected that training was already completed!')
  if checkpoint_dir is not None:
    print(f'checkpoint founded: {checkpoint_dir}')

  # peft_merge 모드에서는 checkpoint가 필수
  if peft_merge and checkpoint_dir is None:
    raise ValueError("peft_merge=True requires a checkpoint. No checkpoint found in output_dir.")

  # 멀티 GPU 환경 설정
  n_gpus = 1
  if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
  if is_ipex_available() and torch.xpu.is_available():
    n_gpus = torch.xpu.device_count()

  max_memory = None
  device_map = "auto"
  if args.max_memory_MB:
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}

  # if we are in a distributed setting, we need to set the device map and max memory per device
  if os.environ.get('LOCAL_RANK') is not None:
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device_map = {'': local_rank}
    if max_memory is not None:
      max_memory = {'': max_memory[local_rank]}

  # full_finetune 의 경우 16 또는 32 비트로만 가능
  if args.full_finetune: assert args.bits in [16, 32]

  # *** 모델 선 로드 ***
  print(f'🛤️ Loading base model {args.model_name_or_path}...')
  compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
  # compute_dtype = torch.bfloat16
  model = None
  if args.full_finetune:
    model = transformers.AutoModelForCausalLM.from_pretrained(
      args.model_name_or_path,
      cache_dir=args.cache_dir,
      device_map=device_map,
      max_memory=max_memory,
      torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
      trust_remote_code=args.trust_remote_code,
      token=args.token
    )
  elif peft_merge:
    # peft_merge 모드: CPU에서 양자화 없이 원본 모델 로드 (대용량 모델 지원)
    # GPU 메모리 부족으로 인한 meta tensor 오프로딩 문제를 방지하기 위해 CPU에서 병합 수행
    print('🔀 peft_merge mode: Loading base model on CPU without quantization for LoRA merging...')
    model = transformers.AutoModelForCausalLM.from_pretrained(
      args.model_name_or_path,
      cache_dir=args.cache_dir,
      device_map='cpu',
      torch_dtype=compute_dtype,
      trust_remote_code=args.trust_remote_code,
      token=args.token,
      low_cpu_mem_usage=True,
    )
  else:
    model = transformers.AutoModelForCausalLM.from_pretrained(
      args.model_name_or_path,
      cache_dir=args.cache_dir,
      device_map=device_map,
      max_memory=max_memory,
      quantization_config=transformers.BitsAndBytesConfig(
        load_in_4bit=args.bits == 4, # 4bit 양자화 시
        load_in_8bit=args.bits == 8, # 8bit 양자화 시
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=compute_dtype, # 정규 분포에서 초기화된 가중치에 특별한 4비트 데이터 유형을 사용
        bnb_4bit_use_double_quant=args.double_quant, # 이미 양자화된 가중치를 양자화하기 위해 중첩된 양자화 방식을 사용
        bnb_4bit_quant_type=args.quant_type, # 더 빠른 계산을 위해 bfloat16 사용
      ),
      torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
      # torch_dtype=torch.bfloat16,
      trust_remote_code=args.trust_remote_code,
      token=args.token
    )
  if compute_dtype == torch.float16 and args.bits == 4:
    if torch.cuda.is_bf16_supported():
      print('='*80)
      print('⚠️ Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
      print('='*80)
          
  if compute_dtype == torch.float16 and (is_ipex_available() and torch.xpu.is_available()):
    compute_dtype = torch.bfloat16
    print('⚠️ Intel XPU does not support float16 yet, so switching to bfloat16')

  setattr(model, 'model_parallel', True)
  setattr(model, 'is_parallelizable', True)

  model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
  # model.config.torch_dtype=torch.bfloat16


  # *** 토크나이저 로드 ***
  tokenizer = None

  # AutoTokenizer가 CodeLlamaTokenizer를 감지하지 못해서 따로 처리
  if force_model == 'code_llama':
    tokenizer = transformers.CodeLlamaTokenizer.from_pretrained(
      args.model_name_or_path,
      cache_dir=args.cache_dir,
      padding_side="right",
      use_fast=False, # Fast tokenizer giving issues.
      tokenizer_type='llama' if 'llama' in args.model_name_or_path.lower() else None, # Needed for HF name change
      trust_remote_code=args.trust_remote_code,
      token=args.token,
    )
    # Fixing some of the early LLaMA HF conversion issues.
    # tokenizer.bos_token_id = 1
  else:
    # Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
      args.model_name_or_path,
      cache_dir=args.cache_dir,
      padding_side="right",
      clean_up_tokenization_spaces=True,
      use_fast=False, # Fast tokenizer giving issues.
      tokenizer_type='llama' if 'llama' in args.model_name_or_path.lower() else None, # Needed for HF name change
      trust_remote_code=args.trust_remote_code,
      token=args.token,
    )

  if tokenizer._pad_token is None:
    __smart_tokenizer_and_embedding_resize(
      special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
      tokenizer=tokenizer,
      model=model,
    )


  if 'llama' in args.model_name_or_path.lower() or isinstance(tokenizer, transformers.CodeLlamaTokenizer):
    # LLaMa 토크나이저는 올바른 특수 토큰이 설정되어 있지 않을 수 있습니다.
    # 누락된 경우 다른 토큰으로 분석되는 것을 방지하기 위해 추가합니다.
    # 이들은 vocabulary에 포함되어 있습니다.
    # 또한 `model.config.pad_token_id`는 `<unk>` 토큰에 해당하는 0입니다.
    print('🦙 LLaMa Detected> Adding special tokens.')
    tokenizer.add_special_tokens({
      "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
      "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
      "unk_token": tokenizer.convert_ids_to_tokens(
        model.config.pad_token_id if (model.config.pad_token_id != None and model.config.pad_token_id != -1) else tokenizer.pad_token_id
      ),
    })

  # *** 모델 후 설정 ***
  # 토크나이저 조정 후에 로드 해야지만 차원 오류가 발생하지 않음
  if (not args.full_finetune) and args.do_train:
    print('🪀 Preparing model for K-bit training...')
    # 훈련을 위해 양자화된 모델을 전처리
    model = peft.prepare_model_for_kbit_training(
      model,
      use_gradient_checkpointing=args.gradient_checkpointing,
      gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs
    )

  if not args.full_finetune:
    if peft_merge:
      # peft_merge 모드: CPU에서 LoRA 병합 후 양자화하여 GPU로 로드
      print("🔗 Loading adapters from checkpoint for merging (on CPU)...")
      model = peft.PeftModel.from_pretrained(
        model,
        os.path.abspath(os.path.join(checkpoint_dir, 'adapter_model')),
        is_trainable=False,
        device_map='cpu',
      )
      print("🔀 Merging LoRA weights into base model (on CPU)...")
      model = model.merge_and_unload()
      print("✅ LoRA weights merged successfully.")

      # 병합된 모델 양자화하여 GPU로 로드
      if args.bits in [4, 8]:
        print(f"🗜️ Quantizing merged model to {args.bits}-bit and loading to GPU...")
        quantization_config = transformers.BitsAndBytesConfig(
          load_in_4bit=args.bits == 4,
          load_in_8bit=args.bits == 8,
          llm_int8_threshold=6.0,
          llm_int8_has_fp16_weight=False,
          bnb_4bit_compute_dtype=compute_dtype,
          bnb_4bit_use_double_quant=args.double_quant,
          bnb_4bit_quant_type=args.quant_type,
        )
        # 병합된 모델을 양자화하여 다시 로드
        # 먼저 임시로 저장 후 양자화하여 GPU로 로드
        import tempfile
        import gc
        with tempfile.TemporaryDirectory() as tmp_dir:
          print(f"💾 Saving merged model to temporary directory...")
          model.save_pretrained(tmp_dir)
          tokenizer.save_pretrained(tmp_dir)
          # CPU 메모리 해제
          del model
          gc.collect()
          torch.cuda.empty_cache()
          print(f"🔄 Reloading merged model with {args.bits}-bit quantization to GPU...")
          model = transformers.AutoModelForCausalLM.from_pretrained(
            tmp_dir,
            device_map=device_map,
            max_memory=max_memory,
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
            trust_remote_code=args.trust_remote_code,
          )
        print(f"✅ Merged model quantized to {args.bits}-bit and loaded to GPU successfully.")
    elif checkpoint_dir is not None:
      print("🔗 Loading adapters from checkpoint...")
      model = peft.PeftModel.from_pretrained(
        model,
        os.path.abspath(os.path.join(checkpoint_dir, 'adapter_model')),
        is_trainable=args.do_train
      )
    else:
      print(f'➕ adding LoRA modules...')
      modules = find_all_linear_names(args, model)
      config = peft.LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        # 🍞
        task_type="CAUSAL_LM",
      )
      model = peft.get_peft_model(model, config)

  if args.do_train:
    print('✂️ Modify model layer dtype...')
    for name, module in model.named_modules():
      if isinstance(module, peft.tuners.lora.LoraLayer):
        if args.bf16:
          print(f'✂️ LoraLayer module to bfloat16 ({name})')
          module = module.to(torch.bfloat16)
      if 'norm' in name:
        print(f'✂️ Norm module to float32 ({name})')
        module = module.to(torch.float32)
      if 'lm_head' in name or 'embed_tokens' in name:
        if hasattr(module, 'weight'):
          if args.bf16 and module.weight.dtype == torch.float32:
            print(f'✂️ lm_head or embed_tokens Module to bfloat16 ({name})')
            module = module.to(torch.bfloat16)
  return model, tokenizer
