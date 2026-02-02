import codecs
import importlib
import json
import os
import packaging
import random
import subprocess
import warnings

import numpy
import torch
import transformers



PREFIX_CHECKPOINT_DIR = transformers.trainer_utils.PREFIX_CHECKPOINT_DIR



# def is_ipex_available():
#   """
#     __DEPRECATED__
#     Intel Extension for PyTorch가 사용 가능한지 확인합니다.
#   """
#   def get_major_and_minor_from_version(full_version):
#     return str(packaging.version.parse(full_version).major) + "." + str(packaging.version.parse(full_version).minor)

#   _torch_version = importlib.metadata.version("torch")
#   if importlib.util.find_spec("intel_extension_for_pytorch") is None:
#     return False
#   _ipex_version = "N/A"
#   try:
#     _ipex_version = importlib.metadata.version("intel_extension_for_pytorch")
#   except importlib.metadata.PackageNotFoundError:
#     return False
#   torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
#   ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
#   if torch_major_and_minor != ipex_major_and_minor:
#     warnings.warn(
#       f"Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*,"
#       f" but PyTorch {_torch_version} is found. Please switch to the matching version and run again."
#     )
#     return False
#   return True



def nomalize_name_or_path_to_name(name_or_path: str):
  """
    파일 경로에서 파일 이름을 추출합니다.
  """
  return str(name_or_path).split('/')[-1].split('.')[0]



def is_checkpoint_dir(checkpoint_dir):
  """
    체크포인트 디렉토리인지 확인합니다.
  """
  return os.path.exists(checkpoint_dir) and os.path.basename(checkpoint_dir).startswith(PREFIX_CHECKPOINT_DIR)



def get_last_checkpoint(checkpoint_dir):
  """
    마지막 체크포인트 디렉토리를 반환합니다.
  """
  if is_checkpoint_dir(checkpoint_dir):
    return checkpoint_dir, True

  elif os.path.isdir(checkpoint_dir):
    is_completed = os.path.exists(os.path.join(checkpoint_dir, 'completed'))
    # if is_completed: return None, True # already finished
    max_step = 0
    for filename in os.listdir(checkpoint_dir):
      if os.path.isdir(os.path.join(checkpoint_dir, filename)) and filename.startswith(PREFIX_CHECKPOINT_DIR):
        max_step = max(max_step, int(filename.replace(f'{PREFIX_CHECKPOINT_DIR}-', '')))
    if max_step == 0: return None, is_completed # training started, but no checkpoint
    checkpoint_dir = os.path.join(checkpoint_dir, f'{PREFIX_CHECKPOINT_DIR}-{max_step}')
    print(f"🪙 Found a previous checkpoint at: {checkpoint_dir}")
    return checkpoint_dir, is_completed # checkpoint found!

  return None, False # first training



def save_model_struct(
  model,
  path = None,
  model_name = None
):
  """
    디버깅을 위해 모델의 구조를 파일로 저장합니다.
  """
  if path is None:
    if model_name is None:
      path = model.__class__.__name__ + '.txt'
    else:
      path = model_name + '.txt'
    
  with open(path, 'w') as f:
    f.write(model.__str__())

    f.write('\n==========\n')
    state = model.state_dict()
    for key in state:
      f.write(f'{key}: {state[key].shape}\n')
    
    # f.write('\n==========\n')
    # for  name, module in model.named_modules():
    #   f.write(f'name:{name}\nmodule:{module}\n\n')



def get_preprocessed_file_path(dataset_path: str, tokenizer_name_or_path: str, max_length: int) -> str:
  """
    전처리된 파일의 경로를 반환합니다.
  """
  tokenizer_name = ''
  if '/' in tokenizer_name_or_path:
    tokenizer_file_name_with_extension = os.path.basename(tokenizer_name_or_path)
    tokenizer_name = os.path.splitext(tokenizer_file_name_with_extension)[0]  # 확장자를 제거한 파일 이름

  max_length = max_length

  ext = dataset_path.split('.')[-1] if '.' in dataset_path else ''
  origin_file_path_wo_ext = dataset_path.replace(f'.{ext}', '') if ext else dataset_path
  preprocessed_file_path = f"{origin_file_path_wo_ext}_name_{tokenizer_name}_cut_{max_length}.{ext}"

  return preprocessed_file_path



def print_trainable_parameters(args, model):
    """
    훈련 가능한 파라미터 수를 출력합니다.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4: trainable_params /= 2
    print(
        f"📰 trainable params: {trainable_params:,} / "
        f"all params: {all_param:,} / "
        f"trainable: {(trainable_params / all_param):.4%}%"
    )



def print_model_named_parameters(model):
  """
    모델의 named_parameters를 출력합니다.
  """
  dtypes = {}
  for _, p in model.named_parameters():
    dtype = p.dtype
    if dtype not in dtypes: dtypes[dtype] = 0
    dtypes[dtype] += p.numel()
  total = 0
  for k, v in dtypes.items(): total+= v
  for k, v in dtypes.items():
    print(f'📰 Model named_parameters (dtype: {k} / count: {v:,} / percentage: {v/total:.4%} )')



class SavePeftModelCallback(transformers.TrainerCallback):
  """
    transformers.Trainer 용 PEFT 모델 저장 콜백
  """
  def save_model(self, args, state, kwargs):
    print('🌀 Saving PEFT checkpoint...')
    if state.best_model_checkpoint is not None:
      checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
    else:
      checkpoint_folder = os.path.join(
        args.output_dir,
        f"{transformers.trainer_utils.PREFIX_CHECKPOINT_DIR}-{state.global_step}"
      )

    peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
    kwargs["model"].save_pretrained(peft_model_path)

    pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
    if os.path.exists(pytorch_model_path):
      os.remove(pytorch_model_path)

  def on_save(self, args, state, control, **kwargs):
    self.save_model(args, state, kwargs)
    return control

  def on_train_end(self, args, state, control, **kwargs):
    def touch(fname, times=None):
      with open(fname, 'a'):
        os.utime(fname, times)

    touch(os.path.join(args.output_dir, 'completed'))
    self.save_model(args, state, kwargs)



def command(
  cmd: list[str]
) -> tuple[bytes, bytes]:
  """
    명령을 실행하고 (stdoutBytes, stderrBytes)를 반환합니다.
  """
  process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  output, err = process.communicate()
  if output != b'' or err != b'':
    print("debug stdout:", str(output))
    print("debug command stderr:", str(err))
  return output, err



def run_java_to_generate_input(
  run_type: str,
  java_project_path: str,
  buggy_file: str,
  rem_start: int,
  rem_end: int,
  tmp_file: str,
  config: dict|None = None,
  fixed_file: str|None = None,
  add_start: int|None = None,
  add_end: int|None = None
):
  """
    Java 프로젝트를 실행해 모델평가에 사용할 입력을 생성해 tmp_file로 저장

    fixed_file, add_start, add_end가 제공되면 finetuning 모드로 실행하여 fixed line도 생성
  """
  os.chdir(java_project_path)

  commandArgs = [
    'java', '-cp', '.:target:lib/*'
  ]
  if run_type == 'finetune':
    commandArgs.append('clm.finetuning.FineTuningData')
    # fixed 파일 정보가 있으면 finetuning 모드, 없으면 inference 모드
    if fixed_file and add_start is not None and add_end is not None:
      commandArgs.append('finetuning')
    else:
      commandArgs.append('inference')
  elif run_type == 'codegen':
    commandArgs.append('clm.codegen.CodeGenInputParser')
  else:
    raise ValueError('unrecognized run_type')

  if run_type == 'finetune':
    if fixed_file and add_start is not None and add_end is not None:
      # finetuning 모드: 8개 인자
      commandArgs += [buggy_file, str(rem_start), str(rem_end), fixed_file, str(add_start), str(add_end), tmp_file]
    else:
      # inference 모드: 5개 인자
      commandArgs += [buggy_file, str(rem_start), str(rem_end), tmp_file]
  elif run_type == 'codegen':
    commandArgs += [buggy_file, str(rem_start), str(rem_end), config, tmp_file]

  command(commandArgs)



def raw_output_to_patch(
  output_text: str,
  config_name: str
) -> str:
  if config_name in ['CODEGEN_COMPLETE_CODEFORM_NOCOMMENT', 'CODEGEN_COMPLETE_CODEFORM_COMMENTFORM_NOCOMMENT']:
    """
    파인튜닝 되지 않은 CausalLM 모델의 출력 처리 함수

    첫 "{" 와 매치되는  "}" 찾아서(함수의 시작과 끝을 찾기 위해) 리턴
    """
    output_lines = output_text.strip().split('\n')
    no_comment_output_lines = [line for line in output_lines if not line.strip().startswith('//')]
    no_comment_output = '\n'.join(no_comment_output_lines)

    stack = ['{']
    try:
      start_index = no_comment_output.index('{')
      patch = no_comment_output[: start_index + 1]
      for char in no_comment_output[start_index + 1: ]:
        patch += char
        if char == '}':
          top = stack.pop()
          if top != '{':
            return ''
          if len(stack) == 0:
            return patch.strip()
        elif char == '{':
            stack.append(char)
      return ''
    except Exception as e:
      return ''
  else:
    return ''



def ft_output_to_patch(
  output: str,
  eos: str,
) -> str:
  """
    파인튜닝된 CausalLM 모델의 출력 처리 함수
    
    "fixed lines:" 를 찾아서 EOS 토큰 전까지 리턴
  """
  start_index = 0
  if '// fixed lines: \n' in output:
    start_index = output.index('// fixed lines: \n') + len('// fixed lines: \n')
  output = output[start_index: ]
  end_index = len(output)
  if eos in output:
    end_index = output.index(eos)
  output = output[: end_index]
  return output



def insert_fix(filename, start_line, end_line, patch):
  """
  end_row is included in the buggy lines / buggy function
  """
  with open(filename, 'r') as file:
    data = file.readlines()

  with open(filename, 'w') as file:
    for i in range(start_line - 1):
      file.write(data[i])
    file.write(patch.strip() + '\n')
    for i in range(end_line, len(data)):
      file.write(data[i])



def set_seed(seed: int, deterministic: bool = False):
  """
  Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

  Args:
      seed (`int`):
          The seed to set.
      deterministic (`bool`, *optional*, defaults to `False`):
          Whether to use deterministic algorithms where available. Can slow down training.
  """
  random.seed(seed)
  numpy.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  # ^^ safe to call this function even if cuda is not available
  if deterministic:
    torch.use_deterministic_algorithms(True)
  # if is_torch_mlu_available():
  #   torch.mlu.manual_seed_all(seed)
  # if is_torch_npu_available():
  #   torch.npu.manual_seed_all(seed)
  # if is_torch_xpu_available():
  #   torch.xpu.manual_seed_all(seed)