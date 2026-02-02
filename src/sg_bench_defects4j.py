import json
import os
import time
import subprocess

import transformers
import shutil

import sg_tools
import defects4j_command

def generate_defects4j_single_input(
  bench_tmp_path: str,
  java_project_path: str,
  line: str
) -> None | dict:
  print(f'🔍 Loading line... ({line.strip()})')
  proj, bug_id, path, rem_loc, add_loc = line.strip().split()

  defects4j_command.command_with_timeout(['mkdir', '-p', bench_tmp_path])

  # if path start with '/': remove it
  if path[0] == '/':
    path = path[1:]

  start, end = rem_loc.split('-')
  end = str(int(end) - 1) if end != start else end
  add_start, add_end = add_loc.split('-')
  add_end = str(int(add_end) - 1) if add_end != add_start else add_end
  tmp_file = os.path.join(bench_tmp_path, 'tmp.json')
  pkey = proj + '_' + bug_id + '_' + path + '_' + rem_loc

  # Defects4J 프로젝트를 체크아웃 (buggy 버전)
  buggy_tmp_path = bench_tmp_path + '_buggy'
  fixed_tmp_path = bench_tmp_path + '_fixed'
  defects4j_command.command_with_timeout(['mkdir', '-p', buggy_tmp_path])
  defects4j_command.command_with_timeout(['mkdir', '-p', fixed_tmp_path])
  subprocess.run(['defects4j', 'checkout', '-p', proj, '-v', bug_id + 'b', '-w', buggy_tmp_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  # Defects4J 프로젝트를 체크아웃 (fixed 버전)
  subprocess.run(['defects4j', 'checkout', '-p', proj, '-v', bug_id + 'f', '-w', fixed_tmp_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

  # Jasper Java 프로젝트를 통해 입력 생성
  sg_tools.run_java_to_generate_input(
    run_type = 'finetune',
    java_project_path = java_project_path,
    buggy_file = os.path.join(buggy_tmp_path, path),
    rem_start = start,
    rem_end = end,
    tmp_file = tmp_file,
    config = None,
    fixed_file = os.path.join(fixed_tmp_path, path),
    add_start = add_start,
    add_end = add_end
  )

  if not os.path.exists(tmp_file):
    print('❌', proj, bug_id, 'failed.', tmp_file, 'not found.')
    return None
  result = json.load(open(tmp_file, 'r'))
  if result["buggy function before"].strip() == '' and result["buggy line"].strip() == '' and result["buggy function after"].strip() == '':
    print('❌', proj, bug_id, 'failed. all empty.')
    return None
  return_obj = {
    'id': pkey,
    'loc': rem_loc,
    'input': result['buggy function before'] +
      '// buggy lines start:\n' + result['buggy line'] +
      '// buggy lines end:\n' + result['buggy function after'] +
      '// fixed lines: \n',
    'fixed_line': result['fixed line'],
  }
  print('✅', proj, bug_id, 'succeeded')

  sg_tools.command(['rm', '-rf', tmp_file])
  sg_tools.command(['rm', '-rf', buggy_tmp_path])
  sg_tools.command(['rm', '-rf', fixed_tmp_path])
  return return_obj



def validate_defects4j(
  model_name: str,
  tokenizer: transformers.PreTrainedTokenizer,
  input_file: str,
  output_file: str,
  tmp_dir: str
):
  # INJECTED: debug name and EOS token
  model_name = model_name.lower()

  EOS_STR = None
  if 'incoder' in model_name:
    print('🧂 incoder model detected. add EOS token (<|endofmask|>)')
    EOS_STR = '<|endofmask|>'
  else:
    EOS_STR = tokenizer.eos_token
  # INJECTED END

  plausible, total = 0, 0

  if not os.path.exists(tmp_dir):
    defects4j_command.command_with_timeout(['mkdir', tmp_dir])

  model_output = json.load(open(input_file, 'r'))
  validated_result = {'config': model_output['config'], 'data': {}}
  # validated_result = json.load(open(output_file, 'r'))
  for key in model_output['data']:
    if key in validated_result['data']:
      continue
    if 'output' not in model_output['data'][key]:
      continue

    key_list = key.split('_')
    proj, bug_id, loc = key_list[0], key_list[1], key_list[-1]
    path = '_'.join(key_list[2: -1])

    print('start validating', proj, bug_id)
    total += 1
    
    validated_result['data'][key] = {}
    for k, value in model_output['data'][key].items():
      if k != 'output':
        validated_result['data'][key][k] = value
    validated_result['data'][key]['output'] = []
    start_line, end_line = validated_result['data'][key]['loc'].split('-')
    end_line = str(int(end_line) - 1) if end_line != start_line else end_line

    defects4j_command.clean_tmp_folder(tmp_dir)
    defects4j_command.checkout_defects4j_project(proj, bug_id + 'b', tmp_dir)
    if proj == "Mockito":
      print("Mockito needs separate compilation")
      defects4j_command.compile_fix(tmp_dir)

    # check standard test time
    start_time = time.time()
    init_out, init_err = defects4j_command.defects4j_test_suite(tmp_dir)
    standard_time = int(time.time() - start_time)

    # check failed test cases
    failed_test_cases = str(init_out).split(' - ')[1:]
    for i, failed_test_case in enumerate(failed_test_cases):
      failed_test_cases[i] = failed_test_case.strip()
    init_fail_num = len(failed_test_cases)
    print(init_fail_num, str(standard_time) + 's')

    # check triggering failed test cases
    trigger, err = defects4j_command.defects4j_trigger(tmp_dir)
    triggers = trigger.strip().split('\n')
    for i, trigger in enumerate(triggers):
      triggers[i] = trigger.strip()
    print('trigger number:', len(triggers))

    current_is_correct = False
    for rank, patch in enumerate(model_output['data'][key]['output']):
      filename = os.path.join(tmp_dir, path)
      shutil.copyfile(filename, filename + '.bak')

      # INJECT: 통합 patch 추출 및 insertion
      patch = sg_tools.ft_output_to_patch(patch, EOS_STR)
      sg_tools.insert_fix(filename, int(start_line), int(end_line), patch)
      # INJECT END

      if proj == 'Mockito':
        # Mockito needs seperate compile
        defects4j_command.compile_fix(tmp_dir)

      # trigger cases is few and total time is long, we test trigger cases first.
      outs = []
      correctness = None
      start_time = time.time()
      if standard_time >= 10 and len(triggers) <= 5:
        for trigger in triggers:
          out, err = defects4j_command.defects4j_test_one(tmp_dir, trigger, timeout=min(300, int(2*standard_time)))
          if 'TIMEOUT' in str(err) or 'TIMEOUT' in str(out):
            print(plausible, total, rank, 'Time out for patch: ', patch,
              str(int(time.time() - start_time)) + 's')
            correctness = 'timeout'
            break
          elif 'FAIL' in str(err) or 'FAIL' in str(out):
            print(plausible, total, rank, 'Uncompilable patch:', patch,
              str(int(time.time() - start_time)) + 's')
            correctness = 'uncompilable'
            break
          elif "Failing tests: 0" in str(out):
            continue
          else:
            outs += str(out).split(' - ')[1:]
      if len(set(outs)) >= len(triggers):
        # does not pass any one more
        print(plausible, total, rank, 'Wrong patch:', patch,
          str(int(time.time() - start_time)) + 's')
        correctness = 'wrong'

      if correctness is None:
        # pass at least one more trigger case
        # have to pass all non-trigger
        out, err = defects4j_command.defects4j_test_suite(tmp_dir, timeout=min(300, int(2*standard_time)))
        msg_concat = str(out) + str(err)

        if 'TIMEOUT' in str(err) or 'TIMEOUT' in str(out):
          print(plausible, total, rank, 'Time out for patch: ', patch,
            str(int(time.time() - start_time)) + 's')
          correctness = 'timeout'
        elif 'FAIL' in str(err) or 'FAIL' in str(out):
          print(plausible, total, rank, 'Uncompilable patch:', patch,
            str(int(time.time() - start_time)) + 's')
          correctness = 'uncompilable'
        elif "Failing tests: 0" in str(out):
          if not current_is_correct:
            current_is_correct = True
            plausible += 1
          print(plausible, total, rank, 'Plausible patch:', patch,
            str(int(time.time() - start_time)) + 's')
          correctness = 'plausible'
        elif len(str(out).split(' - ')[1:]) < init_fail_num:
          # fail less, could be correct
          current_failed_test_cases = str(out).split(' - ')[1:]
          no_new_fail = True
          for current_failed_test_case in current_failed_test_cases:
            if current_failed_test_case.strip() not in failed_test_cases:
              no_new_fail = False
              break
          if no_new_fail:
            # fail less and no new fail cases, could be plausible
            if not current_is_correct:
              current_is_correct = True
              plausible += 1
            print(plausible, total, rank, 'Plausible patch:', patch,
                str(int(time.time() - start_time)) + 's')
            correctness = 'plausible'
          else:
            print(plausible, total, rank, 'Wrong patch:', patch,
                str(int(time.time() - start_time)) + 's')
            correctness = 'wrong'
        else:
          print(plausible, total, rank, 'Wrong patch:', patch,
            str(int(time.time() - start_time)) + 's')
          correctness = 'wrong'

      validated_result['data'][key]['output'].append({
        'patch': patch, 'correctness': correctness,
        'raw_output': msg_concat
      })
      shutil.copyfile(filename + '.bak', filename)

    # write after finish validating every bug, to avoid wasting time
    validated_result['result'] = {
      'plausible': plausible,
      'total': total
    }
    json.dump(validated_result, open(output_file, 'w'), indent=2)

  # write the last time after validating all
  validated_result['result'] = {
    'plausible': plausible,
    'total': total
  }
  json.dump(validated_result, open(output_file, 'w'), indent=2)



# 원문 코드는 에러를 줄이기만 했으면 plausible로 판단
# 이 코드는 모든 테스트를 통과해야 plausible로 판단
def strict_validate_defects4j(
  model_name: str,
  tokenizer: transformers.PreTrainedTokenizer,
  input_file: str,
  output_file: str,
  tmp_dir: str
):
  # INJECTED: debug name and EOS token
  model_name = model_name.lower()

  EOS_STR = None
  if 'incoder' in model_name:
    print('🧂 incoder model detected. add EOS token (<|endofmask|>)')
    EOS_STR = '<|endofmask|>'
  else:
    EOS_STR = tokenizer.eos_token
  # INJECTED END

  plausible, total = 0, 0

  if not os.path.exists(tmp_dir):
    defects4j_command.command_with_timeout(['mkdir', tmp_dir])

  model_output = json.load(open(input_file, 'r'))
  validated_result = {'config': model_output['config'], 'data': {}}
  # validated_result = json.load(open(output_file, 'r'))
  for key in model_output['data']:
    if key in validated_result['data']:
      continue
    if 'output' not in model_output['data'][key]:
      continue

    key_list = key.split('_')
    proj, bug_id, loc = key_list[0], key_list[1], key_list[-1]
    path = '_'.join(key_list[2: -1])

    print('start validating', proj, bug_id)
    total += 1
    
    validated_result['data'][key] = {}
    for k, value in model_output['data'][key].items():
      if k != 'output':
        validated_result['data'][key][k] = value
    validated_result['data'][key]['output'] = []
    start_line, end_line = validated_result['data'][key]['loc'].split('-')
    end_line = str(int(end_line) - 1) if end_line != start_line else end_line

    defects4j_command.clean_tmp_folder(tmp_dir)
    defects4j_command.checkout_defects4j_project(proj, bug_id + 'b', tmp_dir)
    if proj == "Mockito":
      print("Mockito needs separate compilation")
      defects4j_command.compile_fix(tmp_dir)

    # check standard test time
    start_time = time.time()
    init_out, init_err = defects4j_command.defects4j_test_suite(tmp_dir)
    standard_time = int(time.time() - start_time)

    # check failed test cases
    failed_test_cases = str(init_out).split(' - ')[1:]
    for i, failed_test_case in enumerate(failed_test_cases):
      failed_test_cases[i] = failed_test_case.strip()
    init_fail_num = len(failed_test_cases)
    print(init_fail_num, str(standard_time) + 's')

    # check triggering failed test cases
    trigger, err = defects4j_command.defects4j_trigger(tmp_dir)
    triggers = trigger.strip().split('\n')
    for i, trigger in enumerate(triggers):
      triggers[i] = trigger.strip()
    print('trigger number:', len(triggers))

    current_is_correct = False
    for rank, patch in enumerate(model_output['data'][key]['output']):
      filename = os.path.join(tmp_dir, path)
      shutil.copyfile(filename, filename + '.bak')

      # INJECT: 통합 patch 추출 및 insertion
      patch = sg_tools.ft_output_to_patch(patch, EOS_STR)
      sg_tools.insert_fix(filename, int(start_line), int(end_line), patch)
      # INJECT END

      if proj == 'Mockito':
        # Mockito needs seperate compile
        defects4j_command.compile_fix(tmp_dir)

      # trigger cases is few and total time is long, we test trigger cases first.
      outs = []
      correctness = None
      msg_concat = ''
      start_time = time.time()
      if standard_time >= 10 and len(triggers) <= 5:
        for trigger in triggers:
          out, err = defects4j_command.defects4j_test_one(tmp_dir, trigger, timeout=min(300, int(2*standard_time)))
          if 'TIMEOUT' in str(err) or 'TIMEOUT' in str(out):
            print(plausible, total, rank, 'Time out for patch: ', patch,
              str(int(time.time() - start_time)) + 's')
            correctness = 'timeout'
            break
          elif 'FAIL' in str(err) or 'FAIL' in str(out):
            print(plausible, total, rank, 'Uncompilable patch:', patch,
              str(int(time.time() - start_time)) + 's')
            correctness = 'uncompilable'
            break
          elif "Failing tests: 0" in str(out):
            continue
          else:
            outs += str(out).split(' - ')[1:]
      if len(set(outs)) >= len(triggers):
        # does not pass any one more
        print(plausible, total, rank, 'Wrong patch:', patch,
          str(int(time.time() - start_time)) + 's')
        correctness = 'wrong'

      if correctness is None:
        # pass at least one more trigger case
        # have to pass all non-trigger
        out, err = defects4j_command.defects4j_test_suite(tmp_dir, timeout=min(300, int(2*standard_time)))
        msg_concat = str(out) + str(err)

        if 'TIMEOUT' in str(err) or 'TIMEOUT' in str(out):
          print(plausible, total, rank, 'Time out for patch: ', patch,
            str(int(time.time() - start_time)) + 's')
          correctness = 'timeout'
        elif 'FAIL' in str(err) or 'FAIL' in str(out):
          print(plausible, total, rank, 'Uncompilable patch:', patch,
            str(int(time.time() - start_time)) + 's')
          correctness = 'uncompilable'
        elif "Failing tests: 0" in str(out):
          if not current_is_correct:
            current_is_correct = True
            plausible += 1
          print(plausible, total, rank, 'Plausible patch:', patch,
            str(int(time.time() - start_time)) + 's')
          correctness = 'plausible'
        else:
          print(plausible, total, rank, 'Wrong patch:', patch,
            str(int(time.time() - start_time)) + 's')
          correctness = 'wrong'

      validated_result['data'][key]['output'].append({
        'patch': patch, 'correctness': correctness,
        'raw_output': msg_concat
      })
      shutil.copyfile(filename + '.bak', filename)

    # write after finish validating every bug, to avoid wasting time
    validated_result['result'] = {
      'plausible': plausible,
      'total': total
    }
    json.dump(validated_result, open(output_file, 'w'), indent=2)

  # write the last time after validating all
  validated_result['result'] = {
    'plausible': plausible,
    'total': total
  }
  json.dump(validated_result, open(output_file, 'w'), indent=2)


def result_v12_v20_splitter(
  input_file: str,
  v12_output_file: str,
  v20_output_file: str
):
  V12_DATA_WITH_MAX_INDEX = {
    'Chart': 26,
    'Closure': 133,
    'Lang': 65,
    'Math': 106,
    'Mockito': 38,
    'Time': 27
  }

  input_data = json.load(open(input_file, 'r'))
  v12_output_data = {'config': input_data['config'], 'data': {}, 'result': {
    'plausible': 0,
    'total': 0
  }}
  v20_output_data = {'config': input_data['config'], 'data': {}, 'result': {
    'plausible': 0,
    'total': 0
  }}

  for key in input_data['data']:
    key_list = key.split('_')
    if len(key_list) < 2:
      print(f'💥 Unexpected defects4j validation key: {key}')
      exit(1)

    proj = key_list[0]
    v12_proj_max_index = V12_DATA_WITH_MAX_INDEX.get(proj)
    target_data = input_data['data'][key]

    is_plausible = False
    for patch in target_data['output']:
      if patch['correctness'] == 'plausible':
        is_plausible = True
        break


    if (v12_proj_max_index is not None) and int(key_list[1]) <= v12_proj_max_index:
      v12_output_data['data'][key] = input_data['data'][key]
      v12_output_data['result']['total'] += 1
      if is_plausible:
        v12_output_data['result']['plausible'] += 1
    else:
      v20_output_data['data'][key] = input_data['data'][key]
      v20_output_data['result']['total'] += 1
      if is_plausible:
        v20_output_data['result']['plausible'] += 1

  json.dump(v12_output_data, open(v12_output_file, 'w'), indent=2)
  json.dump(v20_output_data, open(v20_output_file, 'w'), indent=2)
