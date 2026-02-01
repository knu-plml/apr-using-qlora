#!/usr/bin/env python3
"""
현재 사용자 UID로 실행 중인 모든 Java 프로세스를 SIGKILL(9)로 종료하는 스크립트
"""
import os
import subprocess
import signal

def get_my_java_processes():
    """현재 사용자의 Java 프로세스 목록을 가져옵니다."""
    my_uid = os.getuid()

    try:
        # 현재 사용자의 모든 java 프로세스 PID 찾기
        result = subprocess.run(
            ['pgrep', '-u', str(my_uid), 'java'],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            pids = [int(pid) for pid in result.stdout.strip().split('\n') if pid]
            return pids
        else:
            return []
    except Exception as e:
        print(f"프로세스 검색 중 오류 발생: {e}")
        return []

def get_process_info(pid):
    """프로세스 정보를 가져옵니다."""
    try:
        result = subprocess.run(
            ['ps', '-p', str(pid), '-o', 'pid,ppid,etime,cmd'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                return lines[1]
        return None
    except:
        return None

def kill_process(pid):
    """프로세스에 SIGKILL(9)을 전송합니다."""
    try:
        os.kill(pid, signal.SIGKILL)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return False
    except Exception as e:
        print(f"PID {pid} 종료 실패: {e}")
        return False

def main():
    print("=" * 80)
    print("현재 사용자의 Java 프로세스 종료 스크립트")
    print("=" * 80)

    my_uid = os.getuid()
    print(f"현재 UID: {my_uid}\n")

    # Java 프로세스 찾기
    java_pids = get_my_java_processes()

    if not java_pids:
        print("실행 중인 Java 프로세스가 없습니다.")
        return

    print(f"총 {len(java_pids)}개의 Java 프로세스를 찾았습니다:\n")
    print(f"{'PID':<8} {'PPID':<8} {'ELAPSED':<12} COMMAND")
    print("-" * 80)

    # 프로세스 정보 출력
    for pid in java_pids:
        info = get_process_info(pid)
        if info:
            print(info)

    print("\n" + "=" * 80)

    # 프로세스 종료
    killed_count = 0
    failed_count = 0

    for pid in java_pids:
        if kill_process(pid):
            print(f"✓ PID {pid} 종료됨 (SIGKILL)")
            killed_count += 1
        else:
            print(f"✗ PID {pid} 종료 실패 (이미 종료되었거나 권한 없음)")
            failed_count += 1

    print("\n" + "=" * 80)
    print(f"완료: {killed_count}개 프로세스 종료됨, {failed_count}개 실패")

if __name__ == "__main__":
    main()
