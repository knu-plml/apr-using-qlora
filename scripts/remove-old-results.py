#!/usr/bin/env python3
"""
오래된 결과 파일들을 제거하는 유틸리티 스크립트
"""
import os
import glob
from pathlib import Path

# 검색할 기준 경로 (필요시 수정)
BASE_PATH = "/home/yglee/wlm"

# 검색할 파일 패턴
PATTERNS = [
    "*_input.json",
    "*_validate.json",
    "*_validate_v12.json",
    "*_validate_v20.json"
]

def find_matching_files(base_path, patterns):
    """지정된 패턴과 매칭되는 모든 파일을 찾습니다."""
    matching_files = []

    for pattern in patterns:
        # recursive=True로 하위 디렉토리까지 모두 검색
        search_pattern = os.path.join(base_path, "**", pattern)
        files = glob.glob(search_pattern, recursive=True)
        matching_files.extend(files)

    # 중복 제거 및 정렬
    matching_files = sorted(set(matching_files))
    return matching_files

def main():
    print(f"검색 경로: {BASE_PATH}")
    print(f"검색 패턴: {', '.join(PATTERNS)}")
    print("-" * 80)

    # 매칭되는 파일 찾기
    files_to_delete = find_matching_files(BASE_PATH, PATTERNS)

    if not files_to_delete:
        print("매칭되는 파일을 찾을 수 없습니다.")
        return

    # 찾은 파일 목록 출력
    print(f"\n총 {len(files_to_delete)}개의 파일을 찾았습니다:\n")
    for file_path in files_to_delete:
        print(file_path)

    # 사용자 확인
    print("\n" + "-" * 80)
    response = input(f"\n위 {len(files_to_delete)}개의 파일을 모두 삭제하시겠습니까? (Y/N): ").strip().upper()

    if response == 'Y':
        deleted_count = 0
        failed_count = 0

        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                deleted_count += 1
                print(f"삭제됨: {file_path}")
            except Exception as e:
                failed_count += 1
                print(f"삭제 실패: {file_path} - {e}")

        print("\n" + "=" * 80)
        print(f"완료: {deleted_count}개 파일 삭제됨, {failed_count}개 실패")
    else:
        print("취소되었습니다. 파일이 삭제되지 않았습니다.")

if __name__ == "__main__":
    main()
