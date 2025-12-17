#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import os
import sys

# 이 파일(run_all_experiments.py)의 위치 기준으로 경로를 계산
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run(script_name: str, extra_args):
    """
    script_name : 같은 디렉터리에 있는 파이썬 파일 이름 (예: 'train_one_head.py')
    extra_args  : 그 뒤에 붙일 인자 리스트
    """
    script_path = os.path.join(BASE_DIR, script_name)

    cmd_list = [sys.executable, script_path] + extra_args

    print("\n========================================")
    print("실행:", " ".join(cmd_list))
    print("========================================\n")

    # 부모 프로세스의 환경(PYTHONPATH=/workspace 등)을 그대로 상속받음
    subprocess.run(cmd_list, check=True)


def main():
    # -------- 경로/모델 설정 --------
    train_path = "/workspace/transformer/out_kobert_space_particle.jsonl"
    valid_path = "None"   # 별도 valid 파일 없으면 "None"
    model_name = "monologg/kobert"

    # outdir는 실험별로 분리
    out_case5 = os.path.join(BASE_DIR, "runs_case5_s2s_kobert")
    out_case1 = os.path.join(BASE_DIR, "runs_case1_token_only")
    out_case2 = os.path.join(BASE_DIR, "runs_case2_token_space")
    out_case3 = os.path.join(BASE_DIR, "runs_case3_token_particle")
    out_case4 = os.path.join(BASE_DIR, "runs_case4_all_heads")

    os.makedirs(out_case5, exist_ok=True)
    os.makedirs(out_case1, exist_ok=True)
    os.makedirs(out_case2, exist_ok=True)
    os.makedirs(out_case3, exist_ok=True)
    os.makedirs(out_case4, exist_ok=True)
    
    # CASE 5: KoBERT Seq2Seq (train_s2s_kobert_case5.py)
    run(
        "train_s2s_kobert_case5.py",
        [
            "--train", train_path,
            "--valid", valid_path,
            "--outdir", out_case5,
            "--epochs", "20",
            "--batch_size", "32",
        ],
    )

    # 1) CASE 1: train_one_head.py (token_labels만 학습)
    run(
        "train_one_head.py",
        [
            "--train", train_path,
            "--valid", valid_path,
            "--model", model_name,
            "--outdir", out_case1,
            "--epochs", "20"
        ],
    )

    # 2) CASE 2: train_multi_head_gec.py (token + space)
    run(
        "train_multi_head_gec.py",
        [
            "--train", train_path,
            "--valid", valid_path,
            "--model", model_name,
            "--outdir", out_case2,
            "--use_space",
            "--lambda_space", "1.0",
            "--epochs", "20"
        ],
    )

    # 3) CASE 3: token + particle
    run(
        "train_multi_head_gec.py",
        [
            "--train", train_path,
            "--valid", valid_path,
            "--model", model_name,
            "--outdir", out_case3,
            "--use_particle",
            "--lambda_particle", "1.0",
            "--epochs", "20"
        ],
    )

    # 4) CASE 4: token + space + particle(lambda 0.1)
    run(
        "train_multi_head_gec.py",
        [
            "--train", train_path,
            "--valid", valid_path,
            "--model", model_name,
            "--outdir", out_case4,
            "--use_space",
            "--lambda_space", "1.0",
            "--use_particle",
            "--lambda_particle", "0.1",
            "--epochs", "20"
        ],
    )

    print("\n모든 실험이 순차적으로 종료되었습니다.")


if __name__ == "__main__":
    main()
