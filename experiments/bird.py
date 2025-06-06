import datetime
import os
from typing import Any
from pathlib import Path
import sys
import subprocess

import yaml

MAIN_CONDA_ENV = "xfc"

HOME = os.getenv("HOME")
EVAL_ROOT = f"{HOME}/dev/github/bird"
SERVER_ROOT = f"{HOME}/dev/github/milkie"
REPORT_ROOT = f"{HOME}/dev/github/bird/reports"

EVAL_CONFIG_PATH = f"{EVAL_ROOT}/mini_dev/llm/run/run_gpt.sh"
SERVER_CONFIG_PATH = f"{SERVER_ROOT}/clients/bird/bird.yaml"

def backup_config(config_path: str, report_dir: str):
    if not Path(config_path).exists():
        sys.exit(f"错误：配置文件不存在: {config_path}")

    Path(f"{report_dir}/bak").mkdir(parents=True, exist_ok=True)
    status = os.system(f"cp {config_path} {report_dir}/bak/")
    if status != 0:
        sys.exit(f"错误：备份配置文件失败: {config_path}")

def set_num_cases(config :dict, num_cases :int):
    """设置测试用例数量
    Args:
        num_cases: 要设置的测试用例数量
    """
    set_config(
        config=config, 
        config_path=EVAL_CONFIG_PATH, 
        config_key="num_cases", 
        value=num_cases, 
        sep="=")

def set_model(config :dict, model :str):
    """设置模型
    Args:
        model: 要设置的模型名称
    """
    set_config(
        config=config, 
        config_path=SERVER_CONFIG_PATH, 
        config_key="thought_model", 
        value=model, 
        sep=":")
    set_config(
        config=config, 
        config_path=SERVER_CONFIG_PATH, 
        config_key="sql_model", 
        value=model, 
        sep=":")

def set_table_desc_record_samples(config :dict, num_samples :int):
    """设置表描述记录样本数量
    Args:
        num_cases: 要设置的表描述记录样本数量
    """
    set_config(
        config=config, 
        config_path=SERVER_CONFIG_PATH, 
        config_key="table_desc_record_samples", 
        value=num_samples, 
        sep=":")

def set_table_fields_record_samples(config :dict, num_samples :int):
    """设置表字段记录样本数量
    Args:
        num_samples: 要设置的表字段记录样本数量
    """
    set_config(
        config=config, 
        config_path=SERVER_CONFIG_PATH, 
        config_key="table_fields_record_samples", 
        value=num_samples, 
        sep=":")

def set_config(
        config: dict,   
        config_path: str, 
        config_key: str, 
        value: Any, 
        sep: str):
    """设置配置文件中的键值对
    Args:
        config: 配置字典
        config_key: 要设置的配置键名
        value: 要设置的值
        sep: 键值分隔符，默认为空格
    """
    config[config_key] = value
    if not Path(config_path).exists():
        sys.exit(f"错误：YAML 配置文件不存在: {config_path}")
        
    value_str = f'"{value}"' if isinstance(value, str) else str(value)
    extra_whitespace = " " if config_path.endswith("yaml") else ""
    status = os.system(f"sed -i 's/^\\([[:space:]]*\\){config_key}[[:space:]]*{sep}.*$/\\1{config_key}{sep}{extra_whitespace}{value_str}/' {config_path}")
    if status != 0:
        sys.exit(f"错误：更新配置文件失败: {config_path}")

def apply_config(config :dict):
    set_num_cases(config, config["num_cases"])
    set_table_desc_record_samples(config, config["table_desc_record_samples"])
    set_table_fields_record_samples(config, config["table_fields_record_samples"])
    set_model(config, config["model"])

def backup_configs(report_dir :str):
    backup_config(SERVER_CONFIG_PATH, report_dir)
    backup_config(EVAL_CONFIG_PATH, report_dir)

def restart_bird():
    print("重启BIRD")

    cmd = f"""
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate {MAIN_CONDA_ENV}
    cd {SERVER_ROOT}
    ./clients/bird/bin/bird.sh restart
    """
    subprocess.run(cmd, shell=True, executable='/bin/bash')

def start_eval(report_dir: str):
    print(f"开始评估，报告目录: {report_dir}")
    
    cmd = f"""
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate BIRD
    cd {EVAL_ROOT}
    bash mini_dev/evaluation/eval.sh --run 
    bash mini_dev/evaluation/eval.sh &> {report_dir}/eval.log
    """
    subprocess.run(cmd, shell=True, executable='/bin/bash')

def make_report(config: dict, report_dir: str):
    print(f"生成报告，报告目录: {report_dir}")

    file_basic_info = f"{report_dir}/basic_info.txt"
    try:
        with open(file_basic_info, "w") as f:
            f.write(yaml.dump(config))
    except Exception as e:
        sys.exit(f"错误：写入基础信息失败: {str(e)}")
    
    # 检查必要文件是否存在
    required_files = [EVAL_CONFIG_PATH, SERVER_CONFIG_PATH]
    for file_path in required_files:
        if not Path(file_path).exists():
            sys.exit(f"错误：必要文件不存在: {file_path}")
    
    commands = [
        f"cp {EVAL_CONFIG_PATH} {report_dir}/eval.sh",
        f"cp {SERVER_CONFIG_PATH} {report_dir}/bird.yaml",
        f"cd {SERVER_ROOT}; cp clients/bird/log/bird.log {report_dir}/bird.log",
        f"cd {SERVER_ROOT}; cp nohup.out {report_dir}/server.log"
    ]
    
    for cmd in commands:
        status = os.system(cmd)
        if status != 0:
            sys.exit(f"错误：执行命令失败: {cmd}")

def set_configs() -> list[dict]:
    num_cases = [500]
    max_thoughts = [2]
    max_sqls = [2]
    table_desc_record_samples = [3]
    table_fields_record_samples = [3]
    models = ["deepseek-chat", "Tome-max"]
    configs = []
    for num_case in num_cases:
        for max_thought in max_thoughts:
            for max_sql in max_sqls:
                for table_desc_record_sample in table_desc_record_samples:
                    for table_fields_record_sample in table_fields_record_samples:
                        for model in models:
                            configs.append({
                                "num_cases": num_case,
                                "max_thoughts" : max_thought,
                                "max_sqls" : max_sql,
                                "table_desc_record_samples": table_desc_record_sample,
                                "table_fields_record_samples": table_fields_record_sample,
                                "model": model
                            })
    return configs

def run_eval(configs: list[dict]):
    if not configs:
        sys.exit("错误：配置列表为空")
        
    for config in configs:
        required_keys = [
            "num_cases", 
            "table_desc_record_samples", 
            "table_fields_record_samples", 
            "model"
        ]
        for key in required_keys:
            if key not in config:
                sys.exit(f"错误：配置缺少必要字段: {key}")
        
        try:
            print(f"开始评估，配置: {config}")
            cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
            report_dir = f"{REPORT_ROOT}/{cur_time}"

            try:
                Path(report_dir).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                sys.exit(f"错误：创建报告目录失败: {str(e)}")
            
            backup_configs(report_dir)
            apply_config(config)
            restart_bird()
            start_eval(report_dir)
            make_report(config, report_dir)
        except Exception as e:
            sys.exit(f"错误：执行评估时发生错误: {str(e)}")

if __name__ == "__main__":
    configs = set_configs()
    run_eval(configs)
