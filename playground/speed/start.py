import subprocess

from playground.set_env import EnvSettings, NewEnv

strategy = "raw"
llm_model = "qwenv1.5-chat-14b"
framework = "VLLM"
batch_size = 50
prompt_lookup_num_tokens = None
system_prompt = "system_prompt"
prompt = "qa_strict"
benchmarks = "benchmark/fd100_key.jsonl"

for llm_model in [
        "yi-chat-34b",
        "qwenv1.5-chat-14b",
        "aishuv2-chat-14b",
        "aishuv2-chat-14b-awq",
        "aishuv2-chat-14b-gptq-int8",
        "internlmv2-chat-20b"]:
    for framework in ["VLLM"]:
        for system_prompt in [
                "system_prompt", 
                "system_qwen"]:
            for prompt in [
                    "qa_init", 
                    "qa_strict"]:
                subprocess.run([
                        "python", 
                        "-m",
                        "playground.speed.experiment",
                        "with",
                        f"strategy={strategy}",
                        f"llm_model={llm_model}",
                        f"framework={framework}",
                        f"batch_size={batch_size}",
                        f"prompt_lookup_num_tokens={prompt_lookup_num_tokens}",
                        f"system_prompt={system_prompt}",
                        f"prompt={prompt}",
                        f"benchmarks={benchmarks}",
                        "-l",
                        "DEBUG"
                    ],
                    env=NewEnv)