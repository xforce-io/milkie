import subprocess

benchmarks = "benchmark/fd100_key.jsonl"
prompt = "qa_strict"

for llm_model in [
        "QwenV15S14bChat",
        "QwenV15S14bGPTQINT4Chat",
        "QwenV15S14bGPTQINT8Chat"]:
    for framework in ["LMDEPLOY", "VLLM"]:
        for system_prompt in ["system_prompt", "system_qwen"]:
            for prompt in ["qa_init", "qa_strict"]:
                subprocess.run([
                    "python", 
                    "-m",
                    "playground.speed.experiment",
                    "with",
                    f"llm_model={llm_model}",
                    f"framework={framework}",
                    f"system_prompt={system_prompt}",
                    f"prompt={prompt}",
                    f"benchmarks={benchmarks}",
                    "-l",
                    "DEBUG"
                ])