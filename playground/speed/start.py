import subprocess

benchmarks = "benchmark/fd100_key.jsonl"
prompt = "qa_strict"

for llm_model in [
        "QwenV15S14bChat",
        "AishuReader2_Chat",
        "AishuReader2_Chat_AWQ",
        "AishuReader2_Chat_GPTQ8",
        "Internlm2_Chat_20b"]:
    for framework in ["VLLM"]:
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