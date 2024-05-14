import subprocess

benchmarks = "benchmark/fd100_key.jsonl;benchmark/410_key.jsonl"
prompt = "qa_strict"

for llm_model in [
        "QwenV15S14bChat",
        "QwenV15S14bGPTQINT4Chat",
        "QwenV15S14bGPTQINT8Chat"]:
    for framework in ["LMDEPLOY", "VLLM"]:
        
        subprocess.run([
            "python", 
            "-m",
            "playground.speed.experiment",
            "with",
            f"llm_model={llm_model}",
            f"framework={framework}",
            f"prompt={prompt}",
            f"benchmarks={benchmarks}",
            "-l",
            "DEBUG"
        ])