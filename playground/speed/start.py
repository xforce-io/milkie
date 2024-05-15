import subprocess

strategy = "raw"
llm_model = "QwenV15S14bChat"
framework = "VLLM"
batch_size = 50
prompt_lookup_num_tokens = None
system_prompt = "system_prompt"
prompt = "qa_strict"
benchmarks = "benchmark/fd100_key.jsonl"

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
                    f"strategy={strategy}"
                    f"llm_model={llm_model}",
                    f"framework={framework}",
                    f"batch_size={batch_size}",
                    f"prompt_lookup_num_tokens={prompt_lookup_num_tokens}",
                    f"system_prompt={system_prompt}",
                    f"prompt={prompt}",
                    f"benchmarks={benchmarks}",
                    "-l",
                    "DEBUG"
                ])