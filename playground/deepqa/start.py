import subprocess

from milkie.config.config import RewriteStrategy
from playground.init import NewEnv

strategy = "deepqa"
llm_model = "qwenv1.5-chat-14b"
framework = "VLLM"
benchmarks = "benchmark/santi.jsonl"

for llm_model in [
        "qwenv1.5-chat-14b",
        "aishuv2-chat-14b",
        "aishuv2-chat-14b-awq",
        "aishuv2-chat-14b-gptq-int8",
        "internlmv2-chat-20b"]:
    for rewrite_strategy in [
        RewriteStrategy.NONE.name,
        RewriteStrategy.QUERY_REWRITE.name,
    ]:
        for chunk_augment in [
            "NONE",
            "SIMPLE",
        ]:
            subprocess.run([
                    "python", 
                    "-m",
                    "playground.deepqa.experiment",
                    "with",
                    f"strategy={strategy}",
                    f"llm_model={llm_model}",
                    f"framework={framework}",
                    f"rewrite_strategy={rewrite_strategy}",
                    f"chunk_augment={chunk_augment}",
                    f"benchmarks={benchmarks}",
                    "-l",
                    "DEBUG"
                ],
                env=NewEnv)