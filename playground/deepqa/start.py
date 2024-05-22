import subprocess

from milkie.config.config import RewriteStrategy
from playground.set_env import EnvSettings, NewEnv

strategy = "deepqa"
llm_model = "qwenv1.5-chat-14b"
framework = "VLLM"

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
        subprocess.run([
                "python", 
                "-m",
                "playground.deepqa.experiment",
                "with",
                f"strategy={strategy}",
                f"llm_model={llm_model}",
                f"framework={framework}",
                f"rewrite_strategy={rewrite_strategy}",
                "-l",
                "DEBUG"
            ],
            env=NewEnv)