import subprocess

strategy = "raw"
llm_model = "qwenv1.5-chat-14b"
framework = "VLLM"

subprocess.run([
    "python", 
    "-m",
    "playground.deepqa.experiment",
    "with",
    f"strategy={strategy}",
    f"llm_model={llm_model}",
    f"framework={framework}",
    f"rewrite_strategy=NONE",
    "-l",
    "DEBUG"
])