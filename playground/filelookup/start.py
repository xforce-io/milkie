import subprocess

from milkie.config.config import RewriteStrategy
from playground.init import NewEnv

benchmarks = "benchmark/file_lookup.jsonl"

subprocess.run([
        "python", 
        "-m",
        "playground.filelookup.experiment",
        "with",
        f"strategy=file_lookup",
        f"benchmarks={benchmarks}",
        "-l",
        "DEBUG"
    ],
    env=NewEnv)