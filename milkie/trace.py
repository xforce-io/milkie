def stdout(value :str, info: bool = False, **kwargs):
    if info or ("verbose" in kwargs and kwargs["verbose"]):
        print(value, **{k: v for k, v in kwargs.items() if k in ["end", "flush"]})
