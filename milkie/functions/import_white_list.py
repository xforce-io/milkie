WhiteListImport = [
    "math",
    "random",
    "typing",
    "typing.List",
    "typing.Dict",
    "typing.Optional",
    "typing.Union",
    "datetime",
    "datetime.datetime",
]

PreImport = [
    "datetime",
]

def addPreImport(code :str):
    for preImport in PreImport:
        code = f"import {preImport}\n" + code
    return code