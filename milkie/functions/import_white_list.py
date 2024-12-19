WhiteListImport = [
    "os",
    "os.path",
    "math",
    "random",
    "typing",
    "typing.List",
    "typing.Dict",
    "typing.Optional",
    "typing.Union",
    "datetime",
    "datetime.datetime",
    "csv",
    "json",
    "re",
]

PreImport = [
    "datetime",
    "json",
]

def addPreImport(code :str):
    for preImport in PreImport:
        code = f"import {preImport}\n" + code
    return code