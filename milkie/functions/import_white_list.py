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
]

PreImport = [
    "datetime",
]

def addPreImport(code :str):
    for preImport in PreImport:
        code = f"import {preImport}\n" + code
    return code