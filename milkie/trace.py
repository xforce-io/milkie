def stdout(value :str, args :dict = {}):
    if "__mute__" in args and args["__mute__"]:
        return

    print(value)