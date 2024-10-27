from milkie.config.constant import KeywordMute


def stdout(value :str, **kwargs):
    if KeywordMute in kwargs and kwargs[KeywordMute]:
        return

    print(value, **kwargs)