ModelYi34 = "/mnt/data1/.cache/huggingface/hub/01ai/Yi-34B-Chat/"

Prefix = "/mnt/data1/.cache/modelscope/hub/"
ModelBaichuan13bChat = Prefix+"baichuan-inc/Baichuan2-13B-Chat"
ModelQwen14bChat = Prefix+"qwen/Qwen-14B-Chat"
ModelQwenV15S14bChat = Prefix+"qwen/Qwen1.5-14B-Chat/"
ModelQwenV15S14bGPTQINT4Chat = Prefix+"qwen/Qwen1___5-14B-Chat-GPTQ-Int4/"
ModelQwenV15S14bGPTQINT8Chat = Prefix+"qwen/Qwen1___5-14B-Chat-GPTQ-Int8/"
ModelQwenV15S14bAWQChat = Prefix+"qwen/Qwen1___5-14B-Chat-AWQ/"

PrefixAishuReader = "/mnt/data2/.cache/huggingface/hub/"
ModelAishuReader2_Chat = PrefixAishuReader + "Qwen-14B-Chat-1.5-aishuV2"
ModelAishuReader2_Chat_AWQ = PrefixAishuReader + "Qwen-14B-Chat-1.5-aishuV2-awq"
ModelAishuReader2_Chat_GPTQ8 = PrefixAishuReader + "Qwen-14B-Chat-1.5-aishuV2-gptq-int8"

PrefixInternlm2 = "/mnt/data3/models/"
ModelInternlm2_Chat_20b = PrefixInternlm2 + "internlm2-chat-20b_v2"

ModelInternlm2_Chat_20b_Longrag = PrefixInternlm2 + "../trained_models/internlm2-chat-20b_log_3k5_ruozhiba_longrag"
ModelInternlm2_Chat_20b_Longrag_GPTQ8 = PrefixInternlm2 + "../trained_models/internlm2-chat-20b_log_3k5_ruozhiba_longrag_gptq"

def getModel(name :str) -> str:
    if name == "Yi34":
        return ModelYi34
    elif name == "Baichuan13bChat":
        return ModelBaichuan13bChat
    elif name == "Qwen14bChat":
        return ModelQwen14bChat
    elif name == "QwenV15S14bChat":
        return ModelQwenV15S14bChat
    elif name == "QwenV15S14bGPTQINT4Chat":
        return ModelQwenV15S14bGPTQINT4Chat
    elif name == "QwenV15S14bGPTQINT8Chat":
        return ModelQwenV15S14bGPTQINT8Chat
    elif name == "QwenV15S14bAWQChat":
        return ModelQwenV15S14bAWQChat
    elif name == "AishuReader2_Chat":
        return ModelAishuReader2_Chat
    elif name == "AishuReader2_Chat_AWQ":
        return ModelAishuReader2_Chat_AWQ
    elif name == "AishuReader2_Chat_GPTQ8":
        return ModelAishuReader2_Chat_GPTQ8
    elif name == "Internlm2_Chat_20b":
        return ModelInternlm2_Chat_20b
    elif name == "Internlm2_Chat_20b_Longrag":
        return ModelInternlm2_Chat_20b_Longrag
    elif name == "Internlm2_Chat_20b_Longrag_GPTQ8":
        return ModelInternlm2_Chat_20b_Longrag_GPTQ8
    else:
        raise ValueError(f"Unknown model name: {name}")

