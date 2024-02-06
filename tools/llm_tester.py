from transformers import AutoModelForCausalLM, AutoTokenizer

Models = [
    "/mnt/data1/.cache/modelscope/hub/qwen/Qwen-7B-Chat/",
    "/mnt/data1/.cache/modelscope/hub/qwen/Qwen-14B-Chat/",
    "/mnt/data1/.cache/modelscope/hub/baichuan-inc/Baichuan2-7B-Chat/",
    "/mnt/data1/.cache/modelscope/hub/baichuan-inc/Baichuan2-13B-Chat/",
]

Prompt = '''
'''

def testModel(model, query):
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", trust_remote_code=True).eval()
    response, history = model.chat(tokenizer, query, history=None)
    print("=================================================")
    print("model:", model)
    print("query:", query)
    print("-------------------------------------------------")
    print(response)