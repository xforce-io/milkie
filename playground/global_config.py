from milkie.strategy import Strategy
from milkie.utils.data_utils import loadFromYaml
from milkie.config.config import GlobalConfig
from playground.model_repos import GModelRepo

def makeGlobalConfig(
        strategy :Strategy,
        **kwargs) -> GlobalConfig:
    configYaml = loadFromYaml("config/global.yaml")
    
    if "llm_model" in kwargs:
        model = GModelRepo.getModel(kwargs["llm_model"])
        model_name = model.getModelPath()
        tensor_parallel_size = model.getTensorParrallelSize()

        # 判断模型来源
        source = kwargs.get("source", "local")
        
        # 确保来源在llm配置中存在
        if source not in configYaml["llm"]:
            configYaml["llm"][source] = []
        
        # 查找是否已存在此模型配置
        model_config = None
        for config in configYaml["llm"][source]:
            if config.get("model") == model_name:
                model_config = config
                break
        
        # 如果不存在，创建新的配置
        if not model_config:
            model_config = {"model": model_name}
            configYaml["llm"][source].append(model_config)
        
        model_config["tensor_parallel_size"] = tensor_parallel_size
        
        # 添加其他参数
        if "system_prompt" in kwargs:
            model_config["system_prompt"] = kwargs["system_prompt"]
            
        if "framework" in kwargs:
            model_config["framework"] = kwargs["framework"]

        if "device" in kwargs:
            model_config["device"] = kwargs["device"]

        if "ctx_len" in kwargs:
            model_config["ctx_len"] = kwargs["ctx_len"]

        if "batch_size" in kwargs:
            model_config["batch_size"] = kwargs["batch_size"]
            
        # 确保model_args存在
        if "model_args" not in model_config:
            model_config["model_args"] = {}
            
        if "quantization_type" in kwargs:
            model_config["model_args"]["quantization_type"] = kwargs["quantization_type"]

        if "attn_implementation" in kwargs:
            model_config["model_args"]["attn_implementation"] = kwargs["attn_implementation"]
        
        if "torch_compile" in kwargs:
            model_config["model_args"]["torch_compile"] = kwargs["torch_compile"]
            
        # 确保generation_args存在
        if "generation_args" not in model_config:
            model_config["generation_args"] = {}

        if "repetition_penalty" in kwargs:
            model_config["generation_args"]["repetition_penalty"] = kwargs["repetition_penalty"]

        if "temperature" in kwargs:
            model_config["generation_args"]["temperature"] = kwargs["temperature"]

        if "do_sample" in kwargs:
            model_config["generation_args"]["do_sample"] = kwargs["do_sample"]
        
        if "use_cache" in kwargs:
            model_config["generation_args"]["use_cache"] = kwargs["use_cache"]
            
        if "prompt_lookup_num_tokens" in kwargs:
            model_config["generation_args"]["prompt_lookup_num_tokens"] = kwargs["prompt_lookup_num_tokens"]

    def getAgentConfig(name :str):
        for agentConfig in configYaml["agents"]:
            if agentConfig["config"] == name:
                return agentConfig
        return None

    agentConfig = getAgentConfig(strategy.getAgentName())
    if agentConfig is None:
        return GlobalConfig(configYaml)
    
    if "chunk_augment" in kwargs:
        agentConfig["retrieval"]["chunk_augment"] = kwargs["chunk_augment"]
    
    if "reranker" in kwargs:
        agentConfig["retrieval"]["reranker"]["name"] = kwargs["reranker"]

    if "rerank_position" in kwargs:
        agentConfig["retrieval"]["reranker"]["position"] = kwargs["rerank_position"]

    if "rewrite_strategy" in kwargs:
        agentConfig["retrieval"]["rewrite_strategy"] = kwargs["rewrite_strategy"]

    if "chunk_size" in kwargs:
        agentConfig["index"]["chunk_size"] = kwargs["chunk_size"]

    if "channel_recall" in kwargs:
        agentConfig["retrieval"]["channel_recall"] = kwargs["channel_recall"]
    
    if "similarity_top_k" in kwargs:
        agentConfig["retrieval"]["similarity_top_k"] = kwargs["similarity_top_k"]

    return GlobalConfig(configYaml)