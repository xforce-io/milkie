from milkie.strategy import Strategy
from milkie.utils.data_utils import loadFromYaml
from milkie.config.config import GlobalConfig

def makeGlobalConfig(
        strategy :Strategy,
        **kwargs) -> GlobalConfig:
    configYaml = loadFromYaml("config/global.yaml")
    if "llm_model" in kwargs:
        configYaml["llm"]["model"] = kwargs["llm_model"]

    if "system_prompt" in kwargs:
        configYaml["llm"]["system_prompt"] = kwargs["system_prompt"]

    if "framework" in kwargs:
        configYaml["llm"]["framework"] = kwargs["framework"]

    if "device" in kwargs:
        configYaml["llm"]["device"] = kwargs["device"]

    if "batch_size" in kwargs:
        configYaml["llm"]["batch_size"] = kwargs["batch_size"]

    if "quantization_type" in kwargs:
        configYaml["llm"]["model_args"]["quantization_type"] = kwargs["quantization_type"]

    if "attn_implementation" in kwargs:
        configYaml["llm"]["model_args"]["attn_implementation"] = kwargs["attn_implementation"]
    
    if "torch_compile" in kwargs:
        configYaml["llm"]["model_args"]["torch_compile"] = kwargs["torch_compile"]

    if "repetition_penalty" in kwargs:
        configYaml["llm"]["generation_args"]["repetition_penalty"] = kwargs["repetition_penalty"]

    if "temperature" in kwargs:
        configYaml["llm"]["generation_args"]["temperature"] = kwargs["temperature"]

    if "do_sample" in kwargs:
        configYaml["llm"]["generation_args"]["do_sample"] = kwargs["do_sample"]
    
    if "use_cache" in kwargs:
        configYaml["llm"]["generation_args"]["use_cache"] = kwargs["use_cache"]
        
    if "prompt_lookup_num_tokens" in kwargs:
        configYaml["llm"]["generation_args"]["prompt_lookup_num_tokens"] = kwargs["prompt_lookup_num_tokens"]

    def getAgentConfig(name :str):
        for agentConfig in configYaml["agents"]:
            if agentConfig["config"] == name:
                return agentConfig
        return None

    agentConfig = getAgentConfig(strategy.getAgentName())
    if agentConfig is None:
        return GlobalConfig(configYaml)
    
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