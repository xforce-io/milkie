import axios from 'axios';

// 定义API基础URL（可通过环境变量覆盖）
export const API_BASE_URL = 'http://localhost:8000';

// 创建获取API URL的函数
export const getApiBaseUrl = (): string => {
  // 优先使用window对象中的全局配置
  if (window.API_BASE_URL) {
    return window.API_BASE_URL;
  }
  // 使用环境变量或默认值
  return API_BASE_URL;
};

// 定义接口类型
export interface AgentInfo {
  agent: string;
  description: string;
  args: Record<string, any>;
}

// 添加全局变量声明
declare global {
  interface Window {
    API_BASE_URL?: string;
  }
}

export interface AgentInfosResponse {
  errno: number;
  errmsg: string;
  resp: AgentInfo[];
}

export interface AgentResponse {
  errno: number;
  errmsg: string;
  resp: string;
}

// 创建axios实例
const apiClient = axios.create({
  baseURL: getApiBaseUrl(),
  headers: {
    'Content-Type': 'application/json',
  },
});

// 获取代理信息
export const getAgentInfos = async (): Promise<AgentInfosResponse> => {
  const response = await apiClient.post<AgentInfosResponse>('/v1/agent/infos', {
    agent: ""  // 发送空字符串作为agent参数
  });
  return response.data;
};

// 执行代理查询
export const queryAgent = async (
  agent: string,
  query: string,
  args: Record<string, any> = {}
): Promise<AgentResponse> => {
  const response = await apiClient.post<AgentResponse>('/v1/agent/query', {
    agent,
    query,
    args,
  });
  return response.data;
};

// 执行流式代理查询
export const streamQueryAgent = (
  agent: string,
  query: string,
  onData: (data: string) => void,
  onError: (error: any) => void,
  onComplete: () => void,
  args: Record<string, any> = {}
): () => void => {
  // 创建EventSource对象
  const eventSource = new EventSource(
    `${getApiBaseUrl()}/v1/agent/stream?agent=${encodeURIComponent(agent)}&query=${encodeURIComponent(query)}`
  );

  // 监听消息事件
  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.resp) {
        onData(data.resp);
      }
    } catch (error) {
      console.error('解析消息出错:', error);
      onData(event.data); // 如果解析失败，直接返回原始数据
    }
  };

  // 监听错误事件
  eventSource.onerror = (error) => {
    console.error('流式查询出错:', error);
    onError(error);
    eventSource.close();
  };

  // 监听完成事件
  eventSource.addEventListener('complete', () => {
    onComplete();
    eventSource.close();
  });

  // 返回关闭函数
  return () => {
    eventSource.close();
  };
}; 