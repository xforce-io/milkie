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
  graph?: any;
}

// 定义查询选项接口
export interface QueryOptions {
  graph?: boolean;
  format?: string;
  [key: string]: any;
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

// 流式查询代理
export const streamQueryAgent = (
  agent: string, 
  query: string, 
  onData: (data: string) => void,
  onError: (error: any) => void,
  onComplete: () => void,
  options: QueryOptions = {}
): (() => void) => {
  console.log('启动流式查询，参数:', { agent, query, options });
  
  const url = new URL(getApiBaseUrl() + '/v1/agent/stream');
  url.searchParams.append('agent', agent);
  url.searchParams.append('query', query);
  
  if (options.graph !== undefined) {
    url.searchParams.append('graph', String(options.graph));
  }
  
  if (options.format) {
    url.searchParams.append('format', options.format);
  }
  
  console.log('创建流式连接:', url.toString());
  
  // 创建一个标志位，表示连接是否已关闭
  let isClosed = false;
  
  // 批处理缓冲区，收集少量消息后再更新 UI
  let messageBuffer = '';
  let lastFlushTime = Date.now();
  const FLUSH_INTERVAL = 2000; // 降低到2000ms刷新一次，提高实时性
  
  // 刷新缓冲区的函数
  const flushBuffer = () => {
    if (messageBuffer.length > 0) {
      try {
        console.log('刷新消息缓冲区:', messageBuffer);
        onData(messageBuffer);
      } catch (error) {
        console.error('处理SSE消息时出错:', error);
      }
      messageBuffer = '';
      lastFlushTime = Date.now();
    }
  };
  
  // 设置连接超时保护，确保连接不会无限挂起
  // 30秒无数据自动关闭
  const connectionTimeoutId = setTimeout(() => {
    console.warn('SSE连接超时（10秒无数据），自动关闭');
    if (!isClosed) {
      try {
        // 刷新缓冲区中的所有消息
        flushBuffer();
        
        eventSource.close();
        isClosed = true;
        onComplete();
      } catch (e) {
        console.error('关闭超时SSE连接时出错:', e);
      }
    }
  }, 10000);
  
  // 数据活跃检测，长时间无数据自动关闭
  let lastActivityTime = Date.now();
  const activityCheckerId = setInterval(() => {
    // 如果已关闭，清理检测器
    if (isClosed) {
      clearInterval(activityCheckerId);
      return;
    }
    
    // 如果缓冲区有数据但超过了刷新间隔，刷新缓冲区
    const timeSinceLastFlush = Date.now() - lastFlushTime;
    if (messageBuffer.length > 0 && timeSinceLastFlush > FLUSH_INTERVAL) {
      flushBuffer();
    }
    
    // 如果超过15秒无活动，关闭连接
    const inactiveTime = Date.now() - lastActivityTime;
    if (inactiveTime > 15000) {
      console.warn(`SSE连接${inactiveTime}ms无活动，自动关闭`);
      clearInterval(activityCheckerId);
      clearTimeout(connectionTimeoutId);
      
      // 确保刷新最后的消息
      flushBuffer();
      
      try {
        eventSource.close();
        isClosed = true;
        onComplete();
      } catch (e) {
        console.error('关闭不活跃SSE连接时出错:', e);
      }
    }
  }, 2000); // 降低检查间隔到2秒，提高响应性
  
  const eventSource = new EventSource(url.toString());
  
  // 注册所有可能的事件处理器
  const handlers = {
    open: () => {
      console.log('SSE连接已打开');
      lastActivityTime = Date.now();
    },
    
    message: (event: MessageEvent) => {
      // 如果连接已关闭，不再处理新消息
      if (isClosed) return;
      
      // 更新活动时间
      lastActivityTime = Date.now();
      
      const rawData = event.data;
      console.log('SSE接收到原始数据:', rawData);
      
      try {
        // 尝试解析JSON
        const parsedData = JSON.parse(rawData);
        
        // 检查是否包含节点数据
        if (parsedData.resp && parsedData.resp.nodes) {
          console.log('找到节点数据在resp.nodes中');
        } else if (parsedData.nodes) {
          console.log('找到节点数据在nodes中');
        }
        
        // 将数据传递给处理函数
        onData(rawData);
      } catch (error) {
        console.error('SSE数据解析出错:', error);
        
        // 检查是否是多个JSON对象连接在一起的情况
        if (rawData.includes('}{')) {
          console.log('检测到多个JSON对象，尝试分离处理');
          try {
            // 尝试修复连接在一起的JSON对象
            const fixedData = rawData.replace('}{', '},{');
            const jsonArray = JSON.parse('[' + fixedData + ']');
            // 单独处理每个对象
            jsonArray.forEach((item: any) => {
              onData(JSON.stringify(item));
            });
            return;
          } catch (e) {
            console.error('修复连接的JSON对象失败:', e);
          }
        }
        
        // 仍然传递原始数据
        onData(rawData);
      }
    },
    
    error: (error: Event) => {
      // 如果连接已关闭，不再处理错误
      if (isClosed) return;
      
      console.error('SSE错误:', error);
      
      // 清理定时器
      clearTimeout(connectionTimeoutId);
      clearInterval(activityCheckerId);
      
      // 确保状态一致
      isClosed = true;
      
      // 刷新所有未发送的数据
      flushBuffer();
      
      try {
        onError(error);
      } catch (e) {
        console.error('调用错误回调时出错:', e);
      }
      
      try {
        eventSource.close();
      } catch (e) {
        console.error('关闭SSE连接时出错:', e);
      }
      
      try {
        onComplete();
      } catch (e) {
        console.error('调用完成回调时出错:', e);
      }
    },
    
    complete: (event: Event) => {
      // 如果连接已关闭，不再处理完成事件
      if (isClosed) return;
      
      console.log('服务器发送complete事件，流式查询完成');
      
      // 清理定时器
      clearTimeout(connectionTimeoutId);
      clearInterval(activityCheckerId);
      
      // 确保状态一致
      isClosed = true;
      
      // 刷新最后的数据
      flushBuffer();
      
      try {
        eventSource.close();
      } catch (e) {
        console.error('关闭SSE连接时出错:', e);
      }
      
      try {
        onComplete();
      } catch (e) {
        console.error('调用完成回调时出错:', e);
      }
    }
  };
  
  // 绑定事件处理器
  eventSource.onopen = handlers.open;
  eventSource.onmessage = handlers.message;
  eventSource.onerror = handlers.error;
  eventSource.addEventListener('complete', handlers.complete);
  
  // 返回关闭函数
  return () => {
    console.log('手动关闭流式连接');
    
    // 如果连接已关闭，不再重复关闭
    if (isClosed) {
      console.log('连接已关闭，忽略');
      return;
    }
    
    // 清理定时器
    clearTimeout(connectionTimeoutId);
    clearInterval(activityCheckerId);
    
    // 标记为已关闭
    isClosed = true;
    
    // 刷新最后的数据
    flushBuffer();
    
    // 安全关闭连接
    try {
      eventSource.close();
    } catch (e) {
      console.error('手动关闭SSE连接时出错:', e);
    }
    
    // 通知完成
    try {
      onComplete();
    } catch (e) {
      console.error('调用完成回调时出错:', e);
    }
  };
}; 