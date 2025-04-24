import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { Input, Button, Typography, Card, Spin, message, Switch, Tree, Collapse, Tag, Modal, Space, Row, Col, Tabs } from 'antd';
import { SendOutlined, StopOutlined, DownOutlined, UpOutlined } from '@ant-design/icons';
import { queryAgent, streamQueryAgent, AgentInfo, API_BASE_URL } from '../services/api';
import { throttle } from 'lodash';
import './QueryCanvas.css';
import ReactJson from 'react-json-view';

// 添加必要的类型定义
interface GraphNode {
  id: string;
  type?: string;
  label?: string;
  name?: string;
  children?: GraphNode[];
  [key: string]: any;
}

interface TreeNode {
  key: string;
  title: string;
  children?: TreeNode[];
}

interface TreeData extends TreeNode {}

const { Title, Text } = Typography;
const { Panel } = Collapse;

// 检查是否有有效的图结构
const hasValidGraphStructure = (obj: any): boolean => {
  return obj && (obj.id || obj.name);
};

interface QueryCanvasProps {
  selectedAgent: AgentInfo | null;
  onStreamResponse?: (response: any) => void;
  onClearGraph?: () => void;
}

const QueryCanvas: React.FC<QueryCanvasProps> = ({ selectedAgent, onStreamResponse, onClearGraph }) => {
  // 所有状态定义必须在组件顶部，并且顺序不能变
  const [query, setQuery] = useState<string>('');
  const [response, setResponse] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<string>('graph');
  const [graphData, setGraphData] = useState<any>(null);
  const [treeExpandedKeys, setTreeExpandedKeys] = useState<React.Key[]>(['root']);
  const [graphEnabled, setGraphEnabled] = useState<boolean>(true);
  const [responseFullscreen, setResponseFullscreen] = useState<boolean>(false);
  const [graphFullscreen, setGraphFullscreen] = useState<boolean>(false);
  const [expandedCards, setExpandedCards] = useState<Set<string>>(new Set());
  
  // ref 定义
  const closeStreamRef = useRef<(() => void) | null>(null);
  const responseRef = useRef<HTMLDivElement>(null);
  const graphRenderTimeoutRef = useRef<number | null>(null);
  
  // 所有回调函数必须在这里定义，并且使用 useCallback 包装
  // 处理流式响应
  const handleStreamData = useCallback((data: string) => {
    try {
      // 尝试解析JSON
      let parsedData;
      try {
        parsedData = typeof data === 'string' ? JSON.parse(data) : data;
        console.log('成功解析流式数据:', {
          类型: typeof parsedData,
          数据: parsedData,
          是否有nodes: !!parsedData.nodes,
          是否有graph: !!parsedData.graph
        });
      } catch (e) {
        console.error('解析JSON失败:', e);
        return;
      }
      
      // 标准化节点数据格式
      let normalizedData = parsedData;
      
      // 如果数据中有graph字段
      if (parsedData.graph) {
        normalizedData = parsedData.graph;
      }
      
      // 确保有nodes字段
      if (!normalizedData.nodes && Array.isArray(normalizedData)) {
        normalizedData = { nodes: normalizedData };
      }
      
      // 如果没有nodes字段但有children字段，转换格式
      if (!normalizedData.nodes && normalizedData.children) {
        // 提取所有children作为nodes
        const extractNodes = (node: any, parentId?: string): any[] => {
          const nodes = [];
          const currentNode = {
            ...node,
            id: node.id || `node_${Math.random().toString(36).substr(2, 9)}`,
            parentId
          };
          
          delete currentNode.children;
          nodes.push(currentNode);
          
          if (Array.isArray(node.children)) {
            node.children.forEach((child: any) => {
              nodes.push(...extractNodes(child, currentNode.id));
            });
          }
          
          return nodes;
        };
        
        normalizedData = { 
          nodes: extractNodes(normalizedData)
        };
      }
      
      // 确保每个节点都有必要的字段
      if (normalizedData.nodes) {
        normalizedData.nodes = normalizedData.nodes.map((node: any, idx: number) => ({
          id: node.id || `node_${idx}`,
          type: node.type || 'COMMON',
          label: node.label || node.name || node.type || 'Node',
          name: node.name || node.label || `节点 ${idx + 1}`,
          ...node
        }));
      }
      
      console.log('规范化后的数据:', {
        类型: typeof normalizedData,
        是否有nodes: !!normalizedData.nodes,
        节点数量: normalizedData.nodes?.length || 0,
        节点示例: normalizedData.nodes?.[0]
      });
      
      // 更新执行图数据
      if (onStreamResponse && normalizedData && normalizedData.nodes?.length > 0) {
        onStreamResponse(normalizedData);
      }
      
      // 更新响应文本
      if (parsedData.resp) {
        if (parsedData.resp.content) {
          setResponse(prev => prev + parsedData.resp.content);
        } else if (typeof parsedData.resp === 'string') {
          setResponse(prev => prev + parsedData.resp);
        }
      }
    } catch (error: any) {
      console.error('[QueryCanvas] 解析流式数据失败', error.message);
    }
  }, [onStreamResponse]);

  // 递归合并图节点
  const mergeGraphNodes = useCallback((oldNodes: GraphNode[], newNodes: GraphNode[]): GraphNode[] => {
    // 使用Map优化查找性能
    const oldNodesMap = new Map(oldNodes.map(node => [node.id, node]));
    const mergedNodes = [...oldNodes];
    
    newNodes.forEach(newNode => {
      const existingNode = oldNodesMap.get(newNode.id);
      if (existingNode) {
        // 只更新变化的属性
        Object.keys(newNode).forEach(key => {
          const k = key as keyof GraphNode;
          if (k !== 'id' && k !== 'children' &&
              JSON.stringify(existingNode[k]) !== JSON.stringify(newNode[k])) {
            existingNode[k] = newNode[k];
          }
        });
        
        // 递归处理子节点
        if (newNode.children && newNode.children.length > 0) {
          existingNode.children = mergeGraphNodes(
            existingNode.children || [],
            newNode.children
          );
        }
      } else {
        mergedNodes.push(newNode);
      }
    });
    
    return mergedNodes;
  }, []);

  // 使用新数据更新图结构
  const updateGraphWithNewData = useCallback(throttle((jsonObj: any) => {
    console.log('收到新的图数据更新:', jsonObj);
    
    // 尝试处理字符串格式的数据
    let processedObj = jsonObj;
    if (typeof jsonObj === 'string') {
      try {
        processedObj = JSON.parse(jsonObj);
        console.log('成功解析字符串数据:', processedObj);
      } catch (e) {
        console.error('解析字符串数据失败:', e);
        return;
      }
    }

    // 检查数据有效性
    if (!processedObj || typeof processedObj !== 'object') {
      console.error('无效的图数据格式');
      return;
    }

    // 提取图数据
    if (processedObj.graph) {
      processedObj = processedObj.graph;
    }

    // 确保有必要的字段
    if (!processedObj.nodes && !processedObj.children) {
      console.error('数据中缺少nodes或children字段');
      return;
    }

    // 通知父组件
    if (onStreamResponse) {
      console.log('向父组件发送处理后的数据');
      onStreamResponse(processedObj);
    }
  }, 500), [onStreamResponse]);

  // 提取图数据的函数也用 useCallback 包装
  const extractGraphData = useCallback((jsonObj: any): GraphNode | null => {
    // 如果是字符串，尝试解析为JSON对象
    if (typeof jsonObj === 'string') {
      try {
        const parsed = JSON.parse(jsonObj);
        return extractGraphData(parsed);
      } catch (e) {
        console.error('[QueryCanvas] 字符串解析为JSON失败');
        return null;
      }
    }
    
    // 如果对象自身就是有效的图结构
    if (hasValidGraphStructure(jsonObj)) {
      return jsonObj;
    }
    
    return null;
  }, []);

  const handleStreamQuery = useCallback(async () => {
    if (!selectedAgent || !query.trim()) {
      message.warning('请选择代理并输入查询内容');
      return;
    }

    setLoading(true);
    setIsStreaming(true);
    setResponse('');
    setGraphData(null);
    
    // 清除执行图数据
    if (onClearGraph) {
      onClearGraph();
    }

    try {
      console.log('开始流式查询，确保包含图数据');
      const closeStream = streamQueryAgent(
        selectedAgent.agent,
        query,
        handleStreamData,
        (error) => {
          console.error('[QueryCanvas] 流式查询出错:', error);
          message.error('查询出错，请检查服务器连接');
          setLoading(false);
          setIsStreaming(false);
        },
        () => {
          console.log('[QueryCanvas] 流式查询完成');
          setLoading(false);
          setIsStreaming(false);
        },
        { graph: true, format: 'json' }  // 添加json格式选项
      );
      closeStreamRef.current = closeStream;
    } catch (error) {
      console.error('[QueryCanvas] 流式查询失败:', error);
      message.error('查询失败，请重试');
      setLoading(false);
      setIsStreaming(false);
    }
  }, [selectedAgent, query, handleStreamData, onClearGraph]);

  const handleStopStream = useCallback(() => {
    if (closeStreamRef.current) {
      closeStreamRef.current();
      closeStreamRef.current = null;
    }
    setLoading(false);
    setIsStreaming(false);
  }, []);

  // CSS样式
  const queryCanvasStyles = useMemo(() => ({
    textResponse: {
      maxHeight: '500px',
      overflowY: 'auto' as const,
      fontSize: '14px',
      lineHeight: '1.5',
      whiteSpace: 'pre-wrap' as const,
      wordBreak: 'break-word' as const,
      contain: 'content' as const,
    }
  }), []);

  // 添加高亮动画样式
  useEffect(() => {
    const styleElement = document.createElement('style');
    styleElement.textContent = `
      @keyframes highlight-update {
        0% {
          background-color: #ffff9c;
          border-color: #faad14;
        }
        100% {
          background-color: #fff;
          border-color: #f0f0f0;
        }
      }
    `;
    document.head.appendChild(styleElement);
    
    return () => {
      document.head.removeChild(styleElement);
    };
  }, []);
  
  // 组件卸载时清理资源
  useEffect(() => {
    return () => {
      if (closeStreamRef.current) {
        try {
          closeStreamRef.current();
        } catch (error) {
          console.error('[QueryCanvas] 组件卸载时关闭流式连接错误:', error);
        } finally {
          closeStreamRef.current = null;
        }
      }
      
      if (graphRenderTimeoutRef.current !== null) {
        window.clearTimeout(graphRenderTimeoutRef.current);
        graphRenderTimeoutRef.current = null;
      }
    };
  }, []);

  const renderResponseArea = useCallback(() => {
    return (
      <Card
        title="响应结果"
        extra={
          <Space>
            <Button
              icon={responseFullscreen ? <UpOutlined /> : <DownOutlined />}
              onClick={() => setResponseFullscreen(!responseFullscreen)}
            >
              {responseFullscreen ? '收起' : '展开'}
            </Button>
          </Space>
        }
        className={`response-card ${responseFullscreen ? 'fullscreen' : ''}`}
      >
        <div ref={responseRef} style={queryCanvasStyles.textResponse}>
          {response}
        </div>
      </Card>
    );
  }, [response, responseFullscreen, queryCanvasStyles]);

  const renderQueryButtons = useCallback(() => {
    return (
      <Space>
        <Button
          type="primary"
          icon={<SendOutlined />}
          onClick={handleStreamQuery}
          loading={loading}
          disabled={!selectedAgent || !query.trim()}
        >
          发送
        </Button>
        {isStreaming && (
          <Button
            danger
            icon={<StopOutlined />}
            onClick={handleStopStream}
          >
            停止
          </Button>
        )}
      </Space>
    );
  }, [selectedAgent, query, loading, isStreaming, handleStreamQuery, handleStopStream]);

  if (!selectedAgent) {
    return (
      <div className="empty-state">
        <Title level={4}>请从左侧选择一个代理</Title>
      </div>
    );
  }

  return (
    <div className="query-canvas">
      {/* 上半部分 - 查询区域 */}
      <div className="query-upper-section">
        <div className="agent-header">
          <Title level={4} style={{ margin: 0, fontWeight: 600 }}>
            {selectedAgent.agent}
          </Title>
          <Text type="secondary" style={{ fontSize: '14px', marginTop: '8px', display: 'block' }}>
            {selectedAgent.description}
          </Text>
        </div>
        
        <div className="query-input-container">
          <Input.TextArea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="请输入您的查询内容..."
            autoSize={{ minRows: 2, maxRows: 4 }}
            className="query-input"
            disabled={loading}
          />
          {/* 使用新的renderQueryButtons函数替换原来的按钮组 */}
          {renderQueryButtons()}
        </div>
      </div>

      {/* 使用渲染函数替换原来的下半部分 */}
      {renderResponseArea()}
    </div>
  );
};

export default QueryCanvas;