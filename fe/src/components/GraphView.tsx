import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { Tree, Card, Typography, Tag, Button, Switch, message } from 'antd';

const { Text } = Typography;

// 导出GraphView相关的类型定义
export interface GraphNode {
  id: string;
  type: string;
  name?: string;
  query?: string;
  content?: any;
  children?: GraphNode[];
  [key: string]: any;
}

export interface TreeNode {
  key: string;
  type: string;
  name?: string;
  query?: string;
  level: number;
  children?: TreeNode[];
  [key: string]: any;
}

export interface TreeData {
  key: string;
  title: React.ReactNode;
  children?: TreeData[];
}

export interface GraphViewProps {
  graphData: any;
  onExpandKeysChange?: (keys: React.Key[]) => void;
}

// GraphView组件用于树形渲染执行图
const GraphView: React.FC<GraphViewProps> = ({ graphData, onExpandKeysChange }) => {
  const [expandedKeys, setExpandedKeys] = useState<React.Key[]>([]);
  const [autoExpand, setAutoExpand] = useState<boolean>(true);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const hasAutoExpandedRef = useRef<boolean>(false);
  const [renderKey, setRenderKey] = useState<number>(0);
  
  // 将expandedKeys的变化通知到父组件
  useEffect(() => {
    if (onExpandKeysChange) {
      onExpandKeysChange(expandedKeys);
    }
  }, [expandedKeys, onExpandKeysChange]);
  
  // 全屏切换事件
  const toggleFullscreen = useCallback(() => {
    setIsFullscreen(prev => !prev);
  }, []);
  
  // 添加CSS样式以居中节点和添加虚线连接
  useEffect(() => {
    const styleElement = document.createElement('style');
    styleElement.textContent = `
      /* 重置和强制覆盖所有Tree相关样式 */
      .centered-tree,
      .centered-tree * {
        box-sizing: border-box;
      }
      
      /* 容器样式 */
      .centered-tree {
        width: 100% !important;
        overflow: auto !important;
        text-align: center !important;
        padding: 20px 0 !important;
        position: relative !important;
      }
      
      /* 强制树节点居中 */
      .centered-tree .ant-tree {
        width: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        background: transparent !important;
      }
      
      .centered-tree .ant-tree-list {
        width: 100% !important;
        position: static !important;
        display: flex !important;
        justify-content: center !important;
      }
      
      .centered-tree .ant-tree-list-holder {
        width: 100% !important;
        position: static !important;
        display: flex !important;
        justify-content: center !important;
      }
      
      .centered-tree .ant-tree-list-holder-inner {
        position: static !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        width: auto !important;
        transform: none !important;
      }
      
      /* 树节点样式 */
      .centered-tree .ant-tree-treenode {
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
        padding: 8px 0 !important;
        position: relative !important;
      }
      
      .centered-tree .ant-tree-node-content-wrapper {
        position: relative !important;
        display: flex !important;
        justify-content: center !important;
        padding: 4px !important;
        background: transparent !important;
      }
      
      /* 显示切换器 */
      .centered-tree .ant-tree-switcher {
        display: inline-block !important;
        width: 24px !important;
        height: 24px !important;
        line-height: 24px !important;
        text-align: center !important;
        cursor: pointer !important;
        transition: all 0.3s !important;
      }
      
      /* 显示节点内容 */
      .centered-tree .ant-tree-node-content-wrapper:hover {
        background: rgba(0, 0, 0, 0.04) !important;
      }
      
      /* 完全去除左侧缩进 */
      .centered-tree .ant-tree-indent, 
      .centered-tree .ant-tree-indent-unit {
        display: none !important;
        width: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
      }
      
      /* 隐藏原生切换器，使用自定义的 */
      .centered-tree .ant-tree-switcher {
        display: none !important;
        width: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
      }
      
      /* 展开按钮样式 - 确保可见 */
      .tree-node-toggle-btn {
        position: absolute !important;
        top: 10px !important;
        right: 10px !important;
        width: 28px !important;
        height: 28px !important;
        border-radius: 50% !important;
        background-color: #f5f5f5 !important;
        border: 1px solid #d9d9d9 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        z-index: 100 !important;
        user-select: none !important;
        font-size: 14px !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
        opacity: 1 !important;
        visibility: visible !important;
      }
      
      /* 去除所有可能影响布局的元素 */
      .centered-tree .ant-tree-node-selected,
      .centered-tree .ant-tree-node-content-wrapper::before,
      .centered-tree .ant-tree-node-content-wrapper::after {
        display: none !important;
        background: transparent !important;
      }
      
      /* 增强线条样式 */
      .centered-tree .tree-node-line {
        position: absolute !important;
        border: 2px dashed #d9d9d9 !important;
        z-index: 0 !important;
        pointer-events: none !important;
      }
      
      /* 垂直连接线 */
      .centered-tree .tree-node-line-vertical {
        width: 0 !important;
        top: -50% !important;
        bottom: 50% !important;
        left: 50% !important;
        border-left: 2px dashed #d9d9d9 !important;
        border-right: 0 !important;
        border-top: 0 !important;
        border-bottom: 0 !important;
      }
      
      /* 卡片样式 */
      .tree-node-card {
        padding: 12px !important;
        margin: 8px 0 !important;
        background: #fff !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
        border: 1px solid #f0f0f0 !important;
        transition: all 0.3s !important;
        position: relative !important;
        width: 360px !important;
        display: inline-block !important;
        text-align: left !important;
        z-index: 5 !important;
      }
      
      /* 根节点卡片样式 */
      .tree-node-card-root {
        width: 420px !important;
      }
      
      /* 卡片hover效果 */
      .tree-node-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
        border-color: #91d5ff !important;
      }
      
      /* 展开按钮样式 */
      .tree-node-toggle-btn {
        position: absolute !important;
        top: 10px !important;
        right: 10px !important;
        width: 28px !important;
        height: 28px !important;
        border-radius: 50% !important;
        background-color: #f5f5f5 !important;
        border: 1px solid #d9d9d9 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        z-index: 100 !important;
        user-select: none !important;
        font-size: 14px !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
      }
      
      .tree-node-toggle-btn:hover {
        background-color: #e6f7ff !important;
        border-color: #91d5ff !important;
      }
      
      .tree-node-toggle-btn:active {
        background-color: #bae7ff !important;
      }
      
      /* 全屏模式样式 */
      .graph-card.fullscreen {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        bottom: 0 !important;
        z-index: 1000 !important;
        margin: 0 !important;
        border-radius: 0 !important;
        width: 100% !important;
        height: 100% !important;
      }
      
      .graph-card.fullscreen .centered-tree {
        max-height: calc(100vh - 70px) !important;
      }
      
      .fullscreen-icon {
        cursor: pointer !important;
        font-size: 16px !important;
        transition: all 0.3s !important;
      }
      
      .fullscreen-icon:hover {
        color: #1890ff !important;
      }
      
      /* 子树样式 */
      .centered-tree .ant-tree-child-tree {
        margin-top: 30px !important;
        width: 100% !important;
        position: relative !important;
      }
      
      /* 禁用所有动画和过渡 */
      .centered-tree .ant-tree-list-motion,
      .centered-tree .ant-tree-motion-collapse,
      .centered-tree .ant-motion-collapse,
      .centered-tree .ant-tree-treenode-motion {
        transition: none !important;
        animation: none !important;
        overflow: visible !important;
      }
    `;
    document.head.appendChild(styleElement);
    
    return () => {
      document.head.removeChild(styleElement);
    };
  }, []);
  
  // 处理全屏模式下的ESC键退出
  useEffect(() => {
    const handleEsc = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && isFullscreen) {
        setIsFullscreen(false);
      }
    };
    window.addEventListener('keydown', handleEsc);
    return () => {
      window.removeEventListener('keydown', handleEsc);
    };
  }, [isFullscreen]);
  
  // 修改处理图数据的函数，添加更强的错误处理
  const processGraphData = useCallback((data: any): any => {
    try {
      // 基本数据检查
      if (!data) {
        return null;
      }

      // 检查数据是否为字符串（可能是嵌套的JSON字符串）
      if (typeof data === 'string') {
        try {
          return processGraphData(JSON.parse(data));
        } catch (e) {
          // 如果无法解析为JSON，创建一个基本节点至少能显示出来
          return {
            id: 'root',
            type: 'ROOT',
            name: '解析错误',
            content: data.length > 200 ? data.substring(0, 200) + '...' : data,
            status: 'error'
          };
        }
      }
      
      // 确保返回一个有效的图结构
      if (!data || typeof data !== 'object') {
        return {
          id: 'root',
          type: 'ROOT',
          name: '无效数据',
          content: JSON.stringify(data),
          status: 'error'
        };
      }
      
      // 如果已经是标准格式
      if (data.id && data.type) {
        return data;
      }

      // 检查是否有errno/errmsg/resp结构 (API响应格式)
      if ('errno' in data && 'resp' in data) {
        if (data.errno === 0) {
          return processGraphData(data.resp);
        } else {
          console.error("[GraphView] API返回错误:", data.errmsg);
          return null;
        }
      }
      
      // 检查是否有graph字段
      if (data.graph) {
        return processGraphData(data.graph);
      }
      
      // 检查是否有resp字段
      if (data.resp) {
        if (typeof data.resp === 'string') {
          try {
            const parsed = JSON.parse(data.resp);
            return processGraphData(parsed);
          } catch (e) {
            console.error("[GraphView] 解析resp失败:", e);
            // 返回简单的包装结构，以便显示原始数据
            return {
              id: 'root',
              type: 'ROOT',
              name: '解析错误',
              content: data.resp.length > 200 ? data.resp.substring(0, 200) + '...' : data.resp,
              status: 'error'
            };
          }
        } else if (typeof data.resp === 'object' && data.resp !== null) {
          return processGraphData(data.resp);
        }
      }
      
      // 搜索任何包含id和type的子对象
      for (const key in data) {
        if (typeof data[key] === 'object' && data[key] !== null) {
          if (data[key].id && data[key].type) {
            return data[key];
          }
        }
      }
      
      // 尝试重建成有效的图结构
      // 如果有id但没有type，添加默认type
      if (data.id && !data.type) {
        return {
          ...data,
          type: 'UNKNOWN'
        };
      }
      
      // 如果数据不是标准格式但可能包含有用信息，创建根节点包装它
      return {
        id: 'root',
        type: 'ROOT',
        name: '根节点',
        // 如果是数组，直接使用作为children；否则尝试提取有意义的字段或整个对象作为单个子节点
        children: Array.isArray(data) ? 
          data.map((item, idx) => {
            // 为数组项添加缺少的id和type
            if (!item.id) {
              item = { ...item, id: `item_${idx}` };
            }
            if (!item.type) {
              item = { ...item, type: 'ITEM' };
            }
            return item;
          }) : 
          [
            {
              id: 'data_1',
              type: 'DATA',
              // 提取可能有意义的字段
              name: data.name || data.title || '数据节点',
              content: JSON.stringify(data, null, 2)
            }
          ]
      };
    } catch (e: any) {
      console.error("[GraphView] 处理图数据时发生严重错误:", e.message);
      // 发生错误时返回基本结构，避免渲染失败
      return {
        id: 'error_' + Date.now(),
        type: 'ERROR',
        name: '处理错误',
        content: e.message,
        status: 'error'
      };
    }
  }, []);

  // 添加强制刷新函数
  const forceRefresh = useCallback(() => {
    setRenderKey(prev => prev + 1);
  }, []);

  // 获取节点类型的颜色
  const getNodeTypeColor = (type: string) => {
    switch (type.toUpperCase()) {
      case 'ROOT': return '#1890ff';  // 蓝色
      case 'AGENT': return '#52c41a'; // 绿色
      case 'LLM': return '#722ed1';   // 紫色
      case 'SKILL': return '#52c41a'; // 技能调用 - 绿色
      case 'TOOL': return '#722ed1';  // 工具使用 - 紫色
      case 'DECISION': return '#faad14'; // 决策分析 - 橙色
      case 'THOUGHT': return '#1890ff';  // 思考过程 - 蓝色
      default: return '#faad14';      // 默认橙色
    }
  };

  // 处理展开/收起逻辑
  const toggleExpand = useCallback((nodeId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    
    const isExpanded = expandedKeys.includes(nodeId);
    
    if (isExpanded) {
      // 如果节点已展开，则将其收起（从 expandedKeys 中移除）
      setExpandedKeys(prev => prev.filter(key => key !== nodeId));
    } else {
      // 如果节点未展开，则将其展开（添加到 expandedKeys 中）
      setExpandedKeys(prev => [...prev, nodeId]);
    }
  }, [expandedKeys]);

  // 判断节点是否展开
  const isNodeExpanded = useCallback((nodeId: string) => {
    return expandedKeys.includes(nodeId);
  }, [expandedKeys]);

  // 监听expandedKeys变化，强制重新渲染树组件
  useEffect(() => {
    // 使用onExpandKeysChange回调通知父组件
    if (onExpandKeysChange) {
      onExpandKeysChange(expandedKeys);
    }
  }, [expandedKeys, onExpandKeysChange]);

  // 收集所有节点ID的函数
  const collectAllNodeIds = useCallback((node: any): string[] => {
    if (!node) return [];
    let ids: string[] = [];
    if (node.id) {
      ids.push(String(node.id));
    }
    if (node.children && Array.isArray(node.children)) {
      node.children.forEach((child: any) => {
        ids = [...ids, ...collectAllNodeIds(child)];
      });
    }
    return ids;
  }, []);

  // 初始化时展开所有节点
  useEffect(() => {
    if (graphData && !hasAutoExpandedRef.current) {
      const allNodeIds = collectAllNodeIds(processGraphData(graphData));
      setExpandedKeys(allNodeIds);
      hasAutoExpandedRef.current = true;
    }
  }, [graphData, collectAllNodeIds]);

  // 自定义树节点渲染
  const renderTreeNode = (node: any, path: string = 'root', level: number = 0): any => {
    if (!node) {
      return null;
    }
    
    // 提取基本属性
    const nodeId = node.id || path;
    const nodeType = node.type || '未知';
    const nodeName = node.name;
    const nodeQuery = node.query;
    const nodeContent = node.content || node.data || node.result;
    // 针对不同类型提取不同的内容
    const nodeInstruct = node.instruct;
    const nodeStatus = node.status;
    
    // 检查是否有子节点
    const hasChildren = node.children && Array.isArray(node.children) && node.children.length > 0;
    
    // 根据节点状态设置样式
    let borderColor = '#f0f0f0';
    let statusColor = '#999';
    let statusText = '';
    
    if (nodeStatus) {
      switch(nodeStatus.toLowerCase()) {
        case 'running':
          borderColor = '#1890ff';
          statusColor = '#1890ff';
          statusText = '执行中';
          break;
        case 'success':
        case 'completed':
          borderColor = '#52c41a';
          statusColor = '#52c41a';
          statusText = '完成';
          break;
        case 'error':
        case 'failed':
          borderColor = '#f5222d';
          statusColor = '#f5222d';
          statusText = '失败';
          break;
        default:
          statusText = nodeStatus;
      }
    }
    
    // 使用卡片形式直接展示节点内容
    return {
      key: nodeId,
      title: (
        <div 
          className={`tree-node-card ${level === 0 ? 'tree-node-card-root' : ''}`}
          data-node-id={nodeId}
          data-node-level={level}
          data-node-type={nodeType}
          style={{ 
            borderColor: borderColor,
            animation: 'highlight-update 2s ease-out' // 添加高亮动画
          }}
        >
          {/* 只对有子节点的节点显示展开/收起按钮 */}
          {hasChildren && (
            <div 
              className="tree-node-toggle-btn"
              onClick={(e) => {
                toggleExpand(nodeId, e);
              }}
              onMouseDown={(e) => {
                e.stopPropagation();
                e.preventDefault();
              }}
              style={{
                position: 'absolute',
                top: '10px',
                right: '10px',
                width: '28px',
                height: '28px',
                borderRadius: '50%',
                backgroundColor: '#f5f5f5',
                border: '1px solid #d9d9d9',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                zIndex: 100,
                userSelect: 'none',
                fontSize: '14px',
                boxShadow: '0 2px 5px rgba(0,0,0,0.1)'
              }}
              role="button"
              tabIndex={0}
              title={isNodeExpanded(nodeId) ? "收起卡片" : "展开卡片"}
              aria-label={isNodeExpanded(nodeId) ? "收起卡片" : "展开卡片"}
            >
              {isNodeExpanded(nodeId) ? "▲" : "▼"}
            </div>
          )}
          
          {/* 连接线 - 在根节点下方的子节点添加 */}
          {level > 0 && (
            <div className="tree-node-line tree-node-line-vertical"></div>
          )}
          
          {/* 卡片头部 - 类型和名称 */}
          <div style={{
            display: 'flex', 
            alignItems: 'center', 
            gap: '8px',
            marginBottom: '8px',
            borderBottom: '1px solid #f0f0f0',
            paddingBottom: '8px',
            paddingRight: '30px' // 为下拉按钮留出空间
          }}>
            <Tag color={getNodeTypeColor(nodeType)} style={{ margin: 0, padding: '2px 8px' }}>
              {nodeType}
            </Tag>
            {nodeName && (
              <Tag color="green" style={{ margin: 0, padding: '2px 8px' }}>
                {nodeName}
              </Tag>
            )}
            {statusText && (
              <Tag color={statusColor} style={{ margin: 0, padding: '2px 8px' }}>
                {statusText}
              </Tag>
            )}
            <div style={{ marginLeft: 'auto', fontSize: '12px', color: '#999' }}>
              {hasChildren ? 
                `${node.children.length}个子节点` : ""}
            </div>
          </div>
          
          {/* 节点ID */}
          <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px' }}>
            <Text strong style={{ marginRight: '4px' }}>ID:</Text>
            <Text copyable={{ text: nodeId }} style={{ wordBreak: 'break-word' }}>
              {nodeId.length > 20 ? `${nodeId.substring(0, 20)}...` : nodeId}
            </Text>
          </div>
          
          {/* 查询内容 */}
          {nodeQuery && (
            <div style={{ marginBottom: '8px' }}>
              <Text strong style={{ fontSize: '12px', color: '#333' }}>查询:</Text>
              <div style={{ 
                background: '#f5f5f5', 
                padding: '8px', 
                borderRadius: '4px',
                fontSize: '12px',
                marginTop: '4px',
                overflow: 'auto'
              }}>
                {nodeQuery}
              </div>
            </div>
          )}
          
          {/* 根据节点类型显示不同的内容 */}
          {(() => {
            switch(nodeType.toUpperCase()) {
              case 'ROOT':
                return (
                  <div>
                    <Text strong style={{ fontSize: '12px', color: '#333' }}>根节点内容:</Text>
                    {nodeContent && (
                      <div style={{ 
                        background: '#f5f5f5', 
                        padding: '8px', 
                        borderRadius: '4px',
                        fontSize: '12px',
                        marginTop: '4px',
                        overflow: 'auto'
                      }}>
                        {typeof nodeContent === 'object' 
                          ? JSON.stringify(nodeContent, null, 2)
                          : nodeContent}
                      </div>
                    )}
                  </div>
                );
              
              case 'LLM':
                return (
                  <>
                    {nodeInstruct && (
                      <div style={{ marginBottom: '8px' }}>
                        <Text strong style={{ fontSize: '12px', color: '#333' }}>指令:</Text>
                        <div style={{ 
                          background: '#f5f5f5', 
                          padding: '8px', 
                          borderRadius: '4px',
                          fontSize: '12px',
                          marginTop: '4px',
                          overflow: 'auto'
                        }}>
                          {typeof nodeInstruct === 'object' 
                            ? JSON.stringify(nodeInstruct, null, 2)
                            : nodeInstruct}
                        </div>
                      </div>
                    )}
                    {nodeContent && (
                      <div>
                        <Text strong style={{ fontSize: '12px', color: '#333' }}>回复:</Text>
                        <div style={{ 
                          background: '#f5f5f5', 
                          padding: '8px', 
                          borderRadius: '4px',
                          fontSize: '12px',
                          marginTop: '4px',
                          overflow: 'auto'
                        }}>
                          {typeof nodeContent === 'object' 
                            ? JSON.stringify(nodeContent, null, 2)
                            : nodeContent}
                        </div>
                      </div>
                    )}
                  </>
                );
              
              case 'SKILL':
                return (
                  <>
                    {nodeQuery && (
                      <div style={{ marginBottom: '8px' }}>
                        <Text strong style={{ fontSize: '12px', color: '#333' }}>技能调用:</Text>
                        <div style={{ 
                          background: '#f5f5f5', 
                          padding: '8px', 
                          borderRadius: '4px',
                          fontSize: '12px',
                          marginTop: '4px',
                          overflow: 'auto'
                        }}>
                          {nodeQuery}
                        </div>
                      </div>
                    )}
                    {nodeContent && (
                      <div>
                        <Text strong style={{ fontSize: '12px', color: '#333' }}>执行结果:</Text>
                        <div style={{ 
                          background: '#f5f5f5', 
                          padding: '8px', 
                          borderRadius: '4px',
                          fontSize: '12px',
                          marginTop: '4px',
                          overflow: 'auto'
                        }}>
                          {typeof nodeContent === 'object' 
                            ? JSON.stringify(nodeContent, null, 2)
                            : nodeContent}
                        </div>
                      </div>
                    )}
                  </>
                );
              
              case 'TOOL':
                return (
                  <>
                    {nodeQuery && (
                      <div style={{ marginBottom: '8px' }}>
                        <Text strong style={{ fontSize: '12px', color: '#333' }}>工具命令:</Text>
                        <div style={{ 
                          background: '#f5f5f5', 
                          padding: '8px', 
                          borderRadius: '4px',
                          fontSize: '12px',
                          marginTop: '4px',
                          overflow: 'auto'
                        }}>
                          {nodeQuery}
                        </div>
                      </div>
                    )}
                    {nodeContent && (
                      <div>
                        <Text strong style={{ fontSize: '12px', color: '#333' }}>工具输出:</Text>
                        <div style={{ 
                          background: '#f5f5f5', 
                          padding: '8px', 
                          borderRadius: '4px',
                          fontSize: '12px',
                          marginTop: '4px',
                          overflow: 'auto'
                        }}>
                          {typeof nodeContent === 'object' 
                            ? JSON.stringify(nodeContent, null, 2)
                            : nodeContent}
                        </div>
                      </div>
                    )}
                  </>
                );
              
              case 'THOUGHT':
                return (
                  <div>
                    <Text strong style={{ fontSize: '12px', color: '#333' }}>思考过程:</Text>
                    {nodeContent && (
                      <div style={{ 
                        background: '#f5f5f5', 
                        padding: '8px', 
                        borderRadius: '4px',
                        fontSize: '12px',
                        marginTop: '4px',
                        overflow: 'auto'
                      }}>
                        {typeof nodeContent === 'object' 
                          ? JSON.stringify(nodeContent, null, 2)
                          : nodeContent}
                      </div>
                    )}
                  </div>
                );
              
              case 'DECISION':
                return (
                  <>
                    {nodeQuery && (
                      <div style={{ marginBottom: '8px' }}>
                        <Text strong style={{ fontSize: '12px', color: '#333' }}>决策依据:</Text>
                        <div style={{ 
                          background: '#f5f5f5', 
                          padding: '8px', 
                          borderRadius: '4px',
                          fontSize: '12px',
                          marginTop: '4px',
                          overflow: 'auto'
                        }}>
                          {nodeQuery}
                        </div>
                      </div>
                    )}
                    {nodeContent && (
                      <div>
                        <Text strong style={{ fontSize: '12px', color: '#333' }}>决策结果:</Text>
                        <div style={{ 
                          background: '#f5f5f5', 
                          padding: '8px', 
                          borderRadius: '4px',
                          fontSize: '12px',
                          marginTop: '4px',
                          overflow: 'auto'
                        }}>
                          {typeof nodeContent === 'object' 
                            ? JSON.stringify(nodeContent, null, 2)
                            : nodeContent}
                        </div>
                      </div>
                    )}
                  </>
                );
              
              default:
                return (
                  <div>
                    <Text strong style={{ fontSize: '12px', color: '#333' }}>内容:</Text>
                    {nodeContent && (
                      <div style={{ 
                        background: '#f5f5f5', 
                        padding: '8px', 
                        borderRadius: '4px',
                        fontSize: '12px',
                        marginTop: '4px',
                        overflow: 'auto'
                      }}>
                        {typeof nodeContent === 'object' 
                          ? JSON.stringify(nodeContent, null, 2)
                          : nodeContent}
                      </div>
                    )}
                  </div>
                );
            }
          })()}
          
          {/* 添加提示文本 */}
          <div style={{ 
            textAlign: 'center', 
            fontSize: '12px', 
            color: '#1890ff', 
            marginTop: '8px' 
          }}>
            点击右上角按钮展开/收起
          </div>
        </div>
      ),
      children: hasChildren ? node.children.map((child: any, idx: number) => 
            renderTreeNode(child, `${path}-${idx}`, level + 1)
          )
        : undefined
    };
  };

  // 使用记忆化减少重复计算
  const treeData = useMemo(() => {
    try {
      const processedData = processGraphData(graphData);
      if (!processedData) {
        return [];
      }
      
      return [renderTreeNode(processedData)];
    } catch (e) {
      console.error("[GraphView] 处理图数据时出错:", e);
      message.error("执行图渲染失败，请检查数据格式");
      return [];
    }
  }, [graphData, processGraphData, renderTreeNode]);

  return (
    <Card 
      className={`graph-card ${isFullscreen ? 'fullscreen' : ''}`}
      title={
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>执行图</span>
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <Switch
              size="small"
              checked={autoExpand}
              onChange={(checked) => {
                setAutoExpand(checked);
                if (checked) {
                  hasAutoExpandedRef.current = false;
                }
              }}
              style={{ marginRight: '8px' }}
            />
            <span style={{ marginRight: '16px', fontSize: '12px' }}>自动展开</span>
            
            <Button
              size="small"
              onClick={forceRefresh}
              style={{ marginRight: '8px' }}
              title="手动刷新图表"
            >
              刷新
            </Button>
            
            <span 
              className="fullscreen-icon" 
              onClick={toggleFullscreen}
              title={isFullscreen ? "退出全屏" : "全屏显示"}
            >
              {isFullscreen ? "⤓" : "⤢"}
            </span>
          </div>
        </div>
      } 
      style={{ marginBottom: isFullscreen ? 0 : '16px' }}
    >
      {treeData.length > 0 ? (
        <div className="centered-tree" 
          style={{ 
            maxHeight: isFullscreen ? 'calc(100vh - 70px)' : '800px'
          }}
        >
          <Tree
            key={renderKey}
            expandedKeys={expandedKeys}
            onExpand={(newExpandedKeys) => {
              setExpandedKeys(newExpandedKeys);
            }}
            treeData={treeData}
            showLine={false}
            selectable={false}
            className="centered-tree-instance"
            motion={false}
            blockNode
            autoExpandParent={true}
            defaultExpandParent={true}
            showIcon={false}
            virtual={false}
          />
        </div>
      ) : (
        <div style={{ padding: '30px', textAlign: 'center' }}>
          {graphData ? (
            <div>执行图数据处理失败</div>
          ) : (
            <div>暂无执行图数据</div>
          )}
        </div>
      )}
    </Card>
  );
};

export default GraphView; 