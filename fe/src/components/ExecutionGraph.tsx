import React, { useState, useEffect, useCallback, useRef, CSSProperties } from 'react';
import ReactFlow, { 
  Node, 
  Edge, 
  Background, 
  Controls, 
  BackgroundVariant,
  Position,
  Handle,
  MiniMap,
  ReactFlowInstance,
  ReactFlowProvider,
  useNodesState,
  useEdgesState
} from 'reactflow';
import dagre from 'dagre';
import 'reactflow/dist/style.css';
import './ExecutionGraph.css'; // 注释掉这一行

// 全局声明
declare global {
  interface Window {
    _reactFlowInstance: any;
  }
}

// 执行流程节点接口
interface ExecutionNode {
  id: string;
  type: string;
  label?: string;
  query?: string;
  agent?: string;
  name?: string;
  skillName?: string;
  called?: string;
  instructions?: string[];
  skills?: string[];
  content?: any;  // 添加 content 字段
}

// 流程图节点接口
interface FlowNode extends ExecutionNode {
  level: number;      // 节点在流程图中的层级
  children: string[]; // 子节点ID列表
  parents: string[];  // 父节点ID列表
  index?: number;     // 同一层级内的索引位置
}


// 组件属性
interface ExecutionGraphProps {
  nodes: ExecutionNode[];
}

// 节点样式常量
const NODE_STYLE = {
  width: 280,    // 增加节点尺寸
  height: 160,   // 增加节点尺寸
  padding: 15    // 内边距，单位像素
};

// 使用dagre布局优化连接
const getLayoutedElements = (nodes: Node[], edges: Edge[], direction: 'TB' | 'LR' = 'TB') => {
  if (!nodes.length) return { nodes, edges };
  
  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));
  
  // 增加布局参数
  dagreGraph.setGraph({ 
    rankdir: direction,
    align: 'UL',
    nodesep: 100,
    ranksep: 100,
    edgesep: 50,
    marginx: 20,
    marginy: 20
  });

  // 设置节点大小
  nodes.forEach((node) => {
    dagreGraph.setNode(node.id, { 
      width: NODE_STYLE.width, 
      height: NODE_STYLE.height,
      label: node.data?.label || node.id
    });
  });

  // 添加边
  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });

  // 计算布局
  dagre.layout(dagreGraph);

  // 获取布局后的节点位置
  const layoutedNodes = nodes.map((node) => {
    const nodeWithPosition = dagreGraph.node(node.id);
    return {
      ...node,
      position: { 
        x: nodeWithPosition.x - NODE_STYLE.width / 2 + 200,  // 在 dagre 计算结果上增加 x 偏移
        y: nodeWithPosition.y - NODE_STYLE.height / 2 + 200 // 在 dagre 计算结果上增加 y 偏移
      },
      style: {
        ...node.style,
        width: NODE_STYLE.width,
        height: NODE_STYLE.height,
        opacity: 1,
        visibility: 'visible' as const
      }
    };
  });

  return { nodes: layoutedNodes, edges };
};

// 获取节点颜色 - 为不同类型节点设置不同颜色，提高亮度和对比度
const getNodeColor = (node: ExecutionNode): string => {
  const colors: Record<string, string> = {
    ROOT: '#4D3B70',     // 紫色
    AGENT: '#3B6880',    // 青灰色
    LLM: '#6146AD',      // 紫色
    SKILL: '#3B7A99',    // 青色
    TOOL: '#994D4D',     // 红褐色
    COMMON: '#3B5299',   // 蓝色
    CALL: '#4D7A3B',     // 绿色
    SEQUENCE: '#8C7A3B'  // 棕色
  };

  return colors[node.type] || colors[node.label || ''] || '#555555';
};

// 获取节点边框颜色，使用更亮的颜色
const getNodeBorderColor = (node: ExecutionNode): string => {
  const colors: Record<string, string> = {
    ROOT: '#E5D1FF',     // 更亮的紫色
    AGENT: '#C4E8FF',    // 更亮的蓝色
    LLM: '#F2C4FF',      // 更亮的亮紫色 
    SKILL: '#C4FFFA',    // 更亮的青色
    TOOL: '#FFC4B8',     // 更亮的橙红色
    COMMON: '#C4D9FF',   // 更亮的浅蓝色
    CALL: '#C4FFC4',     // 更亮的绿色
    SEQUENCE: '#FFF2C4'  // 更亮的黄色
  };

  return colors[node.type] || colors[node.label || ''] || '#E5D1FF';
};

// 获取节点标签
const getNodeLabel = (node: ExecutionNode): React.ReactNode => {
  const borderColor = getNodeBorderColor(node);
  
  return (
    <div style={{ 
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      textAlign: 'center',
      height: '100%',
      padding: '4px'
    }}>
      {/* 节点类型标签 */}
      <div style={{ 
        fontSize: '12px', 
        fontWeight: 'bold',
        backgroundColor: `rgba(${borderColor.replace(/[^\d,]/g, '')}, 0.2)`,
        padding: '2px 6px',
        borderRadius: '10px',
        marginBottom: '8px',
        textTransform: 'uppercase',
        color: borderColor
      }}>
        {node.type || node.label || '未知类型'}
      </div>
      
      {/* 节点名称 */}
      {node.name && (
        <div style={{ 
          fontSize: '14px',
          fontWeight: 'bold',
          marginBottom: '8px',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          width: '90%',
          whiteSpace: 'nowrap',
          color: borderColor
        }}>
          {node.name}
        </div>
      )}
      
      {/* 简短预览 */}
      {node.query && (
        <div style={{ 
          fontSize: '12px',
          opacity: 0.9,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          display: '-webkit-box',
          WebkitLineClamp: 2,
          WebkitBoxOrient: 'vertical',
          width: '90%',
          lineHeight: 1.4,
          color: '#fff'
        }}>
          {node.query}
        </div>
      )}
    </div>
  );
};

// 自定义节点组件
const CustomNode: React.FC<{ data: any }> = ({ data }) => {
  const node = data.original || {};
  const [isHovered, setIsHovered] = useState(false);
  const borderColor = getNodeBorderColor(node);
  
  return (
    <div 
      style={{
        padding: NODE_STYLE.padding,
        background: getNodeColor(node),
        borderRadius: '8px',
        width: NODE_STYLE.width,
        height: NODE_STYLE.height,
        color: '#fff',
        boxShadow: isHovered 
          ? `0 0 30px ${borderColor}` 
          : `0 0 15px rgba(0,0,0,0.3)`,  // 增强阴影效果
        border: `3px solid ${isHovered ? borderColor : `${borderColor}`}`,
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        transition: 'all 0.3s ease',
        transform: isHovered ? 'translateY(-4px) scale(1.02)' : 'none',
        cursor: 'pointer',
        position: 'relative',
        zIndex: 20  // 增加z-index确保可见
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {getNodeLabel(node)}
      <Handle 
        type="target" 
        position={Position.Top} 
        style={{ 
          background: borderColor, 
          width: '12px', 
          height: '12px', 
          border: `2px solid ${borderColor}`,
          boxShadow: `0 0 10px ${borderColor}`
        }} 
      />
      <Handle 
        type="source" 
        position={Position.Bottom} 
        style={{ 
          background: borderColor, 
          width: '12px', 
          height: '12px', 
          border: `2px solid ${borderColor}`,
          boxShadow: `0 0 10px ${borderColor}`
        }} 
      />
    </div>
  );
};

// 将nodeTypes定义移到组件外部
const NODE_TYPES = {
  custom: CustomNode
} as const;

// 节点详情弹窗组件
const NodeDetailsModal: React.FC<{
  node: ExecutionNode | null;
  onClose: () => void;
}> = ({ node, onClose }) => {
  if (!node) return null;
  const borderColor = getNodeBorderColor(node);

  return (
    <div style={{
      position: 'fixed',
      top: '50%',
      left: '50%',
      transform: 'translate(-50%, -50%)',
      backgroundColor: '#1a1a1a',
      padding: '20px',
      borderRadius: '8px',
      boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
      zIndex: 1000,
      maxWidth: '600px',
      width: '90%',
      maxHeight: '80vh',
      overflow: 'auto',
      color: '#fff',
      border: `1px solid ${borderColor}`
    }}>
      {/* 关闭按钮 */}
      <div style={{
        position: 'absolute',
        top: '10px',
        right: '10px',
        cursor: 'pointer',
        fontSize: '20px',
        color: '#fff'
      }} onClick={onClose}>
        ×
      </div>

      {/* 节点详情内容 */}
      <div style={{
        backgroundColor: getNodeColor(node),
        color: '#fff',
        padding: '6px 12px',
        borderRadius: '4px',
        display: 'inline-block',
        marginBottom: '16px',
        border: `1px solid ${borderColor}`
      }}>
        {node.type || node.label}
      </div>

      {/* 节点ID */}
      <div style={{ marginBottom: '12px' }}>
        <div style={{ color: '#aaa', fontSize: '12px' }}>ID</div>
        <div style={{ wordBreak: 'break-all' }}>{node.id}</div>
      </div>

      {/* 显示其他节点信息 */}
      {node.name && (
        <div style={{ marginBottom: '12px' }}>
          <div style={{ color: '#aaa', fontSize: '12px' }}>名称</div>
          <div style={{ fontSize: '16px', fontWeight: 'bold', color: borderColor }}>{node.name}</div>
        </div>
      )}

      {node.query && (
        <div style={{ marginBottom: '12px' }}>
          <div style={{ color: '#aaa', fontSize: '12px' }}>查询</div>
          <div style={{ backgroundColor: '#2a2a2a', padding: '12px', borderRadius: '4px', whiteSpace: 'pre-wrap' }}>
            {node.query}
          </div>
        </div>
      )}

      {/* 显示节点内容 */}
      {node.content && (
        <div style={{ marginBottom: '12px' }}>
          <div style={{ color: '#aaa', fontSize: '12px' }}>内容</div>
          <div style={{ backgroundColor: '#2a2a2a', padding: '12px', borderRadius: '4px', overflow: 'auto' }}>
            <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
              {typeof node.content === 'string' ? node.content : JSON.stringify(node.content, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
};

// 主组件
const ExecutionGraph: React.FC<ExecutionGraphProps> = ({ nodes: inputNodes }) => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedNode, setSelectedNode] = useState<ExecutionNode | null>(null);
  const [debugInfo, setDebugInfo] = useState<string>("");
  const reactFlowInstance = useRef<ReactFlowInstance | null>(null);

  useEffect(() => {
    if (!inputNodes || inputNodes.length === 0) {
      setNodes([]);
      setEdges([]);
      setDebugInfo("无输入节点数据");
      return;
    }

    const nodeMap: Record<string, FlowNode> = {};
    inputNodes.forEach(node => {
      nodeMap[node.id] = { ...node, level: 0, children: [], parents: [] };
    });

    // 构建边
    const edges: Edge[] = [];
    let edgeIndex = 0;

    inputNodes.forEach(source => {
      if (source.type === 'ROOT' && source.agent) {
        edges.push({
          id: `edge-${edgeIndex++}`,
          source: source.id,
          target: source.agent,
          type: 'smoothstep',
          animated: true,
          style: { stroke: '#b1b1b7', strokeWidth: 2 }
        });
      }
      if (source.type === 'SEQUENCE' && source.instructions) {
        source.instructions.forEach(targetId => {
          edges.push({
            id: `edge-${edgeIndex++}`,
            source: source.id,
            target: targetId,
            type: 'smoothstep',
            animated: true,
            style: { stroke: '#b1b1b7', strokeWidth: 2 }
          });
        });
      }
    });

    // 创建节点
    const nodes: Node[] = inputNodes.map((node, index) => ({
      id: node.id,
      type: 'custom',
      position: { x: 250 + index * 50, y: 150 + index * 50 }, // 每个节点错开一定距离
      data: {
        label: node.name || node.id,
        original: node
      }
    }));

    // 恢复 dagre 布局
    const { nodes: layoutedNodes, edges: layoutedEdges } = getLayoutedElements(nodes, edges);
    
    setNodes(layoutedNodes); // 使用布局后的节点
    setEdges(layoutedEdges); // 使用布局后的边（虽然此例中边未被布局函数修改，但保持一致性）
    setDebugInfo(`已更新 ${layoutedNodes.length} 个节点`);

  }, [inputNodes, setNodes, setEdges]);

  const onInit = useCallback((instance: ReactFlowInstance) => {
    reactFlowInstance.current = instance;
    window._reactFlowInstance = instance;
  }, []);

  return (
    <div className="execution-graph-container">
      <ReactFlowProvider>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onInit={onInit}
          onNodeClick={(_, node) => setSelectedNode(node.data?.original || null)}
          nodeTypes={NODE_TYPES}
          style={{ width: '100%', height: '100%' }}
        >
          <Background color="#aaa" gap={20} size={2} variant={BackgroundVariant.Dots} />
          <Controls />
          <MiniMap nodeStrokeWidth={3} maskColor="rgba(0, 0, 0, 0.7)" />
        </ReactFlow>
      </ReactFlowProvider>
      {debugInfo && (
        <div style={{ position: 'absolute', bottom: 10, left: 10, backgroundColor: 'rgba(20,20,20,0.8)', color: '#bbb', padding: '8px 12px', borderRadius: '4px', fontSize: '12px', zIndex: 999 }}>
          {debugInfo}
        </div>
      )}
      {selectedNode && <NodeDetailsModal node={selectedNode} onClose={() => setSelectedNode(null)} />}
    </div>
  );
};

export default ExecutionGraph;