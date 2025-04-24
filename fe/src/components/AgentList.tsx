import React, { useEffect, useState, useCallback } from 'react';
import { List, Card, Typography, Spin, message } from 'antd';
import { getAgentInfos, AgentInfo } from '../services/api';

const { Title } = Typography;

interface AgentListProps {
  onSelectAgent: (agent: AgentInfo) => void;
}

const AgentList: React.FC<AgentListProps> = ({ onSelectAgent }) => {
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [selectedAgentId, setSelectedAgentId] = useState<string>('');

  // 处理选择代理的事件
  const handleSelectAgent = useCallback((agent: AgentInfo) => {
    console.log('点击代理:', agent);
    setSelectedAgentId(agent.agent);
    onSelectAgent(agent);
  }, [onSelectAgent]);

  // 获取代理列表
  const fetchAgents = useCallback(async () => {
    try {
      setLoading(true);
      const response = await getAgentInfos();
      
      if (response.errno === 0) {
        setAgents(response.resp);
        // 自动选中第一个代理
        if (response.resp.length > 0 && !selectedAgentId) {
          handleSelectAgent(response.resp[0]);
        }
      } else {
        message.error(`获取代理列表失败: ${response.errmsg}`);
      }
    } catch (error) {
      message.error('获取代理列表出错');
      console.error('获取代理列表出错:', error);
    } finally {
      setLoading(false);
    }
  }, [handleSelectAgent, selectedAgentId]);

  // 组件挂载时获取代理列表
  useEffect(() => {
    fetchAgents();
  }, [fetchAgents]);

  return (
    <div className="agent-list">
      <Title level={3} className="agent-list-title">可用代理</Title>
      {loading ? (
        <div style={{ textAlign: 'center', padding: '20px' }}>
          <Spin tip="加载中..." />
        </div>
      ) : (
        <List
          dataSource={agents}
          renderItem={(agent) => (
            <List.Item 
              key={agent.agent}
              onClick={() => handleSelectAgent(agent)}
              className={selectedAgentId === agent.agent ? 'selected-item' : ''}
            >
              <Card 
                hoverable
                className={`agent-card ${selectedAgentId === agent.agent ? 'agent-card-selected' : ''}`}
              >
                <Card.Meta
                  title={agent.agent}
                  description={agent.description}
                />
              </Card>
            </List.Item>
          )}
        />
      )}
    </div>
  );
};

export default AgentList; 