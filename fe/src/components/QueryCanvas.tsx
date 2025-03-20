import React, { useState, useEffect, useRef } from 'react';
import { Input, Button, Typography, Card, Spin, message } from 'antd';
import { SendOutlined, StopOutlined } from '@ant-design/icons';
import { queryAgent, streamQueryAgent, AgentInfo } from '../services/api';

const { Title, Text } = Typography;

interface QueryCanvasProps {
  selectedAgent: AgentInfo | null;
}

const QueryCanvas: React.FC<QueryCanvasProps> = ({ selectedAgent }) => {
  const [query, setQuery] = useState<string>('');
  const [response, setResponse] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const closeStreamRef = useRef<(() => void) | null>(null);

  // 组件卸载时清理资源
  useEffect(() => {
    return () => {
      if (closeStreamRef.current) {
        closeStreamRef.current();
      }
    };
  }, []);

  // 处理常规查询
  const handleSendQuery = async () => {
    if (!selectedAgent) {
      message.warning('请先选择一个代理');
      return;
    }

    if (!query.trim()) {
      message.warning('请输入查询内容');
      return;
    }

    try {
      setLoading(true);
      const result = await queryAgent(selectedAgent.agent, query);
      
      if (result.errno === 0) {
        setResponse(result.resp);
      } else {
        message.error(`查询失败: ${result.errmsg}`);
      }
    } catch (error) {
      message.error('执行查询出错');
      console.error('执行查询出错:', error);
    } finally {
      setLoading(false);
    }
  };

  // 处理流式查询
  const handleStreamQuery = () => {
    if (!selectedAgent) {
      message.warning('请先选择一个代理');
      return;
    }

    if (!query.trim()) {
      message.warning('请输入查询内容');
      return;
    }

    // 清空当前响应
    setResponse('');
    setIsStreaming(true);
    setLoading(true);

    // 启动流式查询
    const closeStream = streamQueryAgent(
      selectedAgent.agent,
      query,
      // 处理数据流
      (data) => {
        setResponse((prev) => prev + data);
      },
      // 处理错误
      (error) => {
        message.error('流式查询出错');
        setIsStreaming(false);
        setLoading(false);
      },
      // 处理完成
      () => {
        setIsStreaming(false);
        setLoading(false);
        closeStreamRef.current = null;
      }
    );

    // 保存关闭函数以便后续使用
    closeStreamRef.current = closeStream;
  };

  // 停止流式查询
  const handleStopStream = () => {
    if (closeStreamRef.current) {
      closeStreamRef.current();
      closeStreamRef.current = null;
      setIsStreaming(false);
      setLoading(false);
      message.info('已停止流式查询');
    }
  };

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
            autoSize={{ minRows: 3, maxRows: 6 }}
            className="query-input"
            disabled={loading}
          />
          <div className="query-buttons">
            {isStreaming ? (
              <Button
                danger
                icon={<StopOutlined />}
                onClick={handleStopStream}
                className="query-button"
              >
                停止
              </Button>
            ) : (
              <>
                <Button
                  type="primary"
                  icon={<SendOutlined />}
                  onClick={handleStreamQuery}
                  loading={loading}
                  className="query-button"
                >
                  流式查询
                </Button>
                <Button
                  onClick={handleSendQuery}
                  loading={loading}
                  className="query-button"
                >
                  普通查询
                </Button>
              </>
            )}
          </div>
        </div>
      </div>

      {/* 下半部分 - 响应区域 */}
      <div className="query-lower-section">
        <Card 
          title="响应结果" 
          className="response-card"
          extra={
            loading && !isStreaming ? <Spin size="small" /> : null
          }
        >
          {loading && !isStreaming && !response ? (
            <div className="loading-container">
              <Spin tip="正在执行查询..." />
            </div>
          ) : (
            <pre className="response-pre">
              {response || '执行结果将显示在这里'}
              {isStreaming && (
                <span className="streaming-cursor" />
              )}
            </pre>
          )}
        </Card>
      </div>
    </div>
  );
};

export default QueryCanvas; 