import { useState, useEffect, useCallback } from 'react'
import { Layout, Typography, Divider, Input, Button, Modal, Form, Tabs } from 'antd'
import AgentList from './components/AgentList'
import QueryCanvas from './components/QueryCanvas'
import ExecutionGraph from './components/ExecutionGraph'
import { AgentInfo, API_BASE_URL, getApiBaseUrl } from './services/api'
import 'reactflow/dist/style.css'
import './App.css' // 注释掉这一行

const { Header, Sider, Content } = Layout
const { Title, Text } = Typography

function App() {
  const [selectedAgent, setSelectedAgent] = useState<AgentInfo | null>(null)
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [apiUrl, setApiUrl] = useState(getApiBaseUrl())
  const [form] = Form.useForm()
  const [executionNodes, setExecutionNodes] = useState<any[]>([])
  const [activeTabKey, setActiveTabKey] = useState('1')

  useEffect(() => {
    // 只在模态框打开时初始化表单值
    if (isModalOpen) {
      form.setFieldsValue({ apiUrl: getApiBaseUrl() })
    }
  }, [form, isModalOpen])

  const handleSelectAgent = (agent: AgentInfo) => {
    console.log('选中代理:', agent);
    setSelectedAgent(agent)
  }

  const showModal = () => {
    setIsModalOpen(true)
  }

  const handleOk = () => {
    form.validateFields().then(values => {
      // 设置全局API URL
      window.API_BASE_URL = values.apiUrl
      setApiUrl(values.apiUrl)
      setIsModalOpen(false)
      // 强制重新加载页面应用新的API URL
      window.location.reload()
    })
  }

  const handleCancel = () => {
    setIsModalOpen(false)
  }

  const handleStreamResponse = (response: any) => {
    console.log('App收到流式响应数据:', response);
    
    // 检查数据有效性
    if (!response || typeof response !== 'object') {
      console.warn('无效的响应数据格式');
      return;
    }
    
    // 从响应中提取节点数据
    if (response.nodes && Array.isArray(response.nodes)) {
      const newNodes = response.nodes.filter((node: any) => node && node.id);
      
      // 只有当节点有变化时才更新状态
      if (newNodes.length > 0) {
        setExecutionNodes(prev => {
          // 检查是否与之前的节点相同
          if (JSON.stringify(prev) === JSON.stringify(newNodes)) {
            console.log('节点数据未变化，不更新状态');
            return prev;
          }
          console.log(`更新执行图: ${newNodes.length} 个节点`);
          return newNodes;
        });
      }
    }
  };

  const clearExecutionGraph = useCallback(() => {
    setExecutionNodes([]);
  }, []);

  // 使用items属性定义标签页内容
  const tabItems = [
    {
      key: '1',
      label: '查询',
      children: (
        <QueryCanvas 
          selectedAgent={selectedAgent} 
          onStreamResponse={handleStreamResponse}
          onClearGraph={clearExecutionGraph}
        />
      )
    },
    {
      key: '2',
      label: '执行图',
      children: (
        <div style={{ 
          width: '100%', 
          height: '100%', 
          minHeight: '650px',
          position: 'relative',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          background: '#f0f0f0',
          border: '1px solid #444',
          zIndex: 1  // 添加层级
        }}>
          {/* 只有当executionNodes不为空时才渲染图表 */}
          {executionNodes.length > 0 ? (
            <div style={{
              flex: 1,
              width: '100%',
              height: '100%',
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              minHeight: 650,
              display: 'flex'
            }}>
              <ExecutionGraph 
                key="execution-graph-singleton" 
                nodes={executionNodes} 
              />
            </div>
          ) : (
            <div style={{
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              height: '100%',
              flexDirection: 'column',
              color: '#666'
            }}>
              <div style={{ marginBottom: '16px', fontSize: '16px' }}>
                尚无执行图数据
              </div>
              <div style={{ fontSize: '14px' }}>
                请在查询完成后查看执行流程图
              </div>
            </div>
          )}
        </div>
      )
    }
  ]

  return (
    <Layout className="app-container">
      <Header className="app-header">
        <div className="header-content">
          <Title level={3} style={{ margin: 0 }}>Milkie 代理系统</Title>
          <Button type="primary" onClick={showModal}>配置</Button>
        </div>
      </Header>
      <Divider className="app-divider" />
      <Layout className="app-main-content">
        <Sider width={300} theme="light" className="app-sider">
          <AgentList onSelectAgent={handleSelectAgent} />
        </Sider>
        <Content className="app-content">
          <Tabs 
            activeKey={activeTabKey} 
            onChange={setActiveTabKey}
            items={tabItems} 
          />
        </Content>
      </Layout>

      {/* 配置模态框 */}
      <Modal 
        title="系统配置" 
        open={isModalOpen} 
        onOk={handleOk} 
        onCancel={handleCancel}
        destroyOnClose
      >
        <Form form={form} layout="vertical">
          <Form.Item 
            name="apiUrl" 
            label="API 服务器地址" 
            initialValue={apiUrl}
            rules={[{ required: true, message: '请输入API服务器地址' }]}
          >
            <Input placeholder="例如: http://localhost:8000" />
          </Form.Item>
          <div>
            <Text type="secondary">当前API地址: {apiUrl}</Text>
          </div>
        </Form>
      </Modal>
    </Layout>
  )
}

export default App
