import { useState, useEffect } from 'react'
import { Layout, Typography, Divider, Input, Button, Modal, Form } from 'antd'
import AgentList from './components/AgentList'
import QueryCanvas from './components/QueryCanvas'
import { AgentInfo, API_BASE_URL, getApiBaseUrl } from './services/api'
import './App.css'

const { Header, Sider, Content } = Layout
const { Title, Text } = Typography

function App() {
  const [selectedAgent, setSelectedAgent] = useState<AgentInfo | null>(null)
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [apiUrl, setApiUrl] = useState(getApiBaseUrl())
  const [form] = Form.useForm()

  useEffect(() => {
    // 组件挂载时，初始化表单值
    form.setFieldsValue({ apiUrl: getApiBaseUrl() })
  }, [form])

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
          <QueryCanvas selectedAgent={selectedAgent} />
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
