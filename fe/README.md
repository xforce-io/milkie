# Milkie 代理系统前端

这是Milkie代理系统的前端界面，用于访问和使用各种代理服务。

## 功能特性

- 显示所有可用的代理列表
- 允许用户选择特定代理并发送查询
- 实时展示查询结果
- 响应式界面设计

## 快速开始

### 安装依赖

```bash
npm install
```

### 开发模式运行

```bash
npm run dev
```

### 构建生产版本

```bash
npm run build
```

### 运行预览版本

```bash
npm run preview
```

## 使用说明

1. 启动后端服务（确保后端服务运行在 http://localhost:8000）
2. 启动前端应用
3. 从左侧列表选择一个代理
4. 在右侧查询框中输入查询内容
5. 点击"发送查询"按钮执行查询
6. 查看执行结果

## 配置说明

默认后端API地址为 `http://localhost:8000`，如需修改，请在 `src/services/api.ts` 中更新 `API_BASE_URL` 常量。

## 技术栈

- React
- TypeScript
- Ant Design
- Axios
