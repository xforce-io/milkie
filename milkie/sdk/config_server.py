class ConfigServer:
    def __init__(self, addr: str = "http://localhost:8123"):
        # 确保地址格式正确
        if not addr.startswith("http://") and not addr.startswith("https://"):
            addr = f"http://{addr}"
        self.addr = addr.rstrip("/")  # 移除末尾的斜杠
    
    def getAddr(self) -> str:
        return self.addr
