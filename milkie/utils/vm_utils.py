import logging
import os
from typing import Optional, List, Dict, Any

from milkie.config.config import VMConfig, GlobalConfig
from milkie.utils.security import SecurityUtils
from milkie.vm.vm import VMFactory, VM

logger = logging.getLogger(__name__)

class VMUtils:
    """VM工具类，提供便捷的VM相关操作功能"""
    
    @staticmethod
    def getVM(config: Optional[VMConfig] = None) -> VM:
        """获取VM实例
        
        Args:
            config: VM配置，如果为None则使用全局配置
            
        Returns:
            VM: VM实例
        """
        if config is None:
            # 从全局配置中获取VM配置
            try:
                globalConfig = GlobalConfig("")  # 使用空字符串，获取已初始化的实例
                config = globalConfig.getVMConfig()
            except Exception as e:
                logger.error(f"获取VM配置失败: {str(e)}")
                raise RuntimeError("未找到有效的VM配置")
        
        return VMFactory.createVM(config)
    
    @staticmethod
    def execBashCommand(command: str, config: Optional[VMConfig] = None) -> str:
        """便捷方法：在VM上执行Bash命令
        
        Args:
            command: 要执行的Bash命令
            config: 可选的VM配置
            
        Returns:
            str: 命令执行结果
        """
        with VMUtils.getVM(config) as vm:
            return vm.execBash(command)
    
    @staticmethod
    def execPythonCommand(command: str, varDict: Optional[Dict[str, Any]] = None, config: Optional[VMConfig] = None) -> str:
        """便捷方法：在VM上执行Python命令
        
        Args:
            command: 要执行的Python命令
            varDict: 可选的变量字典，会在执行代码前设置到Python环境中
            config: 可选的VM配置
            
        Returns:
            str: 命令执行结果
        """
        with VMUtils.getVM(config) as vm:
            return vm.execPython(command, varDict)
    
    @staticmethod
    def uploadFile(localPath: str, remotePath: str, config: Optional[VMConfig] = None) -> bool:
        """便捷方法：上传文件到VM
        
        Args:
            localPath: 本地文件路径
            remotePath: 远程文件路径
            config: 可选的VM配置
            
        Returns:
            bool: 上传是否成功
        """
        with VMUtils.getVM(config) as vm:
            return vm.uploadFile(localPath, remotePath)
    
    @staticmethod
    def downloadFile(remotePath: str, localPath: str, config: Optional[VMConfig] = None) -> bool:
        """便捷方法：从VM下载文件
        
        Args:
            remotePath: 远程文件路径
            localPath: 本地文件路径
            config: 可选的VM配置
            
        Returns:
            bool: 下载是否成功
        """
        with VMUtils.getVM(config) as vm:
            return vm.downloadFile(remotePath, localPath)
    
    @staticmethod
    def encryptPassword(password: str) -> str:
        """加密VM密码，用于配置文件
        
        Args:
            password: 明文密码
            
        Returns:
            str: 加密后的密码
        """
        encryptKey = SecurityUtils.get_env_password()
        return SecurityUtils.encrypt(password, encryptKey)
    
    @staticmethod
    def createVMConfig(
            host: str, 
            port: int, 
            username: str, 
            password: str, 
            isSSH: bool = True) -> VMConfig:
        """创建VM配置
        
        Args:
            host: 主机地址
            port: 端口
            username: 用户名
            password: 明文密码
            isSSH: 是否为SSH连接类型
            
        Returns:
            VMConfig: 创建的VM配置
        """
        from milkie.config.config import VMConnectionType
        
        # 加密密码
        encryptedPassword = VMUtils.encryptPassword(password)
        
        return VMConfig(
            connectionType=VMConnectionType.SSH if isSSH else VMConnectionType.DOCKER,
            host=host,
            port=port,
            username=username,
            encryptedPassword=encryptedPassword
        )
    
    @staticmethod
    def listFiles(remotePath: str, config: Optional[VMConfig] = None) -> List[str]:
        """列出远程目录中的文件
        
        Args:
            remotePath: 远程目录路径
            config: 可选的VM配置
            
        Returns:
            List[str]: 文件列表
        """
        command = f"ls -la {remotePath}"
        result = VMUtils.execBashCommand(command, config)
        return [line for line in result.strip().split('\n') if line]
    
    @staticmethod
    def checkCommandExists(command: str, config: Optional[VMConfig] = None) -> bool:
        """检查命令是否存在
        
        Args:
            command: 要检查的命令
            config: 可选的VM配置
            
        Returns:
            bool: 命令是否存在
        """
        checkCmd = f"which {command} > /dev/null 2>&1 && echo 'EXISTS' || echo 'NOT_EXISTS'"
        result = VMUtils.execBashCommand(checkCmd, config)
        return "EXISTS" in result 