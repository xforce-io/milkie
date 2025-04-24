from abc import abstractmethod
import ast
import os
import paramiko
import logging
from typing import Any, Dict, Optional, Tuple, List
import tempfile
import random
import string
import reprlib

from milkie.config.config import VMConfig, VMConnectionType
from milkie.functions.import_white_list import PreImport
from milkie.utils.security import SecurityUtils

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VM:
    """虚拟机基类，定义了虚拟机操作的接口"""

    @abstractmethod
    def connect(self) -> bool:
        """连接到虚拟机
        
        Returns:
            bool: 连接是否成功
        """
        pass

    @abstractmethod
    def execBash(self, command: str) -> str:
        """执行Bash命令
        
        Args:
            command: 要执行的Bash命令
            
        Returns:
            str: 命令执行结果
        """
        pass

    def execPython(
            self, 
            code: str, 
            varDict: Optional[Dict[str, Any]] = None, 
            **kwargs) -> str:
        """执行Python命令
        
        Args:
            code: 要执行的Python代码
            varDict: 可选的变量字典，会在执行代码前设置到Python环境中
            
        Returns:
            str: 命令执行结果
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """断开与虚拟机的连接"""
        pass

    @staticmethod
    def deserializePythonResult(result: str) -> Any:
        if not result:
            return result
        
        if result[0] == '[' or result[0] == '{':
            return ast.literal_eval(result)
        else:
            return result
           

class VMSSH(VM):
    """基于SSH协议的虚拟机连接实现"""
    
    def __init__(self, config: VMConfig):
        """初始化SSH连接
        
        Args:
            config: 包含SSH连接信息的配置对象
        """
        self.config = config
        self.client = None
        self.connected = False
        self.attempt_count = 0
        
    def connect(self) -> bool:
        """使用Paramiko建立SSH连接
        
        支持密码认证和SSH密钥认证
        
        Returns:
            bool: 连接是否成功
        """
        if not self.config.validate():
            logger.error("VM配置验证失败")
            return False
            
        # 重置连接尝试次数
        self.attempt_count = 0
        return self._attempt_connect()
    
    def _attempt_connect(self) -> bool:
        """尝试连接，支持重试
        
        Returns:
            bool: 连接是否成功
        """
        while self.attempt_count < self.config.retryCount:
            try:
                self.attempt_count += 1
                logger.info(f"正在连接到 {self.config.host}:{self.config.port} (尝试 {self.attempt_count}/{self.config.retryCount})")
                
                self.client = paramiko.SSHClient()
                self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                
                connect_args = {
                    "hostname": self.config.host,
                    "port": self.config.port,
                    "username": self.config.username,
                    "timeout": self.config.timeout
                }
                
                # 优先使用SSH密钥认证
                if self.config.sshKeyPath and os.path.exists(self.config.sshKeyPath):
                    logger.info(f"使用SSH密钥认证: {self.config.sshKeyPath}")
                    connect_args["key_filename"] = self.config.sshKeyPath
                else:
                    # 使用密码认证
                    logger.info("使用密码认证")
                    password = self._decryptPassword(self.config.encryptedPassword)
                    connect_args["password"] = password
                
                self.client.connect(**connect_args)
                self.connected = True
                logger.info(f"成功连接到 {self.config.host}")
                return True
                
            except Exception as e:
                logger.warning(f"连接失败 (尝试 {self.attempt_count}/{self.config.retryCount}): {str(e)}")
                if self.client:
                    self.client.close()
                    self.client = None
                
                # 如果已经达到最大重试次数，返回失败
                if self.attempt_count >= self.config.retryCount:
                    logger.error(f"连接失败，已达到最大重试次数 ({self.config.retryCount})")
                    self.connected = False
                    return False
                
                # 等待一段时间后重试
                import time
                time.sleep(2)  # 等待2秒后重试
        
        return False
    
    def _decryptPassword(self, encryptedPassword: str) -> str:
        """解密存储的加密密码
        
        Args:
            encryptedPassword: 加密后的密码
            
        Returns:
            str: 解密后的密码
        """
        try:
            # 从环境变量获取密码解密的密钥
            password = SecurityUtils.get_env_password()
            # 解密保存的加密密码
            return SecurityUtils.decrypt(encryptedPassword, password)
        except Exception as e:
            logger.error(f"密码解密失败: {str(e)}")
            # 如果解密失败，可能是因为密码未加密，直接返回
            return encryptedPassword
    
    def _checkConnection(self) -> bool:
        """检查连接状态，如果未连接则尝试连接
        
        Returns:
            bool: 连接是否可用
        """
        if not self.connected or self.client is None:
            return self.connect()
            
        # 测试连接是否还活着
        try:
            transport = self.client.get_transport()
            if transport is None or not transport.is_active():
                logger.warning("SSH连接已断开，尝试重新连接")
                self.disconnect()
                return self.connect()
            return True
        except Exception as e:
            logger.warning(f"检查连接状态时出错: {str(e)}")
            self.disconnect()
            return self.connect()
    
    def execBash(self, command: str) -> str:
        """通过SSH执行Bash命令
        
        Args:
            command: 要执行的Bash命令
            
        Returns:
            str: 命令执行结果
        """
        if not self._checkConnection():
            return "连接失败，无法执行命令"
        
        command = self._preprocessCode(command, "bash")
        try:
            stdin, stdout, stderr = self.client.exec_command(f"source base/bin/activate && {command}")
            
            # 读取命令输出
            output = stdout.read().decode('utf-8')
            error = stderr.read().decode('utf-8')
            
            if error:
                logger.warning(f"命令执行产生错误: {error}")
                import pdb; pdb.set_trace()
                return f"输出: {output}\n错误: {error}"
            
            return output
        except Exception as e:
            logger.error(f"执行命令时出错: {str(e)}")
            return f"执行命令时出错: {str(e)}"

    def execPython(
            self, 
            code: str, 
            varDict: Optional[Dict[str, Any]] = None, 
            **kwargs) -> str:
        """通过SSH执行Python代码，使用SFTP上传脚本文件以避免转义问题。

        要获取执行结果，请在提供的 code 的最后，
        将需要返回的值赋给特殊变量 return_value。
        例如: code=\"a=1\\nb=2\\nreturn_value = a+b\"
        
        Args:
            code: 要执行的Python代码。代码的最后应将结果赋给 return_value。
            varDict: 可选的变量字典，会在执行代码前设置到Python环境中
            
        Returns:
            str: 命令执行结果，通常是 return_value 的值或执行过程中的输出/错误。
        """
        code = self._preprocessCode(code, "python")
        
        # 生成随机后缀以避免冲突
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        remoteTmpFile = f"/tmp/milkie_python_cmd_{os.getpid()}_{random_suffix}.py"
        remoteTmpVarFile = None
        result = "执行失败" # 默认结果
        localTmpFilePath = None 
        varDictTmpFilePath = None
        try:
            # 准备Python脚本行列表
            script_lines = [] 
            # 添加必要的导入
            script_lines.append("# -*- coding: utf-8 -*-")
            script_lines.append("import json")
            script_lines.append("import reprlib")
            script_lines.append("return_value = None")
            for preImport in PreImport:
                # 避免重复导入
                if f"import {preImport}" not in script_lines:
                    script_lines.append(f"import {preImport}")
            
            # 添加用户代码
            script_lines.append(code)
            
            # 将所有行合并为脚本字符串
            script_content = "\n".join(script_lines)
            
            # 创建本地临时文件来存储脚本
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as localTmpFile:
                localTmpFile.write(script_content)
                localTmpFilePath = localTmpFile.name
            
            logger.info(f"本地临时脚本: {localTmpFilePath}")
            logger.info(f"远程临时脚本: {remoteTmpFile}")

            # 如果提供了变量字典，将其保存到临时JSON文件并上传
            if varDict and isinstance(varDict, dict):
                import json
                # 创建变量字典的临时文件
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as varDictTmpFile:
                    json.dump(varDict, varDictTmpFile, ensure_ascii=False)
                    varDictTmpFilePath = varDictTmpFile.name
                
                # 上传变量字典文件到远程服务器
                remoteTmpVarFile = f"/tmp/milkie_vars_{os.getpid()}_{random_suffix}.json"
                logger.info(f"将变量字典保存到临时文件并上传: {varDictTmpFilePath} -> {remoteTmpVarFile}")
                
                if not self.uploadFile(varDictTmpFilePath, remoteTmpVarFile):
                    logger.error("上传变量字典文件失败")
                    raise IOError("上传变量字典文件失败")
                
                # 修改脚本，从文件读取变量字典
                prepend_script = [
                    "# 从临时文件读取变量字典",
                    "import json",
                    f"with open('{remoteTmpVarFile}', 'r', encoding='utf-8') as var_file:",
                    "    _var_dict = json.load(var_file)",
                    "globals().update(_var_dict)"
                ]
                
                # 在脚本开头插入读取变量的代码
                script_content = "\n".join(prepend_script) + "\n" + script_content
                
                # 重新写入更新后的脚本
                with open(localTmpFilePath, 'w', encoding='utf-8') as f:
                    f.write(script_content)

            # 上传脚本文件到远程服务器
            if not self.uploadFile(localTmpFilePath, remoteTmpFile):
                logger.error("上传脚本文件失败")
                raise IOError("上传脚本文件失败") 
            
            # 执行远程脚本
            execution_output = self.execBash(f"python3 {remoteTmpFile}")
            result = execution_output.strip() 
            
        except Exception as e:
            logger.error(f"执行Python脚本过程中出错: {str(e)}")
            result = f"执行Python脚本过程中出错: {str(e)}" 
        finally:
            # 清理临时文件
            #self._cleanupTempFile(localTmpFilePath, "本地临时脚本")
            #self._cleanupTempFile(varDictTmpFilePath, "本地变量字典临时文件")
            
            # 清理远程文件
            #if remoteTmpFile:
            #    self._cleanupRemoteTempFile(remoteTmpFile, "远程临时脚本")
            
            #if remoteTmpVarFile:
            #    self._cleanupRemoteTempFile(remoteTmpVarFile, "远程变量字典临时文件")
            pass

        return result
    
    def uploadFile(self, localPath: str, remotePath: str) -> bool:
        """上传文件到远程服务器
        
        Args:
            localPath: 本地文件路径
            remotePath: 远程文件路径
            
        Returns:
            bool: 上传是否成功
        """
        if not self._checkConnection():
            return False
        
        try:
            sftp = self.client.open_sftp()
            
            # 确保目标目录存在
            remoteDir = os.path.dirname(remotePath)
            if remoteDir:
                try:
                    self.execBash(f"mkdir -p {remoteDir}")
                except Exception as e:
                    logger.warning(f"创建目录失败: {remoteDir}: {str(e)}")
            
            sftp.put(localPath, remotePath)
            sftp.close()
            logger.info(f"文件上传成功: {localPath} -> {remotePath}")
            return True
        except Exception as e:
            logger.error(f"文件上传失败: {str(e)}")
            return False
    
    def downloadFile(self, remotePath: str, localPath: str) -> bool:
        """从远程服务器下载文件
        
        Args:
            remotePath: 远程文件路径
            localPath: 本地文件路径
            
        Returns:
            bool: 下载是否成功
        """
        if not self._checkConnection():
            return False
        
        try:
            sftp = self.client.open_sftp()
            
            # 确保本地目标目录存在
            localDir = os.path.dirname(localPath)
            if localDir:
                os.makedirs(localDir, exist_ok=True)
            
            sftp.get(remotePath, localPath)
            sftp.close()
            logger.info(f"文件下载成功: {remotePath} -> {localPath}")
            return True
        except Exception as e:
            logger.error(f"文件下载失败: {str(e)}")
            return False
    
    def listDir(self, remotePath: str) -> List[str]:
        """列出远程目录中的文件和目录
        
        Args:
            remotePath: 远程目录路径
            
        Returns:
            List[str]: 文件和目录列表
        """
        if not self._checkConnection():
            return []
            
        try:
            sftp = self.client.open_sftp()
            files = sftp.listdir(remotePath)
            sftp.close()
            return files
        except Exception as e:
            logger.error(f"列出目录失败: {str(e)}")
            return []
    
    def disconnect(self) -> None:
        """断开SSH连接"""
        if self.client:
            self.client.close()
            self.client = None
            self.connected = False
            logger.info(f"已断开与 {self.config.host} 的连接")
    
    def _preprocessCode(self, code: str, type: str) -> str:
        """预处理代码
        
        Args:
            code: 要预处理的代码
        """
        startFlag = f"```{type}"
        endFlag = f"```"
        idxStart = code.find(startFlag)
        if idxStart == -1:
            return code

        idxEnd = code.find(endFlag, idxStart + len(startFlag))
        if idxEnd == -1:
            return code[idxStart + len(startFlag):]
        return code[idxStart + len(startFlag):idxEnd]
    
    def _cleanupTempFile(self, filepath: Optional[str], description: str) -> None:
        """清理本地临时文件
        
        Args:
            filepath: 文件路径
            description: 文件描述，用于日志
        """
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(f"已删除{description}: {filepath}")
            except OSError as e:
                logger.warning(f"删除{description}失败: {filepath}, Error: {e}")
    
    def _cleanupRemoteTempFile(self, remotepath: str, description: str) -> None:
        """清理远程临时文件
        
        Args:
            remotepath: 远程文件路径
            description: 文件描述，用于日志
        """
        cleanup_result = self.execBash(f"rm -f {remotepath}")
        if any(err in cleanup_result for err in ["错误", "Error", "No such file", "failed"]):
            logger.warning(f"清理{description}可能失败: {remotepath}, 清理命令输出: {cleanup_result}")
        else:
            logger.info(f"已尝试清理{description}: {remotepath}")
 
    def __enter__(self):
        """支持with语句的上下文管理"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时关闭连接"""
        self.disconnect()

class VMFactory:
    """虚拟机工厂类，用于创建不同类型的VM实例"""
    
    @staticmethod
    def createVM(config: VMConfig) -> VM:
        """根据配置创建相应的VM实例
        
        Args:
            config: VM配置信息
            
        Returns:
            VM: 创建的VM实例
        """
        from milkie.config.config import VMConnectionType
        
        if config.connectionType == VMConnectionType.SSH:
            return VMSSH(config)
        elif config.connectionType == VMConnectionType.DOCKER:
            # 这里可以实现Docker类型的VM
            raise NotImplementedError("Docker类型的VM尚未实现")
        else:
            raise ValueError(f"不支持的VM连接类型: {config.connectionType}")

   
if __name__ == "__main__":
    encryptedPassword = SecurityUtils.encrypt("password", SecurityUtils.get_env_password())
    print(encryptedPassword)
    vmConfig = VMConfig(
        host="localhost",
        port=2222,
        username="myuser",
        encryptedPassword=encryptedPassword,
        connectionType=VMConnectionType.SSH
    )
    vm = VMFactory.createVM(vmConfig)
    print(vm.execPython("import random; print(random.randint(1, 10))"))