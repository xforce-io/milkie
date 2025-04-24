import base64
import logging
import os
from typing import Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

class SecurityUtils:
    """安全工具类，提供加密和解密功能"""
    
    # 默认盐值，实际应用中应从环境变量或配置中获取
    DEFAULT_SALT = b'milkie_default_salt_value'
    
    @staticmethod
    def _get_key(password: str, salt: Optional[bytes] = None) -> bytes:
        """生成加密密钥
        
        Args:
            password: 用于生成密钥的密码
            salt: 可选的盐值，如果未提供则使用默认盐值
            
        Returns:
            bytes: 生成的密钥
        """
        if not salt:
            salt = SecurityUtils.DEFAULT_SALT
            
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))
    
    @staticmethod
    def encrypt(text: str, password: str, salt: Optional[bytes] = None) -> str:
        """加密文本
        
        Args:
            text: 要加密的文本
            password: 用于生成密钥的密码
            salt: 可选的盐值
            
        Returns:
            str: 加密后的文本（Base64编码）
        """
        try:
            key = SecurityUtils._get_key(password, salt)
            f = Fernet(key)
            encrypted_data = f.encrypt(text.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"加密失败: {str(e)}")
            raise RuntimeError(f"加密失败: {str(e)}")
    
    @staticmethod
    def decrypt(encrypted_text: str, password: str, salt: Optional[bytes] = None) -> str:
        """解密文本
        
        Args:
            encrypted_text: 加密的文本（Base64编码）
            password: 用于生成密钥的密码
            salt: 可选的盐值
            
        Returns:
            str: 解密后的文本
        """
        try:
            key = SecurityUtils._get_key(password, salt)
            f = Fernet(key)
            # 先对加密文本进行Base64解码
            decoded_data = base64.urlsafe_b64decode(encrypted_text.encode())
            decrypted_data = f.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"解密失败: {str(e)}")
            return encrypted_text  # 解密失败时返回原始加密文本
    
    @staticmethod
    def generate_key() -> str:
        """生成一个新的Fernet密钥
        
        Returns:
            str: 生成的密钥（Base64编码的字符串）
        """
        key = Fernet.generate_key()
        return key.decode()
    
    @staticmethod
    def get_env_password() -> str:
        """从环境变量获取密码
        
        Returns:
            str: 从环境变量获取的密码，如果未设置则返回默认值
        """
        return os.environ.get('MILKIE_PASSWORD', 'default_password_please_change_me') 