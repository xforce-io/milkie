from typing import Dict, Optional
import yaml
from urllib.parse import urlparse

class RobotPolicy:
    def __init__(self, allowed: bool, delay: float):
        self.allowed = allowed
        self.delay = delay

def loadRobotPolicies(config_path: str) -> Dict[str, RobotPolicy]:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    policies = {}
    for url_pattern, policy in config.get('ROBOTS_WHITELIST', {}).items():
        policies[url_pattern] = RobotPolicy(allowed=policy['allowed'], delay=policy['delay'])
    
    return policies

def getRobotPolicy(url: str, policies: Dict[str, RobotPolicy]) -> RobotPolicy:
    parsed_url = urlparse(url)
    url_path = f"{parsed_url.netloc}{parsed_url.path}"
    
    # 从最长的匹配开始检查
    matching_patterns = sorted(
        [pattern for pattern in policies.keys() if pattern in url_path],
        key=len,
        reverse=True
    )
    
    for pattern in matching_patterns:
        return policies[pattern]
    
    return policies.get('DEFAULT', RobotPolicy(allowed=False, delay=0))