import logging
import os
import json
import yaml
import concurrent.futures
from typing import Dict, List, Optional, Type, Any, TypeVar
from dataclasses import dataclass, asdict
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor

from milkie.config.config import GlobalConfig, DataSourceType

from .datasource.datasource import DataSource
from .datasource.sql import DataSourceMysql
from .basic.concept import Concept
from .mapping import Mapping

logger = logging.getLogger(__name__)

# 自定义类型
T = TypeVar('T')

class MergeStrategy(Enum):
    """概念合并策略"""
    REPLACE = auto()      # 完全替换已有概念
    EXTEND = auto()       # 保留原有成员，添加新成员
    KEEP_EXISTING = auto()  # 保留已有概念，忽略新概念
    RENAME_NEW = auto()   # 重命名新概念以避免冲突


class OntologyStatus(Enum):
    """本体状态"""
    INITIALIZED = auto()    # 初始化完成
    LOADING = auto()        # 正在加载
    BUILDING = auto()       # 正在构建
    READY = auto()          # 已准备就绪
    ERROR = auto()          # 发生错误


@dataclass
class OntologyStats:
    """本体统计信息"""
    dataSourcesCount: int = 0
    conceptsCount: int = 0
    mappingsCount: int = 0
    lastBuildTime: str = ""
    lastConfigLoadTime: str = ""


class Ontology:
    """本体管理类

    负责管理数据源 (DataSource)、概念 (Concept) 和它们之间的映射 (Mapping)。
    可以从配置文件加载数据源，并触发扫描以自动生成概念和映射。
    支持本体的序列化、验证和状态管理。
    """

    # 数据源类型到实现类的映射
    _DATA_SOURCE_REGISTRY: Dict[DataSourceType, Type[DataSource]] = {
        DataSourceType.MYSQL: DataSourceMysql,
    }

    def __init__(self, globalConfig: GlobalConfig):
        self._globalConfig = globalConfig
        self._dataSources: Dict[str, DataSource] = {}
        self._concepts: Dict[str, Concept] = {}
        self._mappings: Dict[tuple[str, str], Mapping] = {}
        self._status: OntologyStatus = OntologyStatus.INITIALIZED
        self._stats: OntologyStats = OntologyStats()
        logger.info("本体管理器已初始化")
        self._loadDataSourcesFromConfig()

    @property
    def status(self) -> OntologyStatus:
        """获取当前本体状态"""
        return self._status

    @property
    def stats(self) -> OntologyStats:
        """获取本体统计信息"""
        self._stats.dataSourcesCount = len(self._dataSources)
        self._stats.conceptsCount = len(self._concepts)
        self._stats.mappingsCount = len(self._mappings)
        return self._stats

    def registerDataSourceType(self, dataType: DataSourceType, cls: Type[DataSource]) -> None:
        """注册新的数据源类型

        Args:
            dataType (DataSourceType): 数据源类型枚举 (来自 config.py)
            cls (Type[DataSource]): 处理该类型的 DataSource 子类
        """
        Ontology._DATA_SOURCE_REGISTRY[dataType] = cls
        logger.info(f"已注册数据源类型 {dataType.name} -> {cls.__name__}")

    def addDataSource(self, dataSource: DataSource) -> None:
        """添加一个数据源实例"""
        if dataSource.name in self._dataSources:
            logger.warning(f"数据源 '{dataSource.name}' 已存在，将被覆盖")
        self._dataSources[dataSource.name] = dataSource
        logger.debug(f"数据源已添加: {dataSource.name}")

    def getDataSource(self, name: str) -> Optional[DataSource]:
        """根据名称获取数据源实例"""
        return self._dataSources.get(name)

    def getAllDataSources(self) -> List[DataSource]:
        """获取所有数据源实例"""
        return list(self._dataSources.values())

    def addConcept(
            self, 
            concept: Concept, 
            strategy: MergeStrategy = MergeStrategy.REPLACE) -> Concept:
        """添加一个概念实例，使用指定的合并策略处理冲突

        Args:
            concept (Concept): 要添加的概念
            strategy (MergeStrategy): 当概念已存在时的合并策略

        Returns:
            Concept: 最终添加/合并后的概念
        """
        existingConcept = self._concepts.get(concept.name)
        if existingConcept is None:
            # 概念不存在，直接添加
            self._concepts[concept.name] = concept
            logger.debug(f"概念已添加: {concept.name}")
            return concept
            
        # 概念已存在，根据策略处理
        if strategy == MergeStrategy.REPLACE:
            # 完全替换
            self._concepts[concept.name] = concept
            logger.warning(f"概念 '{concept.name}' 已存在，已替换")
            return concept
            
        elif strategy == MergeStrategy.EXTEND:
            # 合并成员
            mergedMembers = dict(existingConcept.members)
            for name, type_ in concept.members.items():
                if name in mergedMembers:
                    logger.debug(f"概念 '{concept.name}' 中的成员 '{name}' 已存在，保留原有类型")
                else:
                    mergedMembers[name] = type_
                    logger.debug(f"概念 '{concept.name}' 中添加了新成员 '{name}'")
            
            # 创建新的 Concept 实例
            mergedConcept = Concept(concept.name, mergedMembers)
            self._concepts[concept.name] = mergedConcept
            logger.info(f"概念 '{concept.name}' 已合并，共 {len(mergedMembers)} 个成员")
            return mergedConcept
            
        elif strategy == MergeStrategy.KEEP_EXISTING:
            # 保留现有概念
            logger.debug(f"概念 '{concept.name}' 已存在，保留原有定义，忽略新定义")
            return existingConcept
            
        elif strategy == MergeStrategy.RENAME_NEW:
            # 重命名新概念
            i = 1
            newName = f"{concept.name}_{i}"
            while newName in self._concepts:
                i += 1
                newName = f"{concept.name}_{i}"
            
            # 创建新的 Concept 实例
            renamedConcept = Concept(newName, concept.members)
            self._concepts[newName] = renamedConcept
            logger.warning(f"概念 '{concept.name}' 已存在，已重命名为 '{newName}'")
            return renamedConcept

    def getConcept(self, name: str) -> Optional[Concept]:
        """根据名称获取概念实例"""
        return self._concepts.get(name)
    
    def getAllConcepts(self) -> List[Concept]:
        """获取所有概念实例"""
        return list(self._concepts.values())

    def getConceptDescription(self, name: str) -> str:
        """获取概念描述"""
        concept = self._concepts.get(name)
        if concept is None:
            return f"概念 '{name}' 不存在"

        return json.dumps(concept.toDict(), ensure_ascii=False, indent=2)

    def getAllConceptsDescription(self) -> str:
        """生成概念描述的 JSON 格式字符串

        Returns:
            str: 包含所有概念及其成员的 JSON 格式字符串
        """
        import json
        
        concepts_data = {}
        for concept in self._concepts.values():
            concepts_data[concept.name] = concept.toDict()
        
        return json.dumps(concepts_data, ensure_ascii=False, indent=2)

    def addMapping(self, mapping: Mapping) -> None:
        """添加一个映射实例"""
        mappingKey = (mapping.dataSource.name, mapping.concept.name)
        if mappingKey in self._mappings:
            logger.warning(f"数据源 '{mapping.dataSource.name}' 到概念 '{mapping.concept.name}' 的映射已存在，将被覆盖")
        self._mappings[mappingKey] = mapping
        logger.debug(f"映射已添加: {mapping.dataSource.name} -> {mapping.concept.name}")

    def getMapping(self, dataSourceName: str, conceptName: str) -> Optional[Mapping]:
        """根据数据源名称和概念名称获取映射实例"""
        return self._mappings.get((dataSourceName, conceptName))

    def getMappingsForDataSource(self, dataSourceName: str) -> List[Mapping]:
        """获取指定数据源的所有映射"""
        return [m for k, m in self._mappings.items() if k[0] == dataSourceName]

    def getMappingsForConcept(self, conceptName: str) -> List[Mapping]:
        """获取指定概念的所有映射"""
        return [m for k, m in self._mappings.items() if k[1] == conceptName]
    
    def getAllMappings(self) -> List[Mapping]:
        """获取所有映射实例"""
        return list(self._mappings.values())

    def getDataSourceFromConcept(self, conceptName: str) -> Optional[DataSource]:
        """根据概念获取数据源"""
        for k, m in self._mappings.items():
            if m.concept.name == conceptName:
                return m.dataSource
        return None

    def getDataSourcesFromConcepts(self, concepts: list) -> list:
        """根据概念获取数据源"""
        return [m.dataSource.config for m in self._mappings.values() if m.concept.name in concepts]

    def getDataSourceSchemasFromConcepts(self, concepts: List[str]) -> Dict[str, Any]:
        """根据概念获取数据源的 schema"""
        data = {}
        for conceptName in concepts:
            dataSource = self.getDataSourceFromConcept(conceptName)
            if dataSource is None:
                continue
            data[conceptName] = dataSource.get_schema()
        return data

    def sampleData(self, conceptNames: List[str], count: int = 1) -> Dict[str, Any]:
        """根据概念获取样本数据"""
        data = {}
        for conceptName in conceptNames:
            dataSource = self.getDataSourceFromConcept(conceptName)
            if dataSource is None:
                continue

            sampledata = dataSource.sampleData(conceptName, count)
            if sampledata:
                data[conceptName] = sampledata
        return data

    def executeSql(self, sql: str) -> List[Dict[str, Any]]:
        """执行 SQL 语句并返回结果

        Args:
            sql (str): 要执行的 SQL 语句

        Returns:
            List[Dict[str, Any]]: 查询结果列表，每个元素是一个字典，键为列名，值为对应值
        """
        return self._dataSources[self._globalConfig.getDataSourceName()].executeSql(sql)

    def buildOntologyFromSources(self, 
                                runScan: bool = True, 
                                concurrent: bool = True,
                                maxWorkers: int = None,
                                conceptStrategy: MergeStrategy = MergeStrategy.EXTEND) -> None:
        """从已添加的数据源构建本体（Concepts 和 Mappings）
        
        Args:
            run_scan (bool): 是否对每个数据源执行 scan 操作，默认为 True
            concurrent (bool): 是否并发扫描多个数据源，默认为 True
            max_workers (int): 最大工作线程数，默认为 None（由系统决定）
            concept_strategy (MergeStrategy): 概念合并策略，默认为 EXTEND
        """
        import datetime
        
        if not self._dataSources:
            logger.warning("没有数据源可用于构建本体")
            return

        self._status = OntologyStatus.BUILDING
        logger.info("开始从数据源构建本体...")
        
        # 如果不运行扫描，直接返回
        if not runScan:
            logger.info("跳过所有数据源扫描 (runScan=False)")
            self._status = OntologyStatus.READY
            return
        
        # 并发扫描数据源
        if concurrent and len(self._dataSources) > 1:
            self._scanDataSourcesConcurrently(maxWorkers, conceptStrategy)
        else:
            self._scanDataSourcesSequentially(conceptStrategy)
            
        self._stats.lastBuildTime = datetime.datetime.now().isoformat()
        self._status = OntologyStatus.READY
        logger.info("本体构建过程完成")

    def validate(self) -> List[str]:
        """验证本体的一致性

        检查数据源、概念和映射之间的引用关系是否一致。

        Returns:
            List[str]: 验证错误消息列表，如果没有错误则为空列表
        """
        errors = []
        
        # 1. 检查所有映射引用的数据源是否存在
        for (dsName, conceptName), mapping in self._mappings.items():
            if mapping.data_source.name != dsName:
                errors.append(f"映射键 ({dsName}, {conceptName}) 与映射对象中的数据源名称 {mapping.data_source.name} 不一致")
            
            if dsName not in self._dataSources:
                errors.append(f"映射 ({dsName}, {conceptName}) 引用了不存在的数据源 '{dsName}'")
                
            # 2. 检查所有映射引用的概念是否存在
            if mapping.concept.name != conceptName:
                errors.append(f"映射键 ({dsName}, {conceptName}) 与映射对象中的概念名称 {mapping.concept.name} 不一致")
                
            if conceptName not in self._concepts:
                errors.append(f"映射 ({dsName}, {conceptName}) 引用了不存在的概念 '{conceptName}'")
                
            # 3. 检查映射中的字段到成员映射是否有效
            for memberName in mapping.fieldToMemberMap.values():
                concept = self._concepts.get(conceptName)
                if concept and memberName not in concept.members:
                    errors.append(f"映射 ({dsName}, {conceptName}) 引用了概念中不存在的成员 '{memberName}'")
        
        if not errors:
            logger.info("本体验证通过，未发现问题")
        else:
            logger.warning(f"本体验证发现 {len(errors)} 个问题")
            for i, error in enumerate(errors, 1):
                logger.warning(f"问题 {i}: {error}")
            
        return errors

    def saveToFile(self, filePath: str) -> bool:
        """将本体保存到文件

        保存的格式是 JSON，包含概念定义、数据源参考和映射关系。
        不会保存数据源连接详情以避免泄露敏感信息。

        Args:
            file_path (str): 保存路径

        Returns:
            bool: 保存是否成功
        """
        try:
            # 1. 收集概念信息
            conceptsData = {}
            for name, concept in self._concepts.items():
                conceptsData[name] = concept.toDict()

            # 2. 收集数据源参考信息（不包含密码等敏感信息）
            datasourcesRef = {}
            for name, ds in self._dataSources.items():
                datasourcesRef[name] = {
                    "name": name,
                    "type": ds.type.name
                }
            
            # 3. 收集映射信息
            mappingsData = []
            for (dsName, conceptName), mapping in self._mappings.items():
                mappingsData.append({
                    "data_source": dsName,
                    "concept": conceptName,
                    "field_to_member_map": mapping.fieldToMemberMap
                })
            
            # 4. 组装完整数据
            ontologyData = {
                "concepts": conceptsData,
                "datasources_ref": datasourcesRef,
                "mappings": mappingsData,
                "stats": asdict(self._stats)
            }
            
            # 5. 写入文件
            with open(filePath, 'w', encoding='utf-8') as f:
                json.dump(ontologyData, f, indent=2, ensure_ascii=False)
                
            logger.info(f"本体已保存到文件: {filePath}")
            return True
            
        except Exception as e:
            logger.exception(f"保存本体到文件 {filePath} 时出错: {e}")
            return False

    def loadFromFile(self, filePath: str) -> bool:
        """从文件加载本体结构

        Args:
            filePath (str): 本体文件路径

        Returns:
            bool: 加载是否成功
        """
        try:
            with open(filePath, 'r', encoding='utf-8') as f:
                ontologyData = json.load(f)
            
            # 验证数据格式
            if not all(key in ontologyData for key in ["concepts", "mappings"]):
                logger.error(f"文件 {filePath} 格式不正确，缺少必要的键")
                return False
            
            # 加载概念
            from .basic.concept import Concept, ConceptMemberType
            concepts = {}
            for name, conceptData in ontologyData["concepts"].items():
                members = {}
                for memberName, typeName in conceptData["members"].items():
                    try:
                        memberType = ConceptMemberType[typeName]
                        members[memberName] = memberType
                    except KeyError:
                        logger.warning(f"未知的成员类型: {typeName}，使用 ANY 替代")
                        members[memberName] = ConceptMemberType.ANY
                
                concepts[name] = Concept(name=name, members=members)
            
            # 加载映射（需要已有的数据源实例）
            from .mapping import Mapping
            mappings = {}
            for mappingData in ontologyData["mappings"]:
                dsName = mappingData["data_source"]
                conceptName = mappingData["concept"]
                fieldMap = mappingData["field_to_member_map"]
                
                # 检查数据源和概念是否可用
                dataSource = self._dataSources.get(dsName)
                concept = concepts.get(conceptName)
                
                if not dataSource:
                    logger.warning(f"映射中引用的数据源 '{dsName}' 不存在，跳过")
                    continue
                    
                if not concept:
                    logger.warning(f"映射中引用的概念 '{conceptName}' 不存在，跳过")
                    continue
                
                # 创建映射
                try:
                    mapping = Mapping(dataSource=dataSource, concept=concept, fieldToMemberMap=fieldMap)
                    mappings[(dsName, conceptName)] = mapping
                except ValueError as e:
                    logger.warning(f"创建映射 ({dsName}, {conceptName}) 失败: {e}")
            
            # 更新内部状态
            self._concepts = concepts
            self._mappings = mappings
            
            logger.info(f"已从文件 {filePath} 加载本体，包含 {len(concepts)} 个概念和 {len(mappings)} 个映射")
            self._status = OntologyStatus.READY
            return True
            
        except FileNotFoundError:
            logger.error(f"本体文件未找到: {filePath}")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"解析本体文件 {filePath} 出错: {e}")
            return False
        except Exception as e:
            logger.exception(f"加载本体文件 {filePath} 时发生意外错误")
            return False

    def reset(self) -> None:
        """重置本体，清除所有数据源、概念和映射"""
        self._data_sources.clear()
        self._concepts.clear()
        self._mappings.clear()
        self._status = OntologyStatus.INITIALIZED
        logger.info("本体已重置")

    def _loadDataSourcesFromConfig(self) -> None:
        """从全局配置加载数据源实例"""
        import datetime

        self._status = OntologyStatus.LOADING
        logger.info("从全局配置加载数据源")
        try:
            # 从全局配置获取 DataSourceConfig 列表
            dataSourcesConfigs: List['DataSourceConfig'] = self._globalConfig.getDataSourcesConfig().getAllSourceConfigs()

            if not dataSourcesConfigs:
                logger.warning("全局配置中没有数据源")
                self._status = OntologyStatus.INITIALIZED
                return

            loadedCount = 0
            # 遍历从配置加载的 DataSourceConfig
            for dsConfig in dataSourcesConfigs:
                name = dsConfig.name
                # 直接使用从 config.py 加载的类型枚举
                dataSourceType: DataSourceType = dsConfig.type

                if not name or not dataSourceType:
                    logger.warning(f"跳过无效的数据源配置（缺少 name 或 type）：{dsConfig}")
                    continue

                # 直接使用 config.DataSourceType 创建 DataSource 实例
                datasourceInstance: Optional[DataSource] = self._createDataSource(name, dataSourceType, dsConfig.__dict__)
                if datasourceInstance:
                    self.addDataSource(datasourceInstance)
                    logger.info(f"成功加载并添加数据源: {name} ({dataSourceType.name})")
                    loadedCount += 1

            self._stats.lastConfigLoadTime = datetime.datetime.now().isoformat()
            logger.info(f"成功从全局配置加载了 {loadedCount} 个数据源")
            self._status = OntologyStatus.READY if loadedCount > 0 else OntologyStatus.INITIALIZED

        except Exception as e:
            logger.exception(f"加载数据源配置时发生意外错误: {e}")
            self._status = OntologyStatus.ERROR

    def _createDataSource(self, name: str, dataType: DataSourceType, config: Dict[str, Any]) -> Optional[DataSource]:
        """根据类型和配置创建数据源实例

        Args:
            name (str): 数据源名称
            dataType (DataSourceType): 数据源类型枚举 (config.DataSourceType)
            config (Dict[str, Any]): 数据源的具体配置
        """
        # 使用传入的 config.DataSourceType 进行查找
        datasourceCls = self._DATA_SOURCE_REGISTRY.get(dataType)
        if not datasourceCls:
            logger.warning(f"数据源类型 '{dataType.name}' 未注册，跳过 '{name}'")
            return None

        try:
            # 创建数据源实例
            datasourceInstance = datasourceCls(name=name, config=config)
            return datasourceInstance
        except Exception as e:
            logger.error(f"创建 {dataType.name} 数据源 '{name}' 实例失败: {e}")
            return None

    def _scanDataSourcesSequentially(self, conceptStrategy: MergeStrategy) -> None:
        """按顺序扫描数据源"""
        for dsName, dataSource in self._dataSources.items():
            self._scanSingleDataSource(dsName, dataSource, conceptStrategy)

    def _scanDataSourcesConcurrently(self, maxWorkers: int, conceptStrategy: MergeStrategy) -> None:
        """并发扫描数据源"""
        with ThreadPoolExecutor(max_workers=maxWorkers) as executor:
            # 提交所有任务
            futures = {
                executor.submit(self._scanSingleDataSource, dsName, dataSource, conceptStrategy): dsName
                for dsName, dataSource in self._dataSources.items()
            }
            
            # 等待完成
            for future in concurrent.futures.as_completed(futures):
                dsName = futures[future]
                try:
                    future.result()  # 获取结果（如果有异常，会在这里抛出）
                except Exception as e:
                    logger.exception(f"并发扫描数据源 {dsName} 时出现未处理的异常: {e}")

    def _scanSingleDataSource(
            self, 
            dsName: str, 
            dataSource: DataSource, 
            conceptStrategy: MergeStrategy) -> None:
        """扫描单个数据源"""
        logger.info(f"正在扫描数据源: {dsName}...")
        try:
            # 测试连接，如果失败则跳过扫描
            if not dataSource.test_connection():
                logger.warning(f"数据源 {dsName} 连接测试失败，跳过扫描")
                return
                
            # 执行扫描以获取 Mappings (scan 内部会创建 Concepts)
            mappings = dataSource.scan()
            if not mappings:
                logger.info(f"数据源 {dsName} 扫描未返回任何映射")
                return
                
            # 添加扫描生成的 Concepts 和 Mappings
            added_mappings = 0
            for mapping in mappings:
                finalConcept = self.addConcept(mapping.concept, strategy=conceptStrategy)
                if finalConcept is not mapping.concept:
                    from .mapping import Mapping
                    newMapping = Mapping(
                        dataSource=mapping.dataSource,
                        space=mapping.space,
                        concept=finalConcept,
                        fieldToMemberMap=mapping.fieldToMemberMap
                    )
                    self.addMapping(newMapping)
                else:
                    self.addMapping(mapping)
                
                added_mappings += 1
                
            logger.info(f"数据源 {dsName} 扫描完成，添加了 {len(mappings)} 个映射")
        
        except NotImplementedError:
            logger.error(f"数据源 {dsName} ({dataSource.type.name}) 的 scan 方法未实现")
        except ConnectionError as e:
            logger.error(f"扫描数据源 {dsName} 时连接失败: {e}")
        except Exception as e:
            logger.exception(f"扫描数据源 {dsName} 时发生意外错误")