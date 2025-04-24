from enum import Enum

class QueryType(Enum):
    STR = 0
    FILEPATH = 1

class QueryStructure:

    PatternFilepath = "@filepath:"

    def __init__(self, queryType :QueryType, query :str) -> None:
        self.queryType = queryType
        self.query = query

def parseQuery(query :str) -> QueryStructure:
    if not query:
        return None
    
    #if query start with "@filepath:", parse as FILEPATH type
    if query.startswith(QueryStructure.PatternFilepath):
        return QueryStructure(QueryType.FILEPATH, query[len(QueryStructure.PatternFilepath):])
    else:
        return QueryStructure(QueryType.STR, query)
