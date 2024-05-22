import os

NewEnv = os.environ.copy()
NewEnv['SCARF_NO_ANALYTICS'] = 'true'
NewEnv['DO_NOT_TRACK'] = 'true'