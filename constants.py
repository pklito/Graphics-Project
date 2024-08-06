import json
from types import SimpleNamespace

def iterativeNamespaceFromDict(data):
    def iterate(holder, key, subdata):
        if type(subdata) == type({}):
            holder[key] = SimpleNamespace()
            for dk, dv in subdata.items():
                iterate(holder[key].__dict__, dk, dv)
        elif type(subdata) == type([]):
            holder[key] = list(range(len(subdata)))
            for dk, dv in enumerate(subdata):
                iterate(holder[key], dk, dv)
        else:
            holder[key] = subdata
    
    package = {0: None}
    iterate(package, 0, data)
    return package[0]


GLOBAL_CONSTANTS = None
def loadConstants(file="constants.json"):
    f = open('constants.json')
    data = json.load(f)
    f.close()

    global GLOBAL_CONSTANTS   
    GLOBAL_CONSTANTS = iterativeNamespaceFromDict(data)


loadConstants()