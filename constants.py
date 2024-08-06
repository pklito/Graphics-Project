import json
from types import SimpleNamespace

# Iterating through the json
# list

GLOBAL_CONSTANTS = SimpleNamespace()

def loadConstants(file="constants.json"):
    f = open('constants.json')
    data = json.load(f)
    f.close()

    global GLOBAL_CONSTANTS

    def iterate(subdict, subdata):
        for key, val in subdata.items():
            if type(val) == type({}):
                subdict[key] = SimpleNamespace()
                iterate(subdict[key].__dict__, val)
            else:
                subdict[key] = val
            

    iterate(GLOBAL_CONSTANTS.__dict__, data)


loadConstants()

print(GLOBAL_CONSTANTS)
print(GLOBAL_CONSTANTS.opencv.HOUGH_PROB_LINE_WIDTH)