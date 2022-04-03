
import os
import yaml
import datetime

def checkOutputDirectoryAndCreate(output_foloder):
    if not os.path.exists('result/' + output_foloder):
        os.makedirs('result/' + output_foloder)

def loadConfig(path):
    f = open(path)
    config = yaml.load(f, Loader=yaml.FullLoader)
    config["output_foloder"] +=datetime.datetime.now().strftime("_%m-%d_%H:%M") 
    return config

def dumpConfig(config):
    with open("./result/" + config["output_foloder"] + "/config.yaml", 'w') as outputFile:
        yaml.dump(config, outputFile)
    return