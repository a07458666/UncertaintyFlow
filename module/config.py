
import os
import yaml
import datetime
import pytz

def checkOutputDirectoryAndCreate(output_folder):
    if not os.path.exists('result/' + output_folder):
        os.makedirs('result/' + output_folder)

def loadConfig(path):
    f = open(path)
    config = yaml.load(f, Loader=yaml.FullLoader)
    tz = datetime.timezone(datetime.timedelta(hours=+8))
    config["output_folder"] +=datetime.datetime.now(tz=tz).strftime("_%m-%d_%H:%M") 
    return config

def dumpConfig(config):
    with open("./result/" + config["output_folder"] + "/config.yaml", 'w') as outputFile:
        yaml.dump(config, outputFile)
    return