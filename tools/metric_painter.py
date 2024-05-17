#coding=utf-8

import matplotlib.pyplot as plt
import numpy as np
import json

InputFilepath = ""

def parseMetricObjsFromMetricLog(inputFilepath :str, keyMetrics :list[str]):
    PatternMetricLog = "exp result"
    results = []
    with open(inputFilepath, 'r') as file:
        data = file.readlines()
        for line in data:
            if line.find(PatternMetricLog) == -1:
                continue
        
            jsonObj = json.loads(line[line.find(PatternMetricLog) + len(PatternMetricLog):])
            resultObj = parseMetricObjFromJsonObj(jsonObj, keyMetrics)
            results.append(resultObj)
    return results

def extractConfigKeysWithSameVal(jsonObjs :list[dict]):
    configKeys = jsonObjs[0]["config"].keys()
    configKeysWithSameVal = []
    for key in configKeys:
        if key == "benchmarks": 
            continue

        if all([jsonObj["config"][key] == jsonObjs[0]["config"][key] for jsonObj in jsonObjs]):
            configKeysWithSameVal.append(key)
    return configKeysWithSameVal
    
def getExpObjFromMetricObjs(metricObjs :list[dict]):
    configKeysWithSameVal = extractConfigKeysWithSameVal(metricObjs)
    return {
        "commonConfig" : {key : metricObjs[0]["config"][key] for key in configKeysWithSameVal},
        "experiments" : [
            {
                "config" : {
                    key : jsonObj["config"][key] 
                        for key in jsonObj["config"] 
                        if key not in configKeysWithSameVal and key not in ("seed", "benchmarks")
                },
                "metrics" : jsonObj["metrics"]
            }
            for jsonObj in metricObjs
        ]
    }

def getMetricNameList(metricObj :dict):
    commonNames = []
    benchmarkNames = []
    for key in metricObj["metrics"]:
        if key == "benchmarks":
            benchmarkNames += [key for key in metricObj["metrics"][key]]
        else:
            commonNames.append(key)
    return (commonNames, benchmarkNames)

def getAnalysisObjFromExpObj(
        expObj :dict, 
        commonNames :list[str], 
        benchmarkNames :list[str]):
    analysisObjs = []
    for name in commonNames:
        analysisObj = {
            "commonConfig" : expObj["commonConfig"],
            "isBenchmark" : False,
            "name" : name, 
            "vals" : []
        }

        for exp in expObj["experiments"]:
            analysisObj["vals"].append({
                "config" : exp["config"],
                "val" : exp["metrics"][name]
            })
        analysisObjs.append(analysisObj)
    
    for name in benchmarkNames:
        analysisObj = {
            "commonConfig" : expObj["commonConfig"],
            "isBenchmark" : True,
            "name" : f"benchmark/{name}",
            "vals" : []
        }

        for exp in expObj["experiments"]:
            analysisObj["vals"].append({
                "config" : exp["config"],
                "val" : exp["metrics"]["benchmarks"][name]["accu"]
            })
        analysisObjs.append(analysisObj)
    return analysisObjs

def parseMetricObjFromJsonObj(jsonObj, keyMetrics :list[str]):
    extractSingleVal = lambda jsonObj: jsonObj["values"][0]
    resultObj = {
        "config" : jsonObj["config"],
        "metrics" : {
            key : extractSingleVal(jsonObj["metrics"][key]) 
                for key in jsonObj["metrics"] 
                if key in keyMetrics and not key.startswith("benchmark")
        }
    }

    def setDeepKv(d, keys, value):
        for key in keys[:-1]:
            if key not in d: d[key] = {}
            d = d[key]
        d[keys[-1]] = value

    metricObj = resultObj["metrics"]
    for key, val in jsonObj["metrics"].items():
        if key.startswith("benchmark"):
            items = key.split(".")
            setDeepKv(metricObj, ["benchmarks", items[1], items[-1]], extractSingleVal(val))
    return resultObj

def drawFigureForSingleAnylysisObj(ax, analysisObj):
    labels = [("\n".join([v for _,v in val["config"].items()])) for val in analysisObj["vals"]]
    vals = [val["val"] for val in analysisObj["vals"]]

    x = np.arange(len(labels)) * 0.1  # the label locations
    width = 0.05  # the width of the bars

    rects = ax.bar(x, vals, width)

    ax.set_ylim(0, max(vals) * 1.5)
    ax.set_ylabel(analysisObj["name"])
    ax.set_title(analysisObj["name"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6, rotation=20)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.3f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 1),
                        textcoords="offset points",
                        fontsize=6,
                        ha='center', 
                        va='bottom')
    autolabel(rects)

def plotAnalysisObjs(analysisObjs :list[dict]):
    fig, axs = plt.subplots(len(analysisObjs), 1, figsize=(5, len(analysisObjs)*2), label="abv")  # 创建子图布局，并设置图形尺寸
    for i, analysisObj in enumerate(analysisObjs):
        drawFigureForSingleAnylysisObj(axs[i], analysisObj)

    plt.text(
        0.5, 
        0.01, 
        json.dumps(analysisObjs[0]["commonConfig"]), 
        ha='center', 
        fontsize=8, 
        transform=fig.transFigure)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Anylysis milkie log')
    parser.add_argument('--logfilepath', type=str, help='filepath of log')
    parser.add_argument('--keymetrics', type=str, default="tokensPerSec,costSec,avgOutputLen", help='key metrics to analysis, split by comma')
    args = parser.parse_args()

    if os.path.exists(args.logfilepath):
        logfilepath = args.logfilepath
    else:
        print("Please provide a valid log file path.")
        exit(1)

    metricObjs = parseMetricObjsFromMetricLog(
        logfilepath, 
        [key.strip() for key in args.keymetrics.split(",") if len(key.strip()) != 0])
    commonNames, benchmarkNames = getMetricNameList(metricObjs[0])
    expObj = getExpObjFromMetricObjs(metricObjs)
    analysisObjs = getAnalysisObjFromExpObj(expObj, commonNames, benchmarkNames)
    plotAnalysisObjs(analysisObjs)