from SOAPify import SOAPclassification
import numpy
import re
from scipy.ndimage import gaussian_filter


import matplotlib.pyplot as plt
import h5py


pFEScmap = plt.cm.coolwarm_r.copy()
pFEScmap.set_over("w")

# __reT = re.compile("T_([0-9]*)")
# The regular expression below will
__reT = re.compile("_([0-9]*)")


def getT(s):
    match = __reT.search(s)
    if match:
        return int(match.group(1))
    else:
        # this is a trick for loading data in createXYZ.py
        return "Ideal"


def getF(proposed: float, concurrentArray, F):
    avversary = F(concurrentArray)
    return F([proposed, avversary])


def getMin(proposed: float, concurrentArray):
    return getF(proposed, concurrentArray, numpy.min)


def getMax(proposed: float, concurrentArray):
    return getF(proposed, concurrentArray, numpy.max)


def loadClassification(
        classificationFile: str,
        dataContainer: dict,
        NPname: str,
        ClassificationPlace: str,
        ClassNameToFind: str,
        ClassNameToSave: str,
        bottomUpLabels: list,
        TimeWindow: slice = slice(None),
):
    with h5py.File(classificationFile, "r") as f:
        Simulations = f[ClassificationPlace]
        for k in Simulations:
            if NPname in k:
                # Get the temperature based on string "SmallExample_100", will get 100
                # T = getT(k)
                T = 303
                print("Temperature is: {}".format(T))
                # if T not in dataContainer:
                #     dataContainer[T] = dict()
            dataContainer[T][ClassNameToSave] = SOAPclassification(
                [], Simulations[k][ClassNameToFind][TimeWindow], bottomUpLabels
            )
            # print(dataContainer[T][ClassNameToSave])


def loadClassificationBottomUp(
        classificationFile: str,
        dataContainer: dict,
        NPname: str,
        bottomUpLabels: list,
        TimeWindow: slice = slice(None),
        ClassificationPlace: str = "/Classifications/ih55-SOAP4_4_4",
):
    loadClassification(
        classificationFile,
        dataContainer,
        NPname,
        ClassificationPlace=ClassificationPlace,
        ClassNameToFind="labelsNN",
        ClassNameToSave="ClassBU",
        bottomUpLabels=bottomUpLabels,
        TimeWindow=TimeWindow
    )


def pcaLoaderBottomUp(filename: str, PCAGroupAddr: str, TimeWindow: slice = slice(None)):
    dataContainer = dict()
    dataContainer["xlims"] = [numpy.finfo(float).max, numpy.finfo(float).min]
    dataContainer["ylims"] = [numpy.finfo(float).max, numpy.finfo(float).min]
    with h5py.File(filename, "r") as f:
        #pcas = f["/PCAs/ih55-SOAP4_4_4"]
        pcas = f[PCAGroupAddr]
        for k in pcas:
            dataContainer["nat"] = pcas[k].shape[1]
            #T = getT(k)
            T = 303
            dataContainer[T] = dict(
                pca=pcas[k][TimeWindow, :, :2].reshape(-1, 2))
            dataContainer["xlims"][0] = getMin(
                dataContainer["xlims"][0], dataContainer[T]["pca"][:, 0]
            )
            dataContainer["xlims"][1] = getMax(
                dataContainer["xlims"][1], dataContainer[T]["pca"][:, 0]
            )
            dataContainer["ylims"][0] = getMin(
                dataContainer["ylims"][0], dataContainer[T]["pca"][:, 1]
            )
            dataContainer["ylims"][1] = getMax(
                dataContainer["ylims"][1], dataContainer[T]["pca"][:, 1]
            )

    return dataContainer


def addPseudoFes(tempData, bins=300, rangeHisto=None):
    hist, xedges, yedges = numpy.histogram2d(
        tempData["pca"][:, 0],
        tempData["pca"][:, 1],
        bins=bins,
        range=rangeHisto,
        density=True,
    )
    tempData["pFES"] = -numpy.log(hist.T)
    tempData["pFES"] -= numpy.min(tempData["pFES"])
    tempData["pFESLimitsX"] = (xedges[:-1] + xedges[1:]) / 2
    tempData["pFESLimitsY"] = (yedges[:-1] + yedges[1:]) / 2


def plotTemperatureData(ax, T, data, xlims, ylims, bottomUpColorMap, zoom=0.01, smooth=0.0):
    pFES = data["pFES"]
    mymax = int(numpy.max(pFES[numpy.isfinite(pFES)]))
    if smooth > 0.0:
        t = numpy.array(pFES)
        t[numpy.isinf(t)] = numpy.max(t[numpy.isfinite(t)])
        pFES = gaussian_filter(t, sigma=smooth, order=0)

    # option for the countour lines
    countourOptions = dict(
        levels=10,
        colors="k",
        linewidths=0.1,
        zorder=2,
    )
    # the pca points
    ax.scatter(
        data["pca"][:, 0],
        data["pca"][:, 1],
        s=0.1,
        c=bottomUpColorMap[data["ClassBU"].references.reshape(-1)],
        alpha=0.5,
    )
    ax.contour(
        data["pFESLimitsX"],
        data["pFESLimitsY"],
        pFES,
        **countourOptions,
    )

    # pseudoFES representation
    pfesPlot = ax.contourf(
        data["pFESLimitsX"],
        data["pFESLimitsY"],
        pFES,
        levels=10,
        cmap=pFEScmap,
        zorder=1,
        extend="max",
        vmax=mymax,
    )

    ax.contour(
        data["pFESLimitsX"],
        data["pFESLimitsY"],
        pFES,
        **countourOptions,
    )

    cbar = plt.colorbar(
        pfesPlot,
        shrink=0.5,
        aspect=10,
        orientation="vertical",
        # cax=axes[f"cbarFes{T}Ax"],
        label="$[k_BT]$",  # "_{\!_{" + self.sub + "}}"
    )


def plotPCAData(ax, T, data, xlims, ylims, bottomUpColorMap, zoom=0.01, smooth=0.0):
    pFES = data["pFES"]
    mymax = int(numpy.max(pFES[numpy.isfinite(pFES)]))
    if smooth > 0.0:
        t = numpy.array(pFES)
        t[numpy.isinf(t)] = numpy.max(t[numpy.isfinite(t)])
        pFES = gaussian_filter(t, sigma=smooth, order=0)

    # option for the countour lines
    countourOptions = dict(
        levels=10,
        colors="k",
        linewidths=0.1,
        zorder=2,
    )
    # the pca points
    ax.scatter(
        data["pca"][:, 0],
        data["pca"][:, 1],
        s=0.1,
        c=bottomUpColorMap[data["ClassBU"].references.reshape(-1)],
        alpha=0.5,
    )
    ax.contour(
        data["pFESLimitsX"],
        data["pFESLimitsY"],
        pFES,
        **countourOptions,
    )

#%%
