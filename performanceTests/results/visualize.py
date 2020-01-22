#! /usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
from matplotlib import pyplot as plt

""" Change Layout Default Settings"""
def setRcParams():
  params = {'axes.facecolor' : 'white',
            'axes.labelsize' : 'x-large', # xx-large
            'axes.titlesize' : 'x-large', # xx-large
            'axes.titlepad' : '12',
            'axes.formatter.limits' : '-4, 4',
            'xtick.labelsize' : 'large',  # xx-large
            #'xtick.direction' : 'out',
            'xtick.major.size' : '7', 
            'xtick.minor.size' : '4',
            'xtick.major.width' : '1.6',
            'xtick.minor.width' : '1.2',
            'xtick.major.pad' : '7',
            'xtick.minor.pad' : '6.8',
            'ytick.major.size' : '7', 
            'ytick.minor.size' : '4',
            'ytick.major.width' : '1.6',
            'ytick.minor.width' : '1.2',
            'ytick.major.pad' : '7',
            'ytick.minor.pad' : '6.8',
            #'ytick.direction' : 'out',
            'ytick.labelsize' : 'xx-large',
            'legend.fontsize' : 'x-large',
            'image.cmap' : 'jet', 
            'savefig.dpi' : '300', 
            'savefig.transparent' : 'False'}
  plt.rcParams.update(params)


def colorGen():
  colors = ['g', 'red', 'y', 'b']
  num = 0
  while True:
    yield colors[num]
    num = (num+1) % len(colors)
   

def readCycleCount(fname):
  workers = []
  cycleMin = []
  cycleMean = []
  cycleMax = []
  f = open(fname, 'r')
  while True:
    line = f.readline()
    if not line:
      break
    if line[0] == '#':
      continue
    chunkSize, maxChunksPerAlloc, mallocMC, \
       waR, tsR, gs, bs, arrCnt, minArr, maxArr  = [int(s) for s in line.split(' ')]
    workers.append(gs*bs)
    cmin, csum, cmax = 2e18, 0, 0
    for i in range(minArr, maxArr):
      line = f.readline()
      n = [int(s) for s in line.split(' ')]
      cnt = n[1]
      cmin = min(n[4], cmin) if cnt > 0 else cmin
      csum, cmax = csum + cnt*n[5], max(n[6], cmax)
    cycleMin.append(cmin)
    cycleMean.append(csum/tsR)
    cycleMax.append(cmax) 
  f.close()
  return workers, cycleMin, cycleMean, cycleMax



def visualizeCycleCount(fnameList, legList, title, showLegend=True):
  setRcParams()
  bx = 10
  fig = plt.figure("mC", figsize=(8, 6))
  ax = fig.add_subplot(111)
  # legList = ['new', 'MC1', 'MC2', 'MC3']
  for fname, color, leg in zip(fnameList, colorGen(), legList):
    print (fname)
    workers, cmin, cmean, cmax = readCycleCount(fname)
    ax.loglog(workers, cmin, ls='--', basex=bx, color=color, lw=0.5)
    ax.loglog(workers, cmean, basex=bx, color=color, label=leg)
    ax.loglog(workers, cmax, ls='--', basex=bx, color=color, lw=0.5)

  ax.set_xlabel("# workers")
  ax.set_ylabel("# cycles")
  ax.set_title(title)
  if showLegend:
    ax.legend()
  plt.show()

if __name__ == "__main__":
  if len(sys.argv) > 2 and len(sys.argv) % 2 == 0:
    visualizeCycleCount(sys.argv[2::2], sys.argv[3::2], sys.argv[1])
  elif len(sys.argv) == 3:
    visualizeCycleCount([sys.argv[2]], ['0'], sys.argv[1], showLegend=False)
  else:
    print ("Shows a single diagram of needed cycles to allocate memory")
    print ("First param: diagram title")
    print ("Every next two [optional params]: fname legendName")
    print ("If only one file name is given the legend name can be omitted")
