__author__ = 'clipo'

import sys
import csv
import argparse
import numpy as np
import scipy as scipy
import scipy.stats
import random as rnd
import scikits.bootstrap as bootstrap
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import operator
import math
import turtle



class shapeMorphology():

    def __init__(self):
        self.xyArray=[]
        self.itemCount=0
        self.centroidArray=[]
        self.numberOfPoints=0
        self.radiiArray=[]
        self.confidenceIntervalArray=[]
        self.meanArray=[]
        self.lowArray=[]
        self.highArray=[]
        self.filename=""
        self.screen_x=300
        self.screen_y=300
        self.screen=turtle.Screen()


    def confidence_interval(self, data, alpha=0.05):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t._ppf((1 + alpha) / 2., n - 1)
        return m, m - h, m + h

    def calculateCentroid(self,xyArray):
        centroid = (sum(xyArray[0])/len(xyArray[0]),sum(xyArray[1])/len(xyArray[1]))
        return centroid

    def dist(self,x,y):
        return np.sqrt(np.sum((x-y)**2))

    def calc_dist(x0,y0,x1,y1):
        return math.sqrt((x0-x1) ** 2 +
                     (y0-y1) ** 2 )

    def openFile(self,filename):
        try:
            file = open(filename, 'r')
        except csv.Error as e:
            sys.exit('file %s does not open: %s') % ( filename, e)
        self.filename=filename
        reader = csv.reader(file, delimiter='\t', quotechar='|')
        rowcount = 0
        for row in reader:
            if rowcount>0:  # skip first row
                xyPoints=[]
                row = map(str,row)
                xValues=row[1::2]
                yValues=row[2::2]
                xyPoints.append(xValues)
                xyPoints.append(yValues)
                xyArray.append(xyPoints)
                self.numberOfPoints=len(xValues)
                self.centroidArray.append(self.calculateCentroid(xyArray))
                self.itemCount += 1
        file.close()


    def addOptions(self, oldargs):
        self.args = {'debug': None, 'alpha': None,
                'inputfile': None, 'outputdirectory': None }
        for a in oldargs:
            self.args[a] = oldargs[a]

    def conductAnalysis(self,args):
        self.addOptions(args)
        self.checkMinimumRequirements()
        self.openFile(self.args['inputfile'])
        for item in self.xyArray:
            centroid = self.calculateCentroid(item)
            self.radiiArray.append(self.radiiCalc(centroid,item))

        for num in range(0,self.numberOfPoints):
            angleArray=[]
            for radii in self.radiiArray:
                angleArray.append(radii[num])
            mean,low,high = self.confidence_interval(angleArray)
            self.meanArray.append(mean)
            self.lowArray.append(low)
            self.highArray.append(high)


    def drawing(self):
        self.screen.title(self.filename)
        self.screen_x, self.screen_y=self.screen.screensize()
        self.drawOutline(self.meanArray,"mean")
        self.drawOutline(self.lowArray,"low")
        self.drawOutline(self.highArray,"high")

    def drawOutline(self,pointArray,kind):
        t =turtle.Turtle()

        ## go through points one by one based on distance

        angleIncrement=360.0/float(self.numberOfPoints)
        currentAngle=-1.0*angleIncrement
        t.setx(0)
        t.sety(0)
        m=0
        if kind == "mean":
            t.pen(fillcolor="white", pencolor="black", pensize=1)
        elif kind == "low":
            t.pen(fillcolor="white", pencolor="red", pensize=1)
        else:
            t.pen(fillcolor="white", pencolor="blue", pensize=1
            )
        for radius in pointArray:
            currentAngle += angleIncrement
            x = radius * math.sin(currentAngle)
            y = radius * math.cos(currentAngle)
            if m==0:
                m=1
            else:
                t.pendown()
            t.goto(x,y)


    def radiiCalc(self,centroid,arrayOfPoints):
        distArray=[]
        for xy in arrayOfPoints:
            dist = self.calc_dist(centroid[0],centroid[1],xy[0],xy[1])
            distArray.append(dist)
        return distArray



    def checkMinimumRequirements(self):
        if self.args['inputfile'] in (None, ""):
            sys.exit("Inputfile is a required input value: --inputfile=../testdata/testdata.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='shape analysis')
    parser.add_argument('--debug', default=None, help='Sets the DEBUG flag for massive amounts of annotated output.')
    parser.add_argument('--alpha', default=0.05, type=float,
                        help="The alpha significance to which the confidence intervals are calculated. Default is 0.05.")
    parser.add_argument('--inputfile',
                        help="<REQUIRED> Enter the name of the data file to process. Default datatype is from PAST.")
    parser.add_argument('--outputdirectory', default=None,
                        help="If you want the output to go someplace other than the /output directory, specify that here.")
    try:
        args = vars(parser.parse_args())
    except IOError, msg:
        parser.error(str(msg))
        sys.exit()


    shapeMorphology = shapeMorphology()
    shapeMorphology.conductAnalysis(args)

''''
From the command line:

python ./shapeMorphology.py --inputfile=../testdata/bifaces.txt"


As a module:

from shapeMorphology import shapeMorphology

shapeMorphology = shapeMorphology()

args={'inputfile':'../testdata/biface.txt','debug':1 }

shapeMorphology.conductAnalysis(args)




'''''__author__ = 'clipo'
