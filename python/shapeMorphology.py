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
import subprocess



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
        self.angleRadiusArray=[]
        self.saveFileName=""
        self.t =turtle.Turtle()

    def confidence_interval(self, data, alpha=0.10):
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

    def calc_dist(self,x0,y0,x1,y1):
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
                txValues=[]
                xValues=[]
                tyValues=[]
                yValues=[]
                rowValues = map(str,row)
                rowValues.pop(0)
                txValues=rowValues[1::2]
                tyValues=rowValues[2::2]
                for m in txValues:
                    xValues.append(float(m))
                for n in tyValues:
                    if n not in ("",None):
                        #print "valu: ", n
                        yValues.append(float(n))
                xyPoints.append(xValues)
                xyPoints.append(yValues)
                self.xyArray.append(xyPoints)
                self.numberOfPoints=len(xValues)
                self.centroidArray.append(self.calculateCentroid(xyPoints))
                self.itemCount += 1
                #print "Now on item: ", self.itemCount
            rowcount += 1
        file.close()
        self.saveFileName=self.filename[0:-4]+"-out.eps"


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
            radius = self.radiiCalc(centroid,item)
            circleArray = self.angleRadiiCalc(centroid,item)
            #print "circleArray: ", circleArray
            #print "length of circleArray: ", len(circleArray)
            self.angleRadiusArray.append(circleArray)

        for angle in range(0,360):
            distanceArray = []
            #print "angleRadiusArray: ", self.angleRadiusArray
            for a in self.angleRadiusArray:
                #print "length of a: ", len(a), "-", angle
                distanceArray.append(a[angle])
            mean,low,high = self.confidence_interval(distanceArray)
            self.meanArray.append(mean)
            self.lowArray.append(low)
            self.highArray.append(high)

        self.drawing()
        self.saveFigure()
        print('Hit any key to exit')
        dummy = input()

    def drawing(self):
        self.screen.title(self.filename)
        self.screen_x, self.screen_y=self.screen.screensize()
        self.drawOutline(self.meanArray,"mean")
        self.drawOutline(self.lowArray,"low")
        self.drawOutline(self.highArray,"high")

    def drawOutline(self,pointArray,kind):

        self.t.ht()
        ## go through points one by one based on distance
        #print "length of point array: ", len(pointArray)
        #angleIncrement=360.0/float(self.numberOfPoints)
        currentAngle=-1.0
        self.t.penup()
        self.t.setx(0)
        self.t.sety(0)
        scale=1000
        m=0
        offset=0
        if kind == "mean":
            self.t.pen(fillcolor="white", pencolor="black", pensize=1)
            offset=5
        elif kind == "low":
            self.t.pen(fillcolor="white", pencolor="red", pensize=1)
            offset=4.5
        else:
            self.t.pen(fillcolor="white", pencolor="blue", pensize=1)
            offset=5.5
        for radius in pointArray:
            currentAngle += 1
            x = radius*math.sin(math.radians(currentAngle))*offset*scale
            y = radius* math.cos(math.radians(currentAngle))*offset*scale
            if m==0:
                self.t.penup()
                m=1
                x1=x
                y1=y
            else:
                self.t.pendown()
            self.t.goto(x,y)
        self.t.goto(x1,y1)


    def angleRadiiCalc(self,centroid,arrayOfPoints):
        #print "array of points:", arrayOfPoints
        angleHash={}
        circleArray = []
        x=arrayOfPoints[0]
        y=arrayOfPoints[1]
        for p in range(0,len(x)-1):
            dist = self.calc_dist(centroid[0],centroid[1],x[p],y[p])
            angle= self.calc_angle(centroid[0],centroid[1],x[p],y[p])
            angleHash[ angle ] = dist

        circleArray = self.circularization(angleHash)
        # returns 0-359 at each 1 degree interval
        return circleArray

    def circularization(self, angleHash):
        circleArray=[]
        for a in range(0,360):
            closest_value= self.get_closest_distance(angleHash,a)
            circleArray.append(closest_value)
        return circleArray

    def radiiCalc(self,centroid,arrayOfPoints):
        #print "array of points:", arrayOfPoints
        distArray=[]
        x=arrayOfPoints[0]
        y=arrayOfPoints[1]
        for p in range(0,len(x)-1):
            dist = self.calc_dist(centroid[0],centroid[1],x[p],y[p])
            distArray.append(dist)
        return distArray

    def calc_angle(self,x0,y0,x1,y1):
        dx = x0 - x1
        dy = y0 - y1
        rads = math.atan2(-dy,dx)
        rads %= 2*math.pi
        degs = math.degrees(rads)
        return degs

    def get_closest_distance(self,angleHash, angle):
        closestValue= angleHash.get(angle, angleHash[min(angleHash.keys(), key=lambda k: abs(k-angle))])
        return closestValue

    def checkMinimumRequirements(self):
        if self.args['inputfile'] in (None, ""):
            sys.exit("Inputfile is a required input value: --inputfile=../testdata/testdata.txt")

    def saveFigure(self):
        ts = self.t.getscreen()
        ts.getcanvas().postscript(file=self.saveFileName)
        command="ps2pdf "+ self.saveFileName+" "+ self.saveFileName[:-4]+".pdf"
        process = subprocess.Popen(command, shell=True)

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




'''''
