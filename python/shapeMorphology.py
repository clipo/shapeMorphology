__author__ = 'clipo'

import sys
import csv
import argparse
import math
import turtle
import subprocess

import scipy as scipy
import scipy.stats
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


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
        #self.screen=turtle.Screen()
        self.angleRadiusArray=[]
        self.saveFileName=""
        #self.t =turtle.Turtle()
        self.minArray=[]
        self.maxArray=[]
        self.scale=0
        self.lengthArray=[]
        self.params = dict(projection='polar', theta_offset=np.pi/2)
        self.fig,self.ax = plt.subplots(subplot_kw=self.params)

    def confidence_interval(self, data, alpha=0.10):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t._ppf((1 + alpha) / 2., n - 1)

        minimum=min(data)
        maximum=max(data)
        return m, m - h, m + h,minimum,maximum

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

                count =0
                length=0
                #print "# of points: ", len(xValues)
                for m in range(0,len(xValues)-1):
                    if count==0:
                        startX=float(xValues[0])
                        startY=float(yValues[0])
                    else:
                        #print "from x: ", startX , "y: ", startY, " to x: ",  xValues[m], " Y: ", yValues[m]
                        length += self.calc_dist(startX,startY, xValues[m],yValues[m])
                        startX=float(xValues[m])
                        startY=float(yValues[m])
                    count += 1
                self.lengthArray.append(length)
                #print "Total Length: ", length

                self.itemCount += 1
                #print "Now on item: ", self.itemCount
            rowcount += 1
        maxLength = max(self.lengthArray)
        self.scale = 600/maxLength
        if maxLength<1 and float(args['fixedCentroidX'])>100:
            print "Check the location of the centroid - the scale appears to be off given that the maximum length of the outline is: ", maxLength
            #self.scale = 100
            #args['fixedCentroidX']=0
            #args['fixedCentroidY']=0
        #print "scale: ", self.scale
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
            if args['fixedCentroidX'] not in ("None", None, 0, "0", False, "False"):
                centroid = [float(args['fixedCentroidX']),float(args['fixedCentroidY'])]
            else:
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
            mean,low,high,min,max = self.confidence_interval(distanceArray)
            self.meanArray.append(mean)
            self.lowArray.append(low)
            self.highArray.append(high)
            self.minArray.append(min)
            self.maxArray.append(max)

        self.drawing()
        self.saveFigure()
        print('Hit any key to exit')
        dummy = input()

    def drawing(self):
        #self.screen.title(self.filename)
        #self.ax.set_title(self.filename, va='bottom')
        #self.screen_x, self.screen_y=self.screen.screensize()
        ##self.drawOutlineWithTurtle(self.meanArray,"mean")
        #self.drawOutlineWithTurtle(self.lowArray,"low")
        #self.drawOutlineWithTurtle(self.highArray,"high")
        #self.drawOutlineWithTurtle(self.minArray,"min")
        #self.drawOutlineWithTurtle(self.maxArray,"max")

        self.ax.set_title(self.filename)
        self.drawOutlineWithMatplotlib(self.meanArray,"mean")
        self.drawOutlineWithMatplotlib(self.lowArray,"low")
        self.drawOutlineWithMatplotlib(self.highArray,"high")
        self.drawOutlineWithMatplotlib(self.minArray,"min")
        self.drawOutlineWithMatplotlib(self.maxArray,"max")
        #self.addGraph()
        plt.show()


    def drawOutlineWithMatplotlib(self, pointArray, kind):

        '''set colors
        b: blue
        g: green
        r: red
        c: cyan
        m: magenta
        y: yellow
        k: black
        w: white
        '''

        if kind == "mean":
            color='c'
            style='solid'
        elif kind == "low":
            color='r'
            scaledArray =[]
            for p in pointArray:
                scaledArray.append(p*0.90)
            pointArray = scaledArray
            style='solid'
        elif kind=="high":
            color='b'
            scaledArray =[]
            for p in pointArray:
                scaledArray.append(p*1.10)
            pointArray = scaledArray
            style='solid'
        elif kind=="min" or kind=="max":
            color='m'
            style='dashed'
        else:
            pass
        angles = np.arange(0, 360, 1.0)
        theta = np.radians(angles)

        self.ax.plot(theta, pointArray, color=color, linewidth=1,linestyle=style)
        #self.ax.set_rmax(2.0)
        self.ax.grid(True)


    def drawOutlineWithTurtle(self,pointArray,kind):

        self.t.ht()
        ## go through points one by one based on distance
        #print "length of point array: ", len(pointArray)
        #angleIncrement=360.0/float(self.numberOfPoints)
        currentAngle=-1.0
        self.t.penup()
        self.t.setx(0)
        self.t.sety(0)

        m=0
        offset=0
        if kind == "mean":
            self.t.pen(fillcolor="white", pencolor="black", pensize=1)
            offset=3
        elif kind == "low":
            self.t.pen(fillcolor="white", pencolor="red", pensize=1)
            offset=2.75
        elif kind=="high":
            self.t.pen(fillcolor="white", pencolor="blue", pensize=1)
            offset=3.25
        elif kind=="min":
            self.t.pen(fillcolor="white", pencolor="gray", pensize=0.5)
            offset=3
        elif kind=="max":
            self.t.pen(fillcolor="white", pencolor="gray", pensize=0.5)
            offset=3
        else:
            pass
        for radius in pointArray:
            currentAngle += 1
            x = radius*math.sin(math.radians(currentAngle))*offset*self.scale
            y = radius* math.cos(math.radians(currentAngle))*offset*self.scale
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

    def addGraph(self):
        ts = self.t.getscreen()
        canvas=ts.getcanvas()
        midX=int(self.screen_x/2)
        midY=int(self.screen_y/2)
        self.t.pen(fillcolor="white", pencolor="gray", pensize=0.25)
        self.t.penup()
        self.t.goto(-2*self.screen_x,0)
        self.t.pendown()
        self.t.goto(2*self.screen_x,0)
        self.t.penup()
        self.t.goto(0,-2*self.screen_y)
        self.t.pendown()
        self.t.goto(0,2*self.screen_y)
        self.t.penup()
        self.t.goto(-2*self.screen_x,-2*self.screen_y)
        self.t.pendown()
        self.t.goto(2*self.screen_x,-2*self.screen_y)
        self.t.goto(2*self.screen_x,2*self.screen_y)
        self.t.goto(-2*self.screen_x,2*self.screen_y)
        self.t.goto(-2*self.screen_x,-2*self.screen_y)
        self.t.penup()

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
    parser.add_argument('--fixedCentroidX', default=0, help="Use this fixed point instead of a calculated centroid.")
    parser.add_argument('--fixedCentroidY', default=0, help="Use this fixed point instead of a calculated centroid.")
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
