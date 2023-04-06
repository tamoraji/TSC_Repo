# Runs on Python 2.7 with scipy library installed.
# Computes win/loss/tie and wilcoxon p value
# for two time series result folders.

from scipy.stats import wilcoxon
from glob import glob
import re

def main(dir1, dir2) :
    d1 = glob(dir1+"/*")
    d2 = glob(dir2+"/*")
    e1 = []
    names1 = []
    for d in d1:
        infile = open(d,"r")
        names1 = names1 + [d.replace(dir1+"\\","").replace(".txt","")]
        for line in infile:
            m = re.search("rror: ([\.\d]+)\s",line)
            if (m) :
                e1 = e1 + [eval(m.group(1))]
                break
    e2 = []
    names2 = []
    for d in d2:
        infile = open(d,"r")
        names2 = names2 + [d.replace(dir2+"\\","").replace(".txt","")]
        for line in infile:
            m = re.search("rror: ([\.\d]+)\s",line)
            if (m) :
                e2 = e2 + [eval(m.group(1))]
                break
    if (len(e1)==len(e2)) :
        win = 0
        loss = 0
        tie = 0
        for i in range(len(e1)) :
            if (names1[i] != names2[i])  :
                print "Error: Different dataset names -",names1[i],names2[i]
                exit
            #print names1[i]," &  &  &  &  & ",round(e2[i],3)," & ",round(e1[i],3)," & ",round(100*(e2[i]-e1[i])/e2[i],2),"\\\\"
            #if (e2[i] > 0) :
              #  print "Percentage error: ", round(100*(e2[i]-e1[i])/e2[i],2)
            #print "\hline"
            #print e1[i], e2[i],"abc"
            if (e1[i] < e2[i]) :
                win += 1
            if (e1[i] > e2[i]) :
                loss += 1
            if (e1[i] == e2[i]) :
                tie += 1
        a,p=wilcoxon(e1,e2)
        print "Total datasets =",len(e1)
        print "Win =",win,"Loss =",loss,"Tie =",tie
        print "p-value from Wilcoxon's test =",p
        outfile = open("plot","w")
        for i in range(len(e1)) :
            outfile.write(str(e1[i])+" "+str(e2[i])+"\n")
        outfile.close()   
    else:
        print "Lengths not equal: ",len(e1),len(e2)
        

    
    
    
