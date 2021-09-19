from os import read
from numpy import sqrt
import csv


def errorCalc(x1,y1,x2,y2):
    dists=list()
    dists.append('distances')
    for i in range(0,len(x2),1):
        dists.append(sqrt(((x2[i]-x1[i])**2)+((y2[i]-y1[i])**2)))
    return dists

def checkingVals(path):
    x1=list()
    y1=list()
    items=list()
    with open(path) as ODC:
        reader=csv.DictReader(ODC)
        items=[item for item in reader]
    x1=[float(item['x']) for item in items]
    y1=[float(item['y']) for item in items]
    return (x1,y1)

def OutputCreator(input_file, output_file, appCol):
   
    with open(input_file, 'r') as read_obj, \
            open(output_file, 'w', newline='') as write_obj:
        csv_reader = csv.reader(read_obj)
        csv_writer = csv.writer(write_obj)
        count=0
        for row in csv_reader:
            row.append(appCol[count])
            count=count+1
            csv_writer.writerow(row)



