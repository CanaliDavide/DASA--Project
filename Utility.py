import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

import scipy.cluster.hierarchy as shc

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering

def EggsPerSeason(dataSet, cycleName:str, wantNumber=False, colorSpring='pink', colorSummer='orange', colorAutumn='brown', colorWinter='blue'):
    """
    Graph of a Cycle wherea each season has a different color

    Parameters:
    ----------
        dataSet: DataFrame
            a DataFrame
        cycleName: str
            a string that identify the cycle
        wantNumber: bool
            true if you want the amount of egg produced per day, false if you want the % of laied per day
        colorSpring: str
            color for spring data, default pink
        colorSummer: str
            color for spring data, default orange
        colorAutumn: str
            color for autumn data, default brown
        colorWinter: str 
            color for winter data, default blue
    """
    yVar="none"
    if wantNumber:
        yVar="Eggs"
    else:
        yVar="Laied"

    data=dataSet.rename(columns={"Date of Laid": "Data", yVar:"EggsProduced"}) #, "Price(euro/100kg)":"Price"-> to show the market demand on top of the production
    data.Data = pd.to_datetime(data.Data, format = '%d/%m/%Y')

    data['month']=data.Data.dt.month
    dataAutumn = data.loc[(data.month==9)|(data.month==10)|(data.month==11)]
    dataWinter = data.loc[(data.month==12)|(data.month==1)|(data.month==2)]
    dataSpring = data.loc[(data.month==3)|(data.month==4)|(data.month==5)]
    dataSummer = data.loc[(data.month==6)|(data.month==7)|(data.month==8)]


    fig, ax1 = plt.subplots()
    ####ax2=ax1.twinx() -> to show the market demand on top of the production
    #ax1.set_ylim(0,1)
    ax1.set_xlabel('Date')
    ax1.set_ylabel(str(yVar)+"", color='black')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=31))

    ax1.plot(dataAutumn.Data,dataAutumn.EggsProduced, 'x', color=colorAutumn, label="Autumn")

    ax1.plot(dataWinter.Data,dataWinter.EggsProduced,'x', color=colorWinter, label="Winter")

    ax1.plot(dataSpring.Data,dataSpring.EggsProduced,'x', color=colorSpring, label="Spring")

    ax1.plot(dataSummer.Data,dataSummer.EggsProduced,'x', color=colorSummer, label="Summer")

    
    ####ax2.plot(data.Data, data.Price, 'o', color="black") -> to show the market demand on top of the production
    
    plt.gcf().autofmt_xdate()
    ax1.tick_params(axis='y')
    plt.xticks(rotation=45)
    

    plt.legend()    
    plt.title(""+str(yVar)+' per day - '+ cycleName)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def AveragePerDays(data, days):
    """
    Do the average of days and create a new array
    Parameters:
        ----------
        data: array
            the orginal array
        days: int
            amount of day to group up
    
    Return:
    ----------
        var: a new vector fullfilled with the average
    """
    var = []
    day=0
    sum=0
    for i in data:
        day=day+1
        sum=sum+i
        if day==days:
            day=0
            var.append(sum/days)
            sum=0
    return var

def CompareVariableInCycles(variables, nameVariable, legendLabels=['Y', 'Z', 'A', 'B', 'C'], dayStart=0, dayEnd=1000, averagePerDays=False, numOfDay=7):
    """
    Plot over time the all the data placed in variables.
    Parameters:
    ----------
        variables: array
            array of all the arrays to plot
        nameVariable: str
            name of the variable
        legendLabels: array
            array of labels to show in legend 
        dayStart: int
            first day to consider in the graph
        dayEnd: int
            last day to consider in the graph
        averagePerDays: bool
            True if you want to plot the average over a certain amount of days
        numOfDay: int
            amount of days to regroup
    """
    count=0
    for e in variables:
        e = e[dayStart:dayEnd]
        if(averagePerDays):
            e=AveragePerDays(e, numOfDay)
        plt.plot(e, label=(legendLabels[count]))
        count=count+1
    
    if(averagePerDays):
        plt.title(nameVariable+" comparison - Average over "+str(numOfDay) +" days" )
        plt.xlabel("# of periods of " + str(numOfDay)+ " days in barns")
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    else:
        plt.title(nameVariable+" comparison")
        plt.xlabel("Days in barns")
    plt.ylabel(nameVariable)
    plt.legend()
    
    plt.show()

def HierarchicalClutering(dataSet, attributes):
    '''
    Cluster 2 variables of the same cycle

    Parameters:
    ----------
        dataSet: DataFrame
            the datset interested in cluster, a cycle
        attributes: array of str
            the 2 attributes interested in cluster 

    Return:
    '''
    dataSet=dataSet.drop(columns=["Arrival Chickens Date","Date of Selling","Date of Laid"])

    scaler = MinMaxScaler()
    data=pd.DataFrame(scaler.fit_transform(dataSet.values), columns=dataSet.columns, index=dataSet.index)
    
    data = data[attributes]
    plt.figure()
    plt.title("Dendrogram")

    # Selecting Annual Income and Spending Scores by index
    selected_data = data
    clusters = shc.linkage(selected_data, 
                method='ward', 
                metric="euclidean")
    shc.dendrogram(Z=clusters)
    plt.show()

    n_cluster = input("How many cluster do you want?")
    clustering_model = AgglomerativeClustering(n_clusters=int(n_cluster), affinity='euclidean', linkage='ward')
    clustering_model.fit(selected_data)
    data_labels = clustering_model.labels_ #print the clust per item
    sns.scatterplot(x=attributes[0], y=attributes[1],data=selected_data, hue=data_labels, palette="rainbow").set_title('Clusters')