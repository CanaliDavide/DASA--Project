import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib as mpl

from collections import Counter

import scipy.cluster.hierarchy as shc

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import silhouette_score

def EggsPerSeason(dataSet, cycleName:str, wantNumber=False, colorSpring='green', colorSummer='orange', colorAutumn='brown', colorWinter='blue'):
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
    data.Data = pd.to_datetime(data.Data, format = '%m/%d/%Y')

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
    ----------
        the labels
    '''
    dataSet=dataSet.drop(columns=["Arrival Chickens Date","Date of Selling","Date of Laid"])

    scaler = MinMaxScaler()
    data=pd.DataFrame(scaler.fit_transform(dataSet.values), columns=dataSet.columns, index=dataSet.index)
    
    data = data[attributes]
    fig, ax1 = plt.subplots()
    ax1.set_title("Dendrogram")

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
    plt.show()
    return data_labels

def DensityClustering(dataSet, attributes):
    '''
    Cluster 2 variables of the same cycle

    Parameters:
    ----------
        dataSet: DataFrame
            the datset interested in cluster, a cycle
        attributes: array of str
            the 2 attributes interested in cluster 

    Return: 
    ----------
        the labels
    '''

    dataSet=dataSet.drop(columns=["Arrival Chickens Date","Date of Selling","Date of Laid"])

    scaler = MinMaxScaler()
    data=pd.DataFrame(scaler.fit_transform(dataSet.values), columns=dataSet.columns, index=dataSet.index)
    
    data = data[attributes]

    # n_neighbors = 5 as kneighbors function returns distance of point to itself (i.e. first column will be zeros) 
    nbrs = NearestNeighbors(n_neighbors=5).fit(data)
    # Find the k-neighbors of a point
    neigh_dist, neigh_ind = nbrs.kneighbors(data)
    # sort the neighbor distances (lengths to points) in ascending order
    # axis = 0 represents sort along first axis i.e. sort along row
    sort_neigh_dist = np.sort(neigh_dist, axis=0)

    k_dist = sort_neigh_dist[:, 4]
    plt.plot(k_dist)
    plt.grid()
    plt.yticks(np.arange(min(k_dist), max(k_dist)+0.01, 0.01))
    plt.ylabel("k-NN distance")
    plt.xlabel("Sorted observations (4th NN)")
    plt.show()

    eps = input("Epsilon?")
    cluster = DBSCAN(eps=float(eps), min_samples=4).fit(data)

    data_labels = cluster.labels_
    sns.scatterplot(x=attributes[0], y=attributes[1],data=data, hue=data_labels,legend="full", palette="rainbow").set_title('Clusters')
    plt.show()
    return data_labels

def KMeanClustering(dataSet, attributes):
    '''
    Cluster 2 variables of the same cycle

    Parameters:
    ----------
        dataSet: DataFrame
            the datset interested in cluster, a cycle
        attributes: array of str
            the 2 attributes interested in cluster 

    Return: 
    ----------
        the labels
    '''

    dataSet=dataSet.drop(columns=["Arrival Chickens Date","Date of Selling","Date of Laid"])

    scaler = MinMaxScaler()
    data=pd.DataFrame(scaler.fit_transform(dataSet.values), columns=dataSet.columns, index=dataSet.index)
    
    data = data[attributes]
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
    silhouette_avg = []
    for num_clusters in range_n_clusters:
        # initialise kmeans
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(data)
        cluster_labels = kmeans.labels_
        
        # silhouette score
        silhouette_avg.append(silhouette_score(data, cluster_labels))
    plt.plot(range_n_clusters,silhouette_avg, 'bx-')
    plt.xlabel('Values of K') 
    plt.ylabel('Silhouette score') 
    plt.title('Silhouette analysis For Optimal k')
    plt.show()
    n_cluster = input("How many cluster do you want?")
    kmeans = KMeans(n_clusters=int(n_cluster))
    kmeans.fit(data)
    data_labels = kmeans.labels_
    sns.scatterplot(x=attributes[0], y=attributes[1],data=data, hue=data_labels,legend="full", palette="rainbow").set_title('Clusters')
    plt.show()
    return data_labels

def TemporalCluster(dataSet, dataLabels ,attributes, attribute):
    '''
        Plot a cluster in a temporal plot

    Parameters:
    ----------
        dataSet: DataFrame
            the datset interested in cluster, a cycle
        attributes: array of str
            the 2 attributes interested in cluster 
        attribute: str
            the attribute to plot

    Return:
    '''

    data_date=dataSet.rename(columns={"Date of Laid": "Date"})
    data=dataSet.drop(columns=["Arrival Chickens Date","Date of Selling","Date of Laid"])
    
    scaler = MinMaxScaler()

    data_orig = data

    data=pd.DataFrame(scaler.fit_transform(data.values), columns=data.columns, index=data.index)
    
    data_date.Date = pd.to_datetime(data_date.Date, format = '%m/%d/%Y')
    data2 = data[attributes]

    label_group = Counter(dataLabels)
    print(label_group)
    elimFrom = int(input("Min number of elements per cluster? (-1 if none)"))
    if elimFrom != -1:
        for el,count in label_group.items():
            if count<= elimFrom:
                for i in range(len(dataLabels)):
                    if dataLabels[i] == el:
                        dataLabels[i] = -1
        label_group = Counter(dataLabels)
        print(label_group)

    cmap = mpl.colormaps['rainbow']
    keys = list(label_group.keys())
    
    colors = []

    num = len(label_group.keys())-1
    step = 1.0/num

    for i in range(num+1):
        colors.append(cmap(step*i))

    sns.scatterplot(x=attributes[0], y=attributes[1],data=data2, hue=dataLabels,legend="full", palette=colors).set_title('Clusters')


    data=data_orig

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Date')
    ax1.set_ylabel(attribute)
    #Select to plot just years
    ax1.xaxis.set_major_locator(mpl.dates.YearLocator())
    ax1.xaxis.set_minor_locator(mpl.dates.MonthLocator((1,4,7,10)))
    ax1.xaxis.set_major_formatter(mpl.dates.DateFormatter("\n%Y"))
    ax1.xaxis.set_minor_formatter(mpl.dates.DateFormatter("%b"))
    # fig.set_figwidth(20)
    plt.setp(ax1.get_xticklabels(), rotation=0, ha="center")
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=31))

    keys.sort()
    print(keys)


    for i in range(len(dataLabels)):
        color = colors[keys.index(dataLabels[i])]
        
        ax1.plot(data_date["Date"][i],data[attribute][i],"x", color=color)

        month =  data_date.Date.dt.month[i]
        if (month==9)|(month==10)|(month==11):
            color='brown'
        elif (month==12)|(month==1)|(month==2):
            color = 'blue'
        elif (month==3)|(month==4)|(month==5):
            color = 'green'
        elif (month==6)|(month==7)|(month==8):
            color = 'orange'
        ax1.axvline(data_date["Date"][i], color=color,alpha=0.05)

    plt.gcf().autofmt_xdate()
    plt.xticks(rotation=45)
    plt.show()

def CycleClusterAnalysis(dataSet, attributesArray):
    '''
    Cluster the cycles with different pairs of attributes

    Parameters:
    ----------
        dataSet: DataFrame
            the datset interested in cluster, a cycle
        attributes: array of array of str
            the array of attribute's pairs to cluster 

    Return: 
    ----------
        the disctionary of all labels per pair
    '''
    allLabels = {}
    labels = {"Hierarchical" : [], "Density" : [], "KMeans" : []}

    for attributes in attributesArray:
        print(attributes)
        print("Hierarchical:")
        labels["Hierarchical"] = HierarchicalClutering(dataSet, attributes)
        print("Density:")
        labels["Density"] = DensityClustering(dataSet, attributes)
        print("KMeans:")
        labels["KMeans"] = KMeanClustering(dataSet, attributes)
        allLabels.update({attributes[0] + "-" + attributes[1] : labels})
    return allLabels