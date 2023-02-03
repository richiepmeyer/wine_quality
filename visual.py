import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pydataset import data
from scipy import stats

#--------------------------------------------------------------------------------------------------

def barplot1(df):

    '''
    Function to plot the Total Acid/Alc+Sugar against Quality in a barplot
    '''

    sns.barplot(y=df.acid_alc_sugar, x=df.quality)
    plt.ylabel('Total Acidity / Alcohol + Sugar')
    plt.xlabel('Quality')
    plt.title('Quality and Total Acidity / Alcohol + Sugar')
    plt.show()

#--------------------------------------------------------------------------------------------------

def barplot2(df):

    '''
    Function to show the difference in quality between White wine and Red wine in a barplot.
    '''

    ax = sns.barplot(x='color_red', y='quality', data=df, palette=['wheat','firebrick'])
    ax.set_xticklabels(['White','Red'])
    plt.xlabel('Wine Color')
    plt.ylabel('Quality')
    plt.title('White Wine vs Red Wine')
    plt.show()

#--------------------------------------------------------------------------------------------------

def barplot3(df):

    '''
    Function to show the relation between Total Acidity/Chlorides and Quality
    '''

    sns.barplot(y=df.acid_chlor, x=df.quality)
    plt.ylabel('Total Acidity / Chlorides')
    plt.xlabel('Quality')
    plt.title('Quality and Total Acidity/Chlorides')
    plt.show()  

#--------------------------------------------------------------------------------------------------

def barplot4(df):

    '''
    Function to show the relation between Density and Quality.
    '''

    sns.barplot(y=df.density, x=df.quality)
    plt.ylabel('Density')
    plt.xlabel('Quality')
    plt.title('Quality and Density')
    plt.show()

#--------------------------------------------------------------------------------------------------

def spearman1(df):

    '''
    Spearman's R test for Total acidity / alcohol+sugar in relatioin to quality
    '''


    corr, p = stats.spearmanr(df.acid_alc_sugar, df.quality)


    print("Spearman's R Test Results")
    print('--------------------------')
    print(f'Correlation: {round(corr,2)}')
    print(f'P-value: {p}')

#--------------------------------------------------------------------------------------------------

def spearman2(df):

    '''
    Spearmans R test for Total acidity / chlorides in relation to quality
    '''

    corr, p = stats.spearmanr(df.acid_chlor, df.quality)


    print("Spearman's R Test Results")
    print('--------------------------')
    print(f'Correlation: {round(corr,2)}')
    print(f'P-value: {p}')

#--------------------------------------------------------------------------------------------------

def chi(df):


    red = df['color_red'] == 1
    qlt = df['quality']

    rw = pd.crosstab(qlt,red, colnames=['Red Wine'])

    chi, p, degf, expected = stats.chi2_contingency(rw)


    print('Chi-square Test Results')
    print('-----------------------')
    print(f'Chi value: {round(chi,2)}')
    print(f'p-value: {p}')

#--------------------------------------------------------------------------------------------------

def pearsonr(df):


    corr, p = stats.pearsonr(df.acid_alc_sugar, df.quality)


    print("Pearson's R Test Results")
    print('-------------------------')
    print(f'Correlation: {round(corr,2)}')
    print(f'P-value: {p}')