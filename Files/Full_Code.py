#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program scrapes football data from fbref.com and analyzes it using continuous-treatment Causal Inference methods
"""


import statsmodels.api as sm
import tqdm
from scipy import stats
import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt

def teams_list(season):
    print("Preparing list of teams and opponents......")
    link = "https://fbref.com/en/comps/12/"+ season 
    html = requests.get(link).content
    soup = BeautifulSoup(html,'html.parser')
    table = soup.find("table")
    
    
    df = pd.read_html(StringIO(str(table)))[0]
    for i in range(len(df)):
        df.loc[i,'Links']=["https://fbref.com"+link.get('href') for link in table.find_all('a',string=df.loc[i,'Squad'])][0]
    cols = df[['Rk','Squad','Links']]
    teams = cols[cols['Rk']%2==1]   
    opponents= cols[cols['Rk']%2==0]
    teams.reset_index(drop=True,inplace=True)
    opponents.reset_index(drop=True,inplace=True)
    print("Done")
    return teams,opponents
    

def fixtures_table(team,team_link,opponents):
    team_soup = BeautifulSoup(requests.get(team_link).content,'html.parser')
    link = team_soup.select("a[href*=Scores-and-Fixtures-La-Liga]")[0]['href']
    html= requests.get("https://fbref.com"+link).content
    soup = BeautifulSoup(html,'html.parser')
    table = soup.find("table")
    df = pd.read_html(StringIO(str(table)))[0]
    
    report_links = table.find_all('a',string = 'Match Report')
    report_links = [tag['href'] for tag in report_links]
    df['Match Report']=report_links 
    df['Match Report'] = "https://fbref.com"+df['Match Report']
    fixtures = df[(df['Opponent'].isin(opponents['Squad']))]
    fixtures.reset_index(drop=True,inplace=True)
    return fixtures
    
def metrics(link):
    html = requests.get(link).content
    soup = BeautifulSoup(html,'html.parser')
    team_tables= [cap.find_parent() for cap in soup.find_all('caption',string=key+' Player Stats Table')]
    dfs = pd.read_html(StringIO(str(team_tables)))
    df0 = dfs[0]
    df1 = dfs[1]
    df0.columns = df0.columns.droplevel(0)
    df1.columns = df1.columns.droplevel(0)
    xA = df1['xA'].iloc[-1]
    touches = df0['Touches'].iloc[-1]
    return touches,xA

def LR(data,xvar,yvar):
    X = data[xvar]
    Y = data[yvar]
    Xm = sm.add_constant(X)
    #Xm = Xm - np.average(Xm) #center data if Cond.No>>1
    model_sm=sm.OLS(Y,Xm).fit()
    model_sm.summary()
    plt.title("{}  vs {}".format(yvar,xvar),fontsize=18)
    plt.xlabel(xvar,fontsize=18)
    plt.ylabel(yvar,fontsize=18)
    plt.plot(X,model_sm.predict(Xm),color='r',label='Fit')
    plt.scatter(X,Y,color='b',label='Data')
    plt.legend()
    print(model_sm.summary())
    return


def response(data,xvar,yvar,confs):

    confounders = ""
    for conf in confs:
        if type(conf[0])==str:
            confounders+= "+C({})".format(conf)
        else:
            confounders+= "+I({}) + I({}**2)".format(conf,conf)
    def conditional_densities(use_confounders=True):
        formula = "{} ~ 1".format(xvar)
        if use_confounders:
            formula += confounders
        model = sm.formula.ols(formula, data=data).fit()
        density = stats.norm(loc=model.fittedvalues,scale=model.resid.std(),)
        densities = density.pdf(data[xvar])
        densities = pd.Series(densities, index=model.fittedvalues.index)
        return densities

    denominator = conditional_densities(use_confounders=True)
    numerator = conditional_densities(use_confounders=False)
    generalized_ipw = numerator / denominator
    msm = sm.formula.wls("{} ~ 1 + {} +I({}**2) ".format(yvar,xvar,xvar),data=data,weights=generalized_ipw,).fit()
    dosage = list(range(296,878))
    dosage = pd.DataFrame(data={xvar:dosage,"I({}**2)".format(xvar):dosage},index=dosage,)
    response=msm.predict(dosage)
    ax = response.plot(color='k',label='Dose-Response')
    ax.set_title("{} vs {}".format(yvar,xvar),fontsize=18)
    ax.set_xlabel(xvar,fontsize=18)
    ax.set_ylabel(yvar,fontsize=18)
    return


Teams,Opponents = teams_list('2022-2023')

Fixtures={}
print("Collecting fixtures for")
for i in tqdm.tqdm(range(len(Teams))):
    print(Teams.iloc[i,1]+ "...")
    Fixtures[Teams.iloc[i,1]] = fixtures_table(Teams.iloc[i,1], Teams.iloc[i,2],Opponents)
print("Fixtures list complete")

print("Collecting Metrics for")
for key in Fixtures:
    print(key+"...")
    DF = Fixtures[key]
    DF['Team']=key
    dflinks = DF['Match Report']
    DF[['Touches','xA' ]] = pd.DataFrame(dflinks.apply(metrics).tolist(), index=DF.index)
    
rev = {'Almería':45.09,'Atlético Madrid':120.43,'Barcelona':155.1,'Celta Vigo':51.17,'Espanyol':51.24,'Getafe':53.38,'Mallorca':45.04,'Osasuna':49.65,'Rayo Vallecano':45.97,'Villarreal':63.35}
Data = pd.concat(Fixtures.values(),ignore_index =True)
Data['Revenue']=Data['Team'].map(rev)

response(Data,"Touches","xG",["Revenue"])
LR(Data,"Touches","xG")
