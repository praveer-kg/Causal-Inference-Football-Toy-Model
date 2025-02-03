<script src="https://rawcdn.githack.com/oscarmorrison/md-page/master/md-page.js"></script><noscript>

<h1> Toy Model for Causal Effect of Possession on Expected Goals</h1>
<h2>Introduction</h2>
<p align="justify">Possession in football is one of the most popular metrics used to guage a team's performance in a given game, tournament or season. Qualitatively, a higher share of posesssion is associated with a team that dominates the flow and tempo of a game. Quantitatively, it is well known that a team’s share of possession is correlated with the number of goals scored and chances created. However, the enduring success of counter-attacking football&mdash; exemplified by infamous victories of Chelsea, Inter Milan and Bayern Munich over possession-heavy 2010s Barcelona&mdash; begs the question of whether causation really underpins this correlation. <br>
Causal Inference techniques have been developed to answer precisely such a question when confounding factors can be identified.<p>

<h2>Causal Inference</h2>
Causal Inference(CI) methods were developed to faciliate the inference of causal relationships from observational data, in the absence of randomized controlled trials(RCTs). The standard trial involves a binary treatment(eg. drug administered or not) on a group of participants, selected carefully to represent a uniform distrubution of attributes(age,gender, etc.). The outcome(eg: incidence) is monitored over a period of time to provide insights into the efficacy of treatment. Using CI techniques, it is possible to replicate the benifits of an RCT while avoiding its drawbacks. The standard procedure is to regress the treatment variable against the confounders to obtain a distribution for the treatment. The outcome is then regressed against this distribution with weights determined by inverse probability of treatment(IPWT). For the question of Possession vs Expected Goals in football, the treatment(Touches taken) not being a binary variable calls for the application of continuous treatment CI methods. More information about such methods is available in this article.
<h3>Confounding Factors</h3>
<p align="justify">
A confounding factor is a variable that affects both the Treatment and Outcome of a given procedure, which in our case are the Possession and Expected Goals(xG), respectively. Expected Goals is an advanced football statistic that calculates the probablity that a given shot(kick of the ball) results in a goal. Possession in this study is represented by the touches taken by a team/player throughout the course of a game. Assuming a correlation between these two metrics, the most relevant question is how much of that correlation is a result of the biggest teams simply having both the players to create higher xG and the expectation to dominate play during matches. In other words, the 'stature' of a team is a confounder that needs to be controlled for. I choose to quantify this attribute by the <b>club revenue</b>, specifically audiovisual revenue, which is readily available for La Liga clubs. A second consideration will be the <b>match venue</b>(Home/Away): teams tend to adopt more aggressive strategies at home while playing more conservatively on away grounds, with consequences for both chance creation and possession share. Controlling for the venue of the game, therefore, would eliminate its backdoor influence as well. 
<p>

<h3>Directed Acyclic Graph</h3>
Every Causal Model makes assumptions about the causal relationships between various factors. These assumptions can be depicted in a graph with vertices representing the various causes and effects, and directed edges indicating causal relationships. The graph below is the DAG corresponding to our assumptions. 

| ![DAG.jpg](https://github.com/praveer-kg/Causal-Inference-Football-Toy-Model/blob/main/Files/DAG.png) | 
|:--:| 
| Figure 1: Directed Acyclic Graph for the current model(via DAGitty)|


Note that while the causal model presented here is only concerned with four vertices, the other two will be relevant for the discussion ahead. It goes without saying that the graph above is an oversimplified version of reality, but football is a complicated web of overlapping factors that no single model can accurately predict and track. The goal of the analysis here is to merely examine the influence of two extra variables on the correlation between Touches and Expected Goals.
<h2>Data Preprocessing</h2>
<p> All data used in this model is collected from fbref.com, an online database of advanced football statistics. The analysis includes games from a single season, namely $2022-2023$ La Liga, with $n=20$ teams playing each other twice in the league. However, including all teams in the analysis would violate the Stable Unit Treatment Value Assumption, essential to causal inference methods. To sidestep this issue, $m$ teams will be chosen from the total number of teams($n$), resulting in $2m(n-m)$ possible games in total, for a maximum of $\frac{n^2}{2}$ games available for analysis. For La Liga $22-23$, this corresponds to a maximum of $10$ teams whose fixtures against the other $10$ teams are collected for a total of $200$ data points. The final standings from the end of the season provide odd and even placed teams, with the latter being selected for analysis.  </p>
<h3>Web Scraping</h3>
<p align = "justify">
I wrote three Python functions to extract data from fbref tables after careful analysis of URL patterns on fbref as well as the HTML code for the respective webpages. As a first step, the function $teams\textunderscore list$ admits a string <var>season</var> and returns a pair of pandas Data Frame objects, <var>Teams</var> and <var>Opponents</var>, which contain a list of club names and URL links to their consolidated statistics for the La Liga season.

 
 ```python
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
    
Teams,Opponents = teams_list('2022-2023')

```

The function $fixtures\textunderscore table$  reads in data from <var>Teams</var> and <var>Opponents</var> to create a dictionary of Data Frames, one for each team. These DFs contain general data about individual fixtures as well as links to detailed match reports for each fixture.

```python
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

Fixtures={}

for i in tqdm.tqdm(range(len(Teams))):
    print("Collecting fixtures for", Teams.iloc[i,1]+ "...")
    Fixtures[Teams.iloc[i,1]] = fixtures_table(Teams.iloc[i,1], Teams.iloc[i,2],Opponents)

```

Finally, the function $metrics$ uses these links to provide advanced statistics from the match reports.

```python

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
    
for key in Fixtures:
    
    DF = Fixtures[key]
    DF['Team']=key
    dflinks = DF['Match Report']
    DF[['Touches']], DF[['xA']]=dflinks.apply(metrics)
```
Once the revenues of the relevant are manually added in as well, all of the data is consolidated into the Data Frame 'Data' for ease of access and analysis.

```python
rev = {'Almería':45.09,'Atlético Madrid':120.43,'Barcelona':155.1,'Celta Vigo':51.17,'Espanyol':51.24,'Getafe':53.38,'Mallorca':45.04,'Osasuna':49.65,'Rayo Vallecano':45.97,'Villareal':63.35}
data = pd.concat(Fixtures.values(),ignore_index =True)
data['Revenue']=data['Team'].map(rev)
```
<u> Note</u>: The functions above require lxml and html5lib for error-free execution.
 
</p>
<h2>Data Analysis</h2>
<h3>The correlation</h3>
The first order of business will be to establish and quantify the correlation between Touches taken and xG generated per game for the retrieved data. The function LR(X,Y) performs a linear regression on variables X and Y using statsmodel, and plots the resulting fit against the data. 

```python

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
LR(Data,"Touches","xG")

```
 
 | ![xG_vs_Touches.jpg](https://github.com/praveer-kg/Causal-Inference-Football-Toy-Model/blob/main/Files/xG_vs_Touches.png) | 
|:--:| 
| Figure 2: Regression Plot and Summary for Expected Goals vs Touches |


With Touches as the independent variable and xG as the dependent one, a summary of the regression model provides a p-value of $0.00$, conclusively rejecting the null hypothesis. The data's R-squared value of $0.14$ is perhaps a bit more ambiguous, since what is considered a 'good' R-squared depends on the field of study. For perspective, let's consider two pairs of football statistics that one would expect to be tightly correlated:
 <dl>
  <dt>Expected Assists vs Expected Goals </dt>
  <dd>Expected Assists(xA) calculates the probability that a given pass results in a goal. Since the bulk of shots taken during a game come from passes, xA and xG should be tightly correlated.  </dd>
  <dt>Expected Goals vs Goals</dt>
  <dd>On average, goals scored(GF) and xG are expected to be close in value at best, and strongly correlated at worst. </dd>
</dl> 
</p>
<p>The following code performs a linear regression analysis between these pairs of variables.<br>

 | ![xG_vs_xA.jpg](https://github.com/praveer-kg/Causal-Inference-Football-Toy-Model/blob/main/Files/xG_vs_xA.png) | 
|:--:| 
| Figure 3: Regression Plot and Summary for Expected Goals vs Expected Assists |

 | ![GF_vs_xG.jpg](https://github.com/praveer-kg/Causal-Inference-Football-Toy-Model/blob/main/Files/GF_vs_xG.png) | 
|:--:| 
| Figure 4: Regression Plot and Summary for Goals Scored vs Expected Goals |


The p-values once again rule out the null hypothesis while the R-squared values provide a reference for strongly correlated statistics in football. Even for xA vs xG, only $52$% of the variations in the dependent variable are accounted for by changes in the idependent variable, with that figure dropping to $40$% for xG and goals scored, despite expectations of a strong dependence. In this context, the R-squared value of $0.14$ for Touches vs xG represents a correlation of moderate strength. The question now is how much of this remains when controlling for club revenue and match venue.</p>
<h3>The Causal Effect</h3>
<p align = "justify">
I adapt the procedure described in this article to infer the actual causal effect of Touches on xG. With a continuous treatment variable, we no longer estimate the probability of treatment given covariates, but rather a probability density function $g(x)$, obtained by regressing the treatment on the covariates. The treatment for an individual with covariates $x_i$ is then assumed to be follow a normal distribution​ with mean $g(x_i)$ and a variance equal to that of the residuals from the regression model. Explicitly, the conditional treatment density given covariates is modelled as:

```math
f(T_i|x_i) =  \exp{\left(\frac{-(T_i-g(x_i))^2}{2\sigma^2}\right)}
```

The corresponding weight $w_i$ is equal to the inverse of this density at a given $x_i$ and treatment $T_i$. One could further stabilize the weights by multiplying them with the marginal treatment density, obtained by regressing the treatment without covariates(i.e fitting only an intercept). In any case, regressing the outcome against the weighted treatment provides the dose-response function, which estimates counterfactual outcomes. The goal of this article is to obtain the dose-response function for xG vs Touches, controlling for confounders $x_i$. The function provided in the cited article has been generalized below to accept a Data Frame,the labels for treatment and outcome columns as well as a list of labels for the confounders. Note that the regression will be non-linear, assuming a quadratic function of the input variables. The output of the function $response$ is the dose-response function.


```python

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
response(data,"Touches","xG",["Revenue","Venue"])
```
 
 | ![Dose_Response.jpg](https://github.com/praveer-kg/Causal-Inference-Football-Toy-Model/blob/main/Files/Dose-Response.png) | 
|:--:| 
| Figure 5: Dose-Response curve for Expected Goals vs Touches, plotted against the Data and Linear Regression Prediction |

<h2>Discussion</h2>
The Dose-Response curve shows clearly that the strength of the correlation drops when controlling for club revenue and match venue. In fact, more Possession is conducive to creating more xG upto a certain dosage. Increments in Possession beyond this point have detrimental effects, perhaps explained by the fact that agressive strategies lead to more chance creation but also more turnovers(loss of Possession), assuming of course that our DAG is accurate. However, as mentioned earlier, the assumptions made here are an oversimplification of reality. For example, the revenue(R) is more likely to affect other factors(squad composition, manager choice etc.) that could themselves be confounders under certain circumstances. Perhaps using goals scored(GF) would ameliorate the issue since manager influence is greatly reduced in finishing chances rather than creating them. With that, I will conclude this discussion and thank the reader who has made it this far. 
</p>
