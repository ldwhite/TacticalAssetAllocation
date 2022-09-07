import numpy as np
import pandas as pd
from datetime import datetime as dt
import pandas_datareader as web
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
import math

# First step after importing our required packages is setting up our data. We use the same tickers as those used in the cited paper, with the exception of 'DBC'—Deutsche Bank Commodity ETF—as the QRAAX cited in the original paper has since been liquidated.

tickers = ['VTSMX', 'FDIVX', 'VEIEX', 'VFISX', 'VBMFX', 'DBC', 'VGSIX']

# Next we set up our start end end times, chosen as to be a similar time window as the original paper.

start = dt(2019,1,4)
end = dt(2022,8,30)

# Now we use pandas datareader to construct a DataFrame using the Adjusted Close price of each of the securities in our list of tickers.

data = web.DataReader(tickers, 'yahoo', start, end)['Adj Close']

# The idea behind this setup is to make it as modular as possible, such that in the future one need only change the tickers and run the code.

n = data.shape[1]
ranks = [1, 2, 3]
weights = np.array([1/ranks[-1]]*ranks[-1])
lookback_period = 30
reallocation_period = 30
capital = 1000

# Relative Strength

## Relative Strength, referred to as relative momentum in Keller & van Putten (2012), is calculated below.

delta = data.diff()

up, down = delta.clip(lower=0), delta.clip(upper=0).abs()
roll_up, roll_down = up.rolling(window = reallocation_period, center = False).mean(), down.rolling(window = reallocation_period, center = False).mean()
r = (roll_up/roll_down)

rank_r1 = pd.DataFrame()

for i in range(len(r))[lookback_period::reallocation_period]:
    rank_r1 = pd.concat([rank_r1, r.iloc[i].rank(ascending = False)], axis = 1)
    
dates_r = rank_r1.columns.copy()
r = r.iloc[lookback_period:]

rank_r = pd.DataFrame(index = ranks)

for i in range(len(dates_r)):
    rank_r[i] = rank_r1[rank_r1.columns[i]].sort_values().iloc[:ranks[-1]].index.to_list()

rank_r.columns = dates_r.copy()

train = data.iloc[data.index.to_list().index(rank_r.T.index.to_list()[0]):] 

r_pfl = pd.DataFrame(index = train.index.to_list())
r_pfl.index.name = 'Date'

for i in ranks:
    r_pfl[i] = rank_r.T[i]

"""    
In order to make our lives easier going forward, we define a function gen_pfl to generate our portfolio. The variable 'x' is the variable on which the threshold is calculated, such that when that varaible is at or below 0, the portfolio goes to cash.
The variable 'rank_x' refers to the DataFrame which has ranked the securities at any particular time t.
'Hedge' refers to the asset that our portfolio will be replaced with once it crosses the threshold (which in this case is 'VFISX' as a cash proxy fund)
'Price' refers to the DataFrame from which securities' prices will be determined
'Pfl' refers to the portfolio DataFrame that this function is generatin.
"""

def gen_pfl(x, rank_x, hedge, weights, price, pfl):
    for i in ranks:
        pfl[i] = rank_x.T[i]
    
    pfl.ffill(inplace=True)

    pfl['No 1'] = np.floor(weights[0]*capital/price[pfl[1].iloc[0]].iloc[0])
    pfl['No 2'] = np.floor(weights[0]*capital/price[pfl[2].iloc[0]].iloc[0])
    pfl['No 3'] = np.floor(weights[0]*capital/price[pfl[3].iloc[0]].iloc[0])

    pfl['Val 1'] = price[pfl[1].iloc[0]]*pfl['No 1'].iloc[0]
    pfl['Val 2'] = price[pfl[2].iloc[0]]*pfl['No 2'].iloc[0]
    pfl['Val 3'] = price[pfl[3].iloc[0]]*pfl['No 3'].iloc[0]


    pfl['Asset Total'] = pfl['Val 1']\
                             + pfl['Val 2']\
                             + pfl['Val 3']\


    pfl['Cash'] = capital - pfl['Asset Total'].iloc[0]
    pfl['Portfolio'] = pfl['Cash'] + pfl['Asset Total']

    for i in range(len(pfl))[reallocation_period::reallocation_period]:
        
        if x[pfl[1].iloc[i]].iloc[i] <= 0:
            pfl['No 1'][i:] = np.floor(weights[0]*pfl['Portfolio'].iloc[i-1]/hedge.iloc[0])
        else:
            pfl['No 1'][i:] = np.floor(weights[0]*pfl['Portfolio'].iloc[i-1]/price[pfl[1].iloc[i]].iloc[i])
    
        if x[pfl[2].iloc[i]].iloc[i] <= 0:
            pfl['No 2'][i:] = np.floor(weights[1]*pfl['Portfolio'].iloc[i-1]/hedge.iloc[0])
        else:
            pfl['No 2'][i:] = np.floor(weights[1]*pfl['Portfolio'].iloc[i-1]/price[pfl[2].iloc[i]].iloc[i])
    
        if x[pfl[3].iloc[i]].iloc[i] <= 0:
            pfl['No 3'][i:] = np.floor(weights[2]*pfl['Portfolio'].iloc[i-1]/hedge.iloc[0])
        else:
            pfl['No 3'][i:] = np.floor(weights[2]*pfl['Portfolio'].iloc[i-1]/price[pfl[3].iloc[i]].iloc[i])
    
    
        if x[pfl[1].iloc[i]].iloc[i] <= 0:
            pfl['Val 1'][i:] = hedge.iloc[i]*pfl['No 1'].iloc[i]
        else:
            pfl['Val 1'][i:] = price[pfl[1].iloc[i]].iloc[i:]*pfl['No 1'].iloc[i]
        
        if x[pfl[2].iloc[i]].iloc[i] <= 0:
            pfl['Val 2'][i:] = hedge.iloc[i]*pfl['No 2'].iloc[i]
        else:
            pfl['Val 2'][i:] = price[pfl[2].iloc[i]].iloc[i:]*pfl['No 2'].iloc[i]
    
        if x[pfl[3].iloc[i]].iloc[i] <= 0:
            pfl['Val 3'][i:] = hedge.iloc[i]*pfl['No 3'].iloc[i]
        else:
            pfl['Val 3'][i:] = price[pfl[3].iloc[i]].iloc[i:]*pfl['No 3'].iloc[i]
        
        
        pfl['Asset Total'][i:] = pfl['Val 1'][i:]\
                                    + pfl['Val 2'][i:]\
                                    + pfl['Val 3'][i:]\
    
        pfl['Cash'][i:] = pfl['Portfolio'].iloc[i-1] - pfl['Asset Total'][i]
        pfl['Portfolio'][i:] = pfl['Asset Total'][i:] + pfl['Cash'][i:]
        
## Since we don't have a DataFrame to fill the 'x' slot in the gen_pfl function, we merely construct a DataFrame called 'ones' of the appropriate dimensions.
        
ones = pd.DataFrame(np.ones([len(r.index), len(r.columns)]), index = r.index, columns = r.columns)

gen_pfl(ones, rank_r, data['VFISX'], weights, train, r_pfl)

# Now we cosntruct a benchmark portfolio, which is a buy and hold strategy using the securities from above

bh = pd.DataFrame(index = data.iloc[lookback_period:].index)
cap_alloc = capital/len(tickers)

for i in tickers:
    bh[i] = data[i].iloc[lookback_period:]*(np.floor(cap_alloc/data[i].iloc[lookback_period]))

bh['Assets'] = bh.iloc[:, 0:len(tickers)].sum(axis=1)
bh['Cash'] = capital - bh['Assets'].iloc[0]
bh['Portfolio'] = bh['Cash'] + bh['Assets']

# Now we print and save a graph of our portfolio, compared to the benchmark.

plt.figure(figsize = (12,8), dpi = 200)
plt.plot((r_pfl['Portfolio']), label = 'Backtest')
plt.plot((bh['Portfolio']), label = 'Benchmark')
plt.legend()
plt.title('Relative Momentum vs Buy & Hold')
plt.savefig('r.png')
plt.show()

# Absolute Momentum

## Absolute momentum is merely the rate of change of the security over the lookback period, in this case, 30 days

m = (data.diff(lookback_period))/data.shift(lookback_period)

m = m.iloc[lookback_period:]

# We then construct the rank DataFrame using the same technique as with Relative Strength.

rank_m1 = pd.DataFrame()

for i in range(len(m))[lookback_period::reallocation_period]:
    rank_m1 = pd.concat([rank_m1, m.iloc[i].rank(ascending = False)], axis = 1)

rm = (rank_r1.T + rank_m1.T).T.rank()

dates_rm = rm.columns.copy()
rank_rm = pd.DataFrame(index = ranks)

for i in range(len(dates_r)):
    rank_rm[i] = rm[rm.columns[i]].sort_values().iloc[:ranks[-1]].index.to_list()

rank_rm.columns = dates_r.copy()

rm_pfl = pd.DataFrame(index = train.index.to_list())
rm_pfl.index.name = 'Date'

for i in ranks:
    rm_pfl[i] = rank_rm.T[i]
    
gen_pfl(m, rank_rm, data['VFISX'], weights, train, rm_pfl)

plt.figure(figsize = (12,8), dpi = 200)
plt.plot((rm_pfl['Portfolio']), label = 'Backtest')
plt.plot((bh['Portfolio']), label = 'Benchmark')
plt.legend()
plt.title('Relative & Absolute Momentum vs Buy & Hold')
plt.savefig('rm.png')
plt.show()

# Momentum & Volatility

## Now we use both momentum and volatility

mv_weights1 = np.array([1/2, 1/2])

r = np.log(data).diff()
v = np.sqrt(np.square(r).rolling(lookback_period).sum())

rank_v = pd.DataFrame()

for i in range(len(v))[lookback_period::reallocation_period]:
    rank_v = pd.concat([rank_v, v.iloc[i].rank(ascending = True)], axis = 1)
    
dates_v = rank_v.columns

mv = (mv_weights1[0]*(rank_r1.T + rank_m1.T) + mv_weights1[1]*(rank_v.T)).T.rank()

dates_mv = mv.columns.copy()
rank_mv = pd.DataFrame(index = ranks)

for i in range(len(dates_mv)):
    rank_mv[i] = mv[mv.columns[i]].sort_values().iloc[:ranks[-1]].index.to_list()

rank_mv.columns = dates_mv

mv_pfl = pd.DataFrame(index = train.index.to_list())
mv_pfl.index.name = 'Date'

gen_pfl(m, rank_mv, data['VFISX'], weights, train, mv_pfl)

plt.figure(figsize = (12,8), dpi = 200)
plt.plot((mv_pfl['Portfolio']), label = 'Backtest')
plt.plot((bh['Portfolio']), label = 'Benchmark')
plt.legend()
plt.title('Momentum & Volatility vs Buy & Hold')
plt.savefig('mv.png')
plt.show()

# Momentum, Volatility, and Correlation

"""
The construction of the rolling average correlation DataFrame is a bit more involved than the others,
this is because we have to compute the rolling correlation between each security with every other security in the basket
and then compute the average of those correlations. To do this, we use a helper function avg_corr(x) that computes
the average correlation of security 'x' with every other security in the basket. Then, we iterate over the list 'tickers'
in order to create a new column for each security in the list
"""

c = pd.DataFrame()

def avg_corr(x):
    c1 = pd.DataFrame()
    mask = tickers.copy()
    mask.remove(mask[x])
    
    for i in range(len(mask)):
        c1[i] = data[tickers[x]].rolling(lookback_period).corr(data[mask[i]])

    c1.columns = mask

    c1['mean'] = np.nan

    for i in range(len(c1)):
        c1['mean'].iloc[i] = c1.iloc[i].mean()

    c[x] = c1['mean']
    
for i in range(len(tickers)):
    avg_corr(i)
    
c.columns = tickers

rank_c = pd.DataFrame()

for i in range(len(c))[lookback_period::reallocation_period]:
    rank_c = pd.concat([rank_c, c.iloc[i].rank(ascending = True)], axis = 1)
    
dates_c = rank_c.columns.copy()

mvc_weights1 = np.array([1, 1/2, 1/2])
mvc = (mvc_weights1[0]*(rank_r1.T + rank_m1.T) + mvc_weights1[1]*(rank_v.T) + mvc_weights1[2]*(rank_c.T)).T.rank()

dates_mvc = mvc.columns.copy()
rank_mvc = pd.DataFrame(index = ranks)

for i in range(len(dates_mvc)):
    rank_mvc[i] = mvc[mvc.columns[i]].sort_values().iloc[:ranks[-1]].index.to_list()

rank_mvc.columns = dates_mvc

mvc_pfl = pd.DataFrame(index = train.index.to_list())
mvc_pfl.index.name = 'Date'

gen_pfl(m, rank_mvc, data['VFISX'], weights, train, mvc_pfl)

plt.figure(figsize = (12,8), dpi = 200)
plt.plot((mvc_pfl['Portfolio']), label = 'Backtest')
plt.plot((bh['Portfolio']), label = 'Buy & Hold')
plt.title('Tactical Asset Allocation with Momentum, Volatility, & Correlation')
plt.legend()
plt.savefig('mvc.png')
plt.show()

# Unequal Weights

u_weights = np.array([1/2, 1/3, 1/6])

gen_pfl(m, rank_mvc, data['VFISX'], u_weights, train, mvc_pfl)

plt.figure(figsize = (12,8), dpi = 200)
plt.plot((mvc_pfl['Portfolio']), label = 'Backtest')
plt.plot((bh['Portfolio']), label = 'Benchmark')
plt.title('Tactical Asset Allocation with Momentum, Volatility, & Correlation, Unequal Weightings')
plt.legend()
plt.savefig('mvc_ue.png')
plt.show()

# Mutual Information

"""
Mutual Information (MI) is unfortunately not as easy to work with in Python as correlation is, likely due to correlation's significantly greater popularity.
Thus, before following the same technique of constructing a helper function for MI as with correlation, we create a 
function to calculate MI itself.
"""

def calc_MI(x, y):

    n = len(x)
    bins = math.floor(math.sqrt(n/5))
    
    c_xy = np.histogram2d(x, y, bins)[0]
    mis = mutual_info_score(None, None, contingency=c_xy)
    return mis

mi = pd.DataFrame()

def avg_mi(x):
    mi1 = pd.DataFrame(index = data.index)
    mask = tickers.copy()
    mask.remove(mask[x])

    for c in range(len(mask)):
        mi1[c] = np.nan

        for i in range(len(data))[30:]:
            mi1[c].iloc[i] = calc_MI(data[tickers[x]].iloc[i-30:i], data[mask[c]].iloc[i-30:i])

    mi1.columns = mask

    mi1['Mean'] = np.NaN

    for i in range(len(mi1)):
        mi1['Mean'].iloc[i] = mi1.iloc[i].mean()

    mi[x] = mi1['Mean']
    
for i in range(len(tickers)):
    avg_mi(i)

mi.columns = tickers

rank_mi = pd.DataFrame()

for i in range(len(mi))[lookback_period::reallocation_period]:
    rank_mi = pd.concat([rank_mi, mi.iloc[i].rank(ascending = True)], axis = 1)
    
dates_mi = rank_mi.columns.copy()

mvmi_weights = np.array([1, 1, 1])
mvmi = (mvmi_weights[0]*(rank_r1.T + rank_m1.T) + mvmi_weights[1]*(rank_v.T) + mvmi_weights[2]*(rank_mi.T)).T.rank()

dates_mvmi = mvmi.columns.copy()
rank_mvmi = pd.DataFrame(index = ranks)

for i in range(len(dates_mvmi)):
    rank_mvmi[i] = mvmi[mvmi.columns[i]].sort_values().iloc[:ranks[-1]].index.to_list()

rank_mvmi.columns = dates_mvmi

mvmi_pfl = pd.DataFrame(index = train.index.to_list())
mvmi_pfl.index.name = 'Date'

gen_pfl(m, rank_mvmi, data['VFISX'], weights, train, mvmi_pfl)

plt.figure(figsize = (12,8), dpi = 200)
plt.plot(np.log(mvmi_pfl['Portfolio']), label = 'Backtest')
plt.plot(np.log(bh['Portfolio']), label = 'Benchmark')
plt.title('Tactical Asset Allocation with Momentum, Volatility, & Mutual Information')
plt.legend()
plt.savefig('mvmi.png')
plt.show()
