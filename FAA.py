import numpy as np
import pandas as pd
from datetime import datetime as dt
import pandas_datareader as web
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
import math

tickers = ['VTSMX', 'FDIVX', 'VEIEX', 'VFISX', 'VBMFX', 'DBC', 'VGSIX']

start = dt(2007,1,4)
end = dt(2022,8,30)

data = web.DataReader(tickers, 'yahoo', start, end)['Adj Close']
upro = web.DataReader('UPRO', 'yahoo', start, end)['Adj Close']

tickers = data.columns.copy()
tickers = tickers.to_list()
data_srm = data[tickers]

n = data_srm.shape[1]
ranks = [1, 2, 3]
weights = np.array([1/ranks[-1]]*ranks[-1])
lookback_period = 120
reallocation_period = 30

capital = 1000

delta = data_srm.diff()

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
        
        if x[pfl[1].iloc[i]].iloc[i] < 0:
            pfl['No 1'][i:] = np.floor(weights[0]*pfl['Portfolio'].iloc[i-1]/hedge.iloc[0])
        else:
            pfl['No 1'][i:] = np.floor(weights[0]*pfl['Portfolio'].iloc[i-1]/price[pfl[1].iloc[i]].iloc[i])
    
        if x[pfl[2].iloc[i]].iloc[i] < 0:
            pfl['No 2'][i:] = np.floor(weights[1]*pfl['Portfolio'].iloc[i-1]/hedge.iloc[0])
        else:
            pfl['No 2'][i:] = np.floor(weights[1]*pfl['Portfolio'].iloc[i-1]/price[pfl[2].iloc[i]].iloc[i])
    
        if x[pfl[3].iloc[i]].iloc[i] < 0:
            pfl['No 3'][i:] = np.floor(weights[2]*pfl['Portfolio'].iloc[i-1]/hedge.iloc[0])
        else:
            pfl['No 3'][i:] = np.floor(weights[2]*pfl['Portfolio'].iloc[i-1]/price[pfl[3].iloc[i]].iloc[i])
    
    
        if x[pfl[1].iloc[i]].iloc[i] < 0:
            pfl['Val 1'][i:] = hedge.iloc[i]*pfl['No 1'].iloc[i]
        else:
            pfl['Val 1'][i:] = price[pfl[1].iloc[i]].iloc[i:]*pfl['No 1'].iloc[i]
        
        if x[pfl[2].iloc[i]].iloc[i] < 0:
            pfl['Val 2'][i:] = hedge.iloc[i]*pfl['No 2'].iloc[i]
        else:
            pfl['Val 2'][i:] = price[pfl[2].iloc[i]].iloc[i:]*pfl['No 2'].iloc[i]
    
        if x[pfl[3].iloc[i]].iloc[i] < 0:
            pfl['Val 3'][i:] = hedge.iloc[i]*pfl['No 3'].iloc[i]
        else:
            pfl['Val 3'][i:] = price[pfl[3].iloc[i]].iloc[i:]*pfl['No 3'].iloc[i]
        
        
        pfl['Asset Total'][i:] = pfl['Val 1'][i:]\
                                    + pfl['Val 2'][i:]\
                                    + pfl['Val 3'][i:]\
    
        pfl['Cash'][i:] = pfl['Portfolio'].iloc[i-1] - pfl['Asset Total'][i]
        pfl['Portfolio'][i:] = pfl['Asset Total'][i:] + pfl['Cash'][i:]

ones = pd.DataFrame(np.ones([len(r.index), len(r.columns)]), index = r.index, columns = r.columns)

gen_pfl(ones, rank_r, upro, weights, train, r_pfl)

bh = pd.DataFrame(index = data_srm.iloc[lookback_period:].index)
cap_alloc = capital/len(tickers)

for i in tickers:
    bh[i] = data_srm[i].iloc[lookback_period:]*(np.floor(cap_alloc/data_srm[i].iloc[lookback_period]))

bh['Assets'] = bh.iloc[:, 0:len(tickers)].sum(axis=1)
bh['Cash'] = capital - bh['Assets'].iloc[0]
bh['Portfolio'] = bh['Cash'] + bh['Assets']

spx = web.DataReader('^GSPC', 'yahoo', r_pfl.index.to_list()[0].to_pydatetime(), r_pfl.index.to_list()[-1].to_pydatetime())['Adj Close']
spx = pd.DataFrame(spx*(np.floor(capital/spx.iloc[0])))
spx['Cash'] = capital - spx['Adj Close'].iloc[0]
spx['Portfolio'] = spx['Cash'] + spx['Adj Close']

plt.figure(figsize = (12,8), dpi = 200)
plt.plot((r_pfl['Portfolio']), label = 'Backtest')
plt.plot((bh['Portfolio']), label = 'Benchmark')
plt.legend()
plt.title('Relative Momentum vs Buy & Hold')
plt.savefig('r.png')
plt.show()

m = (data_srm.diff(lookback_period))/data_srm.shift(lookback_period)

m = m.iloc[lookback_period:]

rank_m1 = pd.DataFrame()

for i in range(len(m))[lookback_period::reallocation_period]:
    rank_m1 = pd.concat([rank_m1, m.iloc[i].rank(ascending = False)], axis = 1)

rm = (rank_r1.T + rank_m1.T).T.rank()

dates_rm = rm.columns.copy()
rank_rm = pd.DataFrame(index = ranks)

for i in range(len(dates_r)):
    rank_rm[i] = rm[rm.columns[i]].sort_values().iloc[:ranks[-1]].index.to_list()

rank_rm.columns = dates_r.copy()

capital = 1000000
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

mv_weights1 = np.array([1/2, 1/2])

r = np.log(data_srm).diff()
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

c = pd.DataFrame()

def avg_corr(x):
    c1 = pd.DataFrame()
    mask = tickers.copy()
    mask.remove(mask[x])
    
    for i in range(len(mask)):
        c1[i] = data_srm[tickers[x]].rolling(lookback_period).corr(data_srm[mask[i]])

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

u_weights = np.array([1/2, 1/3, 1/6])

gen_pfl(m, rank_mvc, data['VFISX'], u_weights, train, mvc_pfl)

plt.figure(figsize = (12,8), dpi = 200)
plt.plot((mvc_pfl['Portfolio']), label = 'Backtest')
plt.plot((bh['Portfolio']), label = 'Benchmark')
plt.title('Tactical Asset Allocation with Momentum, Volatility, & Correlation, Unequal Weightings')
plt.legend()
plt.savefig('mvc_ue.png')
plt.show()

print('Done!')