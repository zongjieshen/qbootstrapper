import datetime
from collections import OrderedDict
import copy
import numpy as np
import pandas as pd
import qbootstrapper as qb
import qbootstrapper.swapscheduler as scd
df=pd.read_excel(r'C:\Users\Zongjie\Documents\GitHub Repo\qbootstrapper\qbootstrapper\CPI_Raw.xlsx')
df['Quarter'] = df['Quarter'].dt.date
#df['Quarter'].values.astype('M8[ms]').astype('O')

df['p'] = round(50 * (df['CPI2012'].div(df['CPI2012'].shift(2))-1),2).where(df['Quarter'] >= datetime.date(2012, 9, 1), round(50 * (df['CPI'].div(df['CPI'].shift(2))-1),2))
CPI_hist = {x: {'CPI':y,'CPI2012':z,'p':k} for x, y, z, k in zip(df['Quarter'],df['CPI'],df['CPI2012'],df['p'])}
CPI_hist_ordered = OrderedDict((k,CPI_hist.get(k)) for k in df['Quarter'])

curve_effective = datetime.datetime(2015, 8, 31)
effective = datetime.datetime(2015, 8, 31)
settle_date = datetime.datetime(2015, 9, 3)
AUDInflationCurve = qb.AUDInflationCurve(curve_effective,loadB = True)

AUDBond_conventions = {'length': 3,
                        'basis': 'Act360',
                        'period_adjustment': 'following',
                        'payment_adjustment': 'following',
                        'ex_div_date': 7,
                        'rate_period': 3,
                        'fixing_lag': 6,
                        'rate_period_length': 'months',
                        'cpi_hist': CPI_hist_ordered}

AUDBond_instruments = [(datetime.datetime(2014,  2, 21),
                       datetime.datetime(2018, 11,  21),  0.01, 0.00015),
                       (datetime.datetime(1996,  8, 20),
                       datetime.datetime(2020, 8,  20),  0.04, 0.00121),
                       (datetime.datetime(2012,  2, 21),
                       datetime.datetime(2022, 2,  21),  0.0125, 0.0027),
                       (datetime.datetime(2009,  9, 20),
                       datetime.datetime(2025, 9,  20),  0.03, 0.00471),
                       (datetime.datetime(2010,  9, 20),
                       datetime.datetime(2030, 9,  20),  0.025, 0.00744),
                       (datetime.datetime(2013,  8, 20),
                       datetime.datetime(2035, 8,  21),  0.02, 0.00927)
                       ]

#self, effective, maturity, rate, curve,
for (issuedate, maturity, rate, realyield) in AUDBond_instruments:
    inst = qb.InflationAUDBond(effective,
                               settle_date,
                                issuedate,
                                maturity,
                                rate,
                                realyield,
                                AUDInflationCurve,
                                **AUDBond_conventions)
    AUDInflationCurve.add_instrument(inst)

AUDInflationCurve.build()
AUDInflationCurve.view()


