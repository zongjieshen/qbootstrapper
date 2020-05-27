#! /usr/bin/env python
# vim: set fileencoding=utf-8
'''
Copyright (c) Kevin Keogh 2016

Implements the Curve ojects that can be used to hold discount factor
curves and implement the build method for bootstrapping.

Note that there must be at least 1 analytic (cash/fra/futures) instrument
in the curve if there are swaps in the curve, otherwise the splines cannot
build and the curve will fail.
'''
# python libraries
import copy
import datetime
import pandas as pd
import dateutil.relativedelta
import numpy as np
import operator
import scipy.interpolate
import time
import matplotlib.pyplot as plt 
# qlib libraries
import qbootstrapper.instruments as instruments


class Curve(object):
    '''Base Interest Rate Swap Curve class
    The Curve class, holds multiple attributes and methods for use with
    interest rate swap curve construction. The class also allows, after
    construction, discount factors to be drawn for arbitrary dates.

    Arguments:
        effective_date (datetime)   : Effective date of the curve

        kwargs
        ------
        discount_curve (Curve)      : Discount curve for dual curve bootstrap
                                      [default: False]
        allow_extrapolation (bool)  : Boolean for allowing the interpolant
                                      to extrapolation

    Attributes:
        curve (np.array)            : Numpy 3xn array of log discount factors
                                      Takes the form:
                                            Date (string), Date (epoch), log(DF)
        discount_curve (Curve)      : If the discount_curve is specified,
                                      holds the reference to the curve

        instruments (list)          : List of the instruments in the curve
        allow_extrapolation (bool)  : Boolean, reflecting whether the
                                      interpolant can extrapolate
    '''
    def __init__(self, effective_date, discount_curve=False,
                 allow_extrapolation=True, loadB = False):
        if type(effective_date) is not datetime.datetime:
            raise TypeError('Effective date must be of type datetime.datetime')

        if not isinstance(discount_curve, Curve) and discount_curve is not False:
            raise TypeError('Discount curve must of of type Curve')

        if type(allow_extrapolation) is not bool:
            raise TypeError('Allow_extrapolation must be of type \'bool\'')

        self.curve = np.array([(np.datetime64(effective_date.strftime('%Y-%m-%d')),
                                     0,                               #time.mktime(effective_date.timetuple()),
                                np.log(1))],
                              dtype=[('maturity', 'datetime64[D]'),
                                     ('timestamp', np.int),
                                     ('discount_factor', np.float64)])

        self.curve_type = 'IR_curve'
        self.discount_curve = discount_curve
        self.instruments = []
        self.effective_date = effective_date
        self._built = False
        self.allow_extrapolation = allow_extrapolation
        self.loadB = loadB

    def add_instrument(self, instrument):
        '''Add an instrument to the curve
        '''
        if isinstance(instrument, instruments.Instrument):
            self._built = False
            self.instruments.append(instrument)
        else:
            raise TypeError('Instruments must be a of type Instrument')

    def build(self):
        '''Initiate the curve construction procedure
        '''
        self.curve = self.curve[0]
        self.instruments.sort(key=operator.attrgetter('maturity'))
        for instrument in self.instruments:
            discount_factor = instrument.discount_factor()
            temp=self.curve[0]
            offset = ((np.datetime64(instrument.maturity) - np.datetime64(self.effective_date))/np.timedelta64(1, 'D')).astype(np.int32)
            array = np.array([(np.datetime64(instrument.maturity.strftime('%Y-%m-%d')),
                              offset,
                              discount_factor)], dtype=self.curve.dtype)
            self.curve = np.append(self.curve, array)

        self._built = True

    def discount_factor(self, date):
        '''Returns the interpolated discount factor for an arbitrary date
        '''
        if type(date) is not datetime.datetime and type(date) is not np.datetime64:
            raise TypeError('Date must be a datetime.datetime or np.datetime64')
        if type(date) == datetime.datetime:
            offset = (date - self.effective_date).days

        return np.exp(self.log_discount_factor(offset))

    def log_discount_factor(self, offset):
        '''Returns the natural log of the discount factor for an arbitrary date
        '''
        interpolator = scipy.interpolate.interp1d(self.curve['timestamp'],
                                                  self.curve['discount_factor'],kind='linear',fill_value='extrapolate')
        return interpolator(offset)

    def view(self, ret=False):
        '''Prints the discount factor curve
        Optionally return tuple of the maturities and discount factors
        '''
        if not self._built:
            self.build()

        maturities = self.curve['maturity']
        discount_factors = np.exp(self.curve['discount_factor'])
        for i in range(len(self.curve)):
            date = maturities[i].astype(object).strftime('%Y-%m-%d')
            print('{0} {1:.10f}'.format(date, discount_factors[i]))

        if ret:
            return maturities, discount_factors

    def zeros(self, ret=False):
        '''Prints the zero rate curve
        Optionally return tuple of the maturities and discount factors
        '''
        if not self._built:
            self.build()

        maturities = self.curve['maturity']
        zero_rates = np.zeros(len(maturities))
        for i in range(1, len(self.curve)):
            days = ((self.curve[i]['maturity'] - self.curve[0]['maturity']) /
                    np.timedelta64(1, 'D')) / 360
            temp=1/np.exp(self.curve[i]['discount_factor'])
            zero_rates[i] = (1/np.exp(self.curve[i]['discount_factor'])) ** (1/days)-1

        for i in range(len(self.curve)):
            print('{0} {1:.4f}%'.format(maturities[i], zero_rates[i] * 100))

        if ret:
            return maturities, zero_rates


class LIBORCurve(Curve):
    '''Implementation of the Curve class for LIBOR curves.
    Build method is over-written to cause the discount curve to be built
    in the case of a dual bootstrap
    '''
    def __init__(self, *args, **kwargs):
        super(LIBORCurve, self).__init__(*args, **kwargs)
        self.curve_type = 'LIBOR_curve'

    def build(self):
        '''Checks to see if the discount curve has already been built before
        running the base class build method
        '''
        if self.discount_curve and self.discount_curve._built is False:
            self.discount_curve.build()

        super(LIBORCurve, self).build()


class OISCurve(Curve):
    '''Implementation of the Curve class for OIS curves
    '''
    def __init__(self, *args, **kwargs):
        super(OISCurve, self).__init__(*args, **kwargs)
        self.curve_type = 'OIS_curve'


class SimultaneousStrippedCurve(Curve):
    '''Implementation of the Curve class for a curve that can simultaneously
    bootstrap OIS and LIBOR curves using AverageIndexBasisSwap instruments
    '''
    def __init__(self, effective_date, discount_curve, projection_curve,
                 projection_discount_curve=False, allow_extrapolation=True):

        if type(effective_date) is not datetime.datetime:
            raise TypeError('Effective date must be of type datetime.datetime')

        if not isinstance(discount_curve, Curve) and discount_curve is not False:
            raise TypeError('Discount curve must of of type Curve')

        if type(allow_extrapolation) is not bool:
            raise TypeError('Allow_extrapolation must be of type \'bool\'')

        self.curve_type = 'Simultaneous_curve'
        self.discount_curve = copy.deepcopy(discount_curve)
        for inst in self.discount_curve.instruments:
            inst.curve = self.discount_curve

        self.projection_curve = copy.deepcopy(projection_curve)
        for inst in self.projection_curve.instruments:
            inst.curve = self.projection_curve

        self.projection_curve.discount_curve = self.discount_curve

        self.projection_discount_curve = copy.deepcopy(projection_discount_curve)
        self.instruments = []
        self._built = False
        self.effective_date = effective_date
        self.allow_extrapolation = allow_extrapolation

    def add_instrument(self, instrument):
        '''Needs special because the discount_curve and projection curve
        are deep copied when the curve is created
        '''
        if isinstance(instrument, instruments.Instrument):
            self._built = False
            instrument.projection_instrument.curve = self.projection_curve
            self.instruments.append(instrument)
        else:
            raise TypeError('Instruments must be a of type Instrument')

    def build(self):
        '''
        '''
        self.discount_curve.build()
        self.projection_curve.build()
        
        # TODO figure out some way of sorting these things
        # self.instruments.sort(key=operator.attrgetter('maturity'))

        for instrument in self.instruments:
            df = instrument.discount_factor()

            if df.success:
                leg_one_df, leg_two_df = df.x
                d_maturity = instrument.discount_instrument.maturity
                p_maturity = instrument.projection_instrument.maturity
                array = np.array([(np.datetime64(instrument.discount_instrument.maturity.strftime('%Y-%m-%d')),
                                   ((np.datetime64(d_maturity) - np.datetime64(self.effective_date))/np.timedelta64(1, 'D')).astype(np.int32),
                                   leg_one_df)], dtype=self.discount_curve.curve.dtype)
                self.discount_curve.curve = np.append(self.discount_curve.curve, array)

                array = np.array([(np.datetime64(instrument.projection_instrument.maturity.strftime('%Y-%m-%d')),
                                   ((np.datetime64(p_maturity) - np.datetime64(self.effective_date))/np.timedelta64(1, 'D')).astype(np.int32),
                                   leg_two_df)], dtype=self.projection_curve.curve.dtype)
                self.projection_curve.curve = np.append(self.projection_curve.curve, array)
                
        self._built = True

    def view(self):
        '''
        '''
        raise NotImplementedError('Please view the individual curves using the'
                                  ' self.discount_curve and'
                                  ' self.projection_curve syntax')

    def zeros(self):
        '''
        '''
        raise NotImplementedError('Please view the individual curves using the'
                                  ' self.discount_curve and'
                                  ' self.projection_curve syntax')


class AUDInflationCurve(Curve):
    '''Implementation of the Curve class for LIBOR curves.
    Build method is over-written to cause the discount curve to be built
    in the case of a dual bootstrap
    '''
    def __init__(self, *args, **kwargs):
        super(AUDInflationCurve, self).__init__(*args, **kwargs)
        self.curve_type = 'AUDInflation_curve'

    def build(self):
        '''Checks to see if the discount curve has already been built before
        running the base class build method
        '''
        if self.discount_curve is False:
            if self.loadB == True:
                self._load_discount_curve()
            else:
                self.discount_curve.build()

        self.curve = self.curve[0]
        self.instruments.sort(key=operator.attrgetter('maturity'))
        for instrument in self.instruments:
            inflation_rate = instrument.calibr_inflation_rate()
            EOQ_maturity = datetime.date(instrument.maturity.year, int((instrument.maturity.month-1)/3+1)*3, 1) - dateutil.relativedelta.relativedelta(months=instrument.fixing_lag)
            offset = ((np.datetime64(EOQ_maturity) - np.datetime64(self.effective_date))/np.timedelta64(1, 'D')).astype(np.int32)
            array = np.array([(np.datetime64(EOQ_maturity.strftime('%Y-%m-%d')),
                              offset,
                              inflation_rate)], dtype=self.curve.dtype)
            self.curve = np.append(self.curve, array)
            interpolator = scipy.interpolate.interp1d(self.curve['timestamp'],
                                self.curve['discount_factor'],kind='linear',fill_value='extrapolate')
            instrument.cpi_hist = instrument.update_cpi_dict(inflation_rate,EOQ_maturity,self.curve, interpolator, instrument.cpi_hist)

        self._built = True

    def _load_discount_curve(self):
        df=pd.read_excel(r'C:\Users\Zongjie\Documents\GitHub Repo\qbootstrapper\qbootstrapper\discount_curve.xlsx')
        df['Tenor'] = df['Tenor'].dt.date
        test = df['Tenor'][0]
        if df['Tenor'][0] == self.effective_date.date():
            df['Offset'] = (df['Tenor'] - df['Tenor'][0]).dt.days
        else:
            raise Exception()

        self.discount_curve = np.array([(np.datetime64(self.effective_date.strftime('%Y-%m-%d')),
                                     0,                               
                                     1)],
                              dtype=[('maturity', 'datetime64[D]'),
                                     ('timestamp', np.int),
                                     ('discount_factor', np.float64)])

        for index, row in df[1:].iterrows():
            array = np.array([(np.datetime64(row[0].strftime('%Y-%m-%d')),
                              row[2],
                              row[1])], dtype=self.curve.dtype)
            self.discount_curve = np.append(self.discount_curve, array)

        self._built = True

    def view(self, ret=False):
        '''Prints the discount factor curve
        Optionally return tuple of the maturities and discount factors
        '''
        if not self._built:
            self.build()

        maturities = self.curve['maturity']
        inflation_rate = self.curve['discount_factor']
        for i in range(len(self.curve)):
            date = maturities[i].astype(object).strftime('%Y-%m-%d')
            print('{0} {1:.10f}'.format(date, inflation_rate[i]))
        
            
        plt.plot(maturities,inflation_rate) 
        plt.show() 


        if ret:
            return maturities, inflation_rate




