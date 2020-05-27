#! /usr/bin/env python
# vim: set fileencoding=utf-8
'''
Copyright (c) Kevin Keogh

Implements the Schedule object that creates a NumPy rec.array of
accrual, fixing, and payments dates for an interest rate swap.
'''
import calendar
import datetime
import dateutil.relativedelta
import numpy as np


class Schedule:
    '''Swap fixing, accrual, and payment dates

    The Schedule class can be used to generate the details for periods
    for swaps.

    Arguments:
        effective (datetime)              : effective date of the swap
        maturity (datetime)               : maturity date of the swap
        length (int)                      : length of the period that the
                                            accrual lasts

        kwargs
        ------
        second (datetime, optional)       : second accrual date of the swap
        penultimate (datetime, optional)  : penultimate accrual date of the swap
        period_adjustment (str, optional) : date adjustment type for the accrual
                                            dates
                                            available: following,
                                                       modified following,
                                                       preceding
                                                       unadjusted
                                            [default: unadjusted]
        payment_adjustment (str, optional): date adjustment type for the
                                            payment dates
                                            available: following,
                                                       modified following,
                                                       preceding
                                                       unadjusted
                                            [default: unadjusted]
        fixing_lag (int, optional)        : fixing lag for fixing dates
                                            [default: 2]

        period_length (str, optional)     : period type for the length
                                            available: months, weeks, days
                                            [default: months]

    Attributes:
        periods (np.recarray)             : numpy record array of period data
                                            takes the form
                                              [fixing_date, accrual_start,
                                               accrual_end, accrual_period,
                                               payment_date, cashflow, PV]
                                            note that cashflow and PV are
                                            empty arrays

    '''
    def __init__(self, effective, maturity, length,
                 second=False, penultimate=False,
                 period_adjustment='unadjusted',
                 payment_adjustment='unadjusted',
                 fixing_lag=6, period_length='months',
                 issuedate = '',
                 instrument_type = ''):

        # variable assignment
        self.effective = effective
        self.maturity = maturity
        self.length = length
        self.period_delta = self._timedelta(length, period_length)
        self.period_adjustment = period_adjustment
        self.payment_adjustment = payment_adjustment
        self.second = second
        self.penultimate = penultimate
        self.fixing_lag = fixing_lag
        self.period_length = period_length
        self.instrument_type = instrument_type
        self.issuedate = issuedate

        # date generation routine
        self._gen_periods()
        self._create_schedule()

    def _gen_periods(self):
        '''Private method to generate the date series
        '''


        self._period_ends = self._gen_dates(self.effective,
                                                self.maturity,
                                                self.period_delta,
                                                'unadjusted')
        self._adjusted_period_ends = self._gen_dates(self.effective,
                                                     self.maturity,
                                                     self.period_delta,
                                                     self.period_adjustment)
        self._period_starts = [self.effective] + self._adjusted_period_ends[:-1]
        if self.instrument_type == 'InflationAUDBond':
            self._fixing_dates = self._gen_cpi_dates(self.effective, #Using end period only for AUD BEI Curve
                                                        self.maturity,
                                                        self.fixing_lag,
                                                        self.period_delta)
        else:
            self._fixing_dates = self._gen_date_adjustments(self._adjusted_period_ends, #Using end period only for AUD BEI Curve
                                                            -self.fixing_lag,
                                                            self.period_adjustment,
                                                            self.period_length)
        self._payment_dates = self._gen_date_adjustments(self._period_ends,
                                                         0,
                                                         adjustment=self.payment_adjustment)

    def _create_schedule(self):
        '''Private function to merge the lists of periods to a np recarray
        '''
        arrays = self._np_dtarrays(self._fixing_dates, self._period_starts,
                                   self._adjusted_period_ends,
                                   self._payment_dates)
        arrays = (arrays + (np.zeros(len(self._fixing_dates), dtype=np.float64),) +
                  (np.zeros(len(self._fixing_dates), dtype=np.float64),))
        self.periods = np.rec.fromarrays((arrays),
                                         dtype=[('fixing_date', 'datetime64[D]'),
                                                ('accrual_start', 'datetime64[D]'),
                                                ('accrual_end', 'datetime64[D]'),
                                                ('payment_date', 'datetime64[D]'),
                                                ('cashflow', np.float64), 
                                                ('PV', np.float64)])


    def _timedelta(self, delta, period_length):
        '''Private function to convert a number and string (eg -- 3, 'months') to
        a dateutil relativedelta object
        '''
        if period_length == 'months':
            return dateutil.relativedelta.relativedelta(months=delta)
        elif period_length == 'weeks':
            return dateutil.relativedelta.relativedelta(weeks=delta)
        elif period_length == 'days':
            return dateutil.relativedelta.relativedelta(days=delta)
        else:
            raise Exception('Period length "{period_length}" not '
                            'recognized'.format(**locals()))

    def _gen_dates(self, effective, maturity, delta, adjustment):
        '''Private function to backward generate a series of dates starting
        from the maturity to the effective.

        Note that the effective date is not returned.
        '''
        dates = []
        current = maturity
        counter = 0
        while current > effective:
            dates.append(self._date_adjust(current, adjustment))
            counter += 1
            current = maturity - (delta * counter)
        return dates[::-1]

    def _gen_cpi_dates(self, effective, maturity, delta,period_length):
        dates = []
        current = maturity

        counter = 0
        while current > effective:
            lag_fixing_date = datetime.date(current.year, int((current.month-1)/3+1)*3, 1) - dateutil.relativedelta.relativedelta(months=delta)
            #lag_fixing_date = lag_fixing_date.replace(day=1)
            dates.append(lag_fixing_date)
            counter += 1
            current = current - period_length

        return dates[::-1]

    def _date_adjust(self, date, adjustment):
        '''Method to return a date that is adjusted according to the
        adjustment convention method defined

        Arguments:
            date (datetime)     : Date to be adjusted
            adjustment (str)    : Adjustment type
                                  available: unadjusted,
                                             following,
                                             preceding,
                                             modified following
        '''
        if adjustment == 'unadjusted':
            return date
        elif adjustment == 'following':
            if date.weekday() < 5:
                return date
            else:
                return date + self._timedelta(7 - date.weekday(), 'days')
        elif adjustment == 'preceding':
            if date.weekday() < 5:
                return date
            else:
                return date - self._timedelta(max(0, date.weekday() - 5), 'days')
        elif adjustment == 'modified following':
            if date.month == self._date_adjust(date, 'following').month:
                return self._date_adjust(date, 'following')
            else:
                return date - self._timedelta(7 - date.weekday(), 'days')
        else:
            raise Exception('Adjustment period not recognized')

    def _gen_date_adjustments(self, dates, delta, adjustment='unadjusted',length = 'days'):
        '''Private function to take a list of dates and adjust each for a number
        of days. It will also adjust each date for a business day adjustment if
        requested.
        '''
        adjusted_dates = []

        if length == 'months':
            for date1 in dates:
                last_month_of_quarter = int((date1.month-1)/3+1)*3
                EOQ = datetime.datetime(date1.year, last_month_of_quarter, 1)
                adjusted_date = EOQ + dateutil.relativedelta.relativedelta(months=delta)  #+ self._timedelta(delta, length)
                adjusted_date = adjusted_date.replace(day=1) #date(adjusted_date.year, adjusted_date.month, calendar.monthrange(adjusted_date.year, adjusted_date.month)[0])
                #adjusted_date = self._date_adjust(adjusted_date, adjustment)
                adjusted_dates.append(adjusted_date)
        else:
            for date in dates:
                adjusted_date = date + self._timedelta(delta, length)
                adjusted_date = self._date_adjust(adjusted_date, adjustment)
                adjusted_dates.append(adjusted_date)
        return adjusted_dates

    def _np_dtarrays(self, *args):
        '''Converts a series of lists of dates to a tuple of np arrays of
        np.datetimes
        '''
        fmt = '%Y-%m-%d'
        arrays = []
        for arg in args:
            arrays.append(np.asarray([np.datetime64(date.strftime(fmt)) for date in arg]))
        return tuple(arrays)
