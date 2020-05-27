from qbootstrapper.swapscheduler import Schedule

class Instrument(object):
    '''Base Instrument convenience class
    Class is primarily used for the date adjustment methods that are used
    by the sub-classes.
    '''
    def __init__(self):
        pass

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
            raise Exception('Adjustment period "{adjustment}" '
                            'not recognized'.format(**locals()))

    @staticmethod
    def _timedelta(length_num, length_type):
        '''Static method to return the date +/- some length with a length type

        Arguments:
            length_num (int)    : Length of the period (e.g., if the period
                                  is 6 months, this is 6)
            length_type (str)   : Period type (e.g., if the period is 6 months,
                                  this is months)
                                  available: months,
                                             weeks,
                                             days
        '''
        if length_type == 'months':
            return dateutil.relativedelta.relativedelta(months=length_num)
        elif length_type == 'weeks':
            return dateutil.relativedelta.relativedelta(weeks=length_num)
        elif length_type == 'days':
            return dateutil.relativedelta.relativedelta(days=length_num)
        else:
            raise Exception('Period length "{length_type}" '
                            'not recognized'.format(**locals()))

    @staticmethod
    def daycount(effective, maturity, basis):
        '''Static method to return the accrual length, as a decimal,
        between an effective and a maturity subject to a basis convention

        Arguments:
            effective (datetime)    : First day of the accrual period
            maturity (datetime)     : Last day of the accrual period
            basis (str)             : Basis convention
                                      available: Act360,
                                                 Act365,
                                                 30360,
                                                 30E360

        '''
        if type(effective) == np.datetime64:
            timestamp = effective.astype('<M8[s]').astype(np.uint64)
            effective = datetime.datetime.fromtimestamp(timestamp)
            timestamp = maturity.astype('<M8[s]').astype(np.uint64)
            maturity = datetime.datetime.fromtimestamp(timestamp)
        if basis.lower() == 'act360':
            accrual_period = (maturity - effective).days / 360
        elif basis.lower() == 'act365':
            accrual_period = (maturity - effective).days / 365
        elif basis.lower() == '30360':
            start, end = min(effective.day, 30), min(maturity.day, 30)
            months = (30 * (maturity.month - effective.month) +
                      360 * (maturity.year - effective.year))
            accrual_period = (end - start + months) / 360
        elif basis.lower() == '30e360':
            start, end = max(0, 30 - effective.day), min(30, maturity.day)
            months = 30 * (maturity.month - effective.month - 1)
            years = 360 * (maturity.year - effective.year)
            accrual_period = (years + months + start + end) / 360
        else:
            raise Exception('Accrual basis "{basis}" '
                            'not recognized'.format(**locals()))
        return accrual_period

class BondInstrument(Instrument):
    '''Base class for bond instruments. See OISSwapInstrument and
    LIBORSwapInstrument for more detailed specs.
    '''
    def __init__(self, effective, maturity, rate, curve,prev_CPI_index,prev_CPI,
                 basis='Act360',length=6,
                 period_length='months', 
                 period_adjustment='unadjusted',
                 payment_adjustment='unadjusted',
                 fixing_lag=6, notional=100,
                 rate_period=1, rate_period_length='days', rate_basis='Act360'):

        # assignments
        self.effective = effective
        self.maturity = maturity
        self.rate = rate
        self.curve = curve
        self.prev_CPI_index = prev_CPI_index
        self.prev_CPI = prev_CPI
        self.fixing_lag = fixing_lag
        self.notional = notional

        self.basis = basis
        self.length = length
        self.period_length = period_length
        self.period_adjustment = period_adjustment
        self.payment_adjustment = payment_adjustment

        self.rate_period = rate_period
        self.rate_period_length = rate_period_length
        self.rate_basis = rate_basis

        self._set_schedules()

    def _set_schedules(self):
        '''Sets the fixed and floating schedules of the swap.
        '''
        self.schedule = Schedule(self.effective, self.maturity,
                                           self.length,
                                           period_length=self.period_length,
                                           period_adjustment=self.period_adjustment,
                                           payment_adjustment=self.payment_adjustment)



class InflationAUDBond(BondInstrument):
    
    def __init__(self, *args, **kwargs):
        super(BondInstrument, self).__init__(*args, **kwargs)
        self.instrument_type = 'InflationAUDBond'

    def discount_factor(self):
        '''Returns the discount factor for the swap using Newton's method
        root finder.
        '''
        return scipy.optimize.newton(self._swap_value, 0)

    def _bond_value(self, guess, args=()):
        '''Private method used for root finding discount factor

        The main function for use with the root-finder. This function returns
        the value of a swap given a discount factor. It appends the discount
        factor to the existent array with the date of the instrument, calculates
        each cashflow and PV for each leg, and returns the net value of the pay
        fixed swap.

        Arguments:
            guess (float)   :   guess to be appended to a copy of the attached
                                curve.

        '''
        if not isinstance(guess, (int, float, long, complex)):
            # simultaneous bootstrapping sets the guess[0] as the ois guess
            guess = guess[0]

        temp_curve = self.curve.curve
        temp_curve = np.append(self.curve.curve,
                               np.array([(np.datetime64(self.maturity.strftime('%Y-%m-%d')),
                                          time.mktime(self.maturity.timetuple()),
                                          guess)],
                                        dtype=self.curve.curve.dtype))
        interpolator = scipy.interpolate.PchipInterpolator(temp_curve['timestamp'],
                                                           temp_curve['discount_factor'])

        #for prev_CPI_Index


        for period in self.float_schedule.periods:
            forward_rate = self.__forward_rate(interpolator, period)
            period['cashflow'] = forward_rate * self.notional

        payment_dates = self.float_schedule.periods['payment_date'].astype('<M8[s]')
        discount_factors = np.exp(interpolator(payment_dates.astype(np.uint64)))
        self.float_schedule.periods['PV'] = self.float_schedule.periods['cashflow'] * discount_factors

        float_leg = self.float_schedule.periods['PV'].sum()

        for period in self.fixed_schedule.periods:
            forward_rate = self.rate
            accrual_period = super(OISSwapInstrument,
                                   self).daycount(period['accrual_start'],
                                                  period['accrual_end'],
                                                  self.fixed_basis)
            period['cashflow'] = forward_rate * accrual_period * self.notional

        payment_dates = self.fixed_schedule.periods['payment_date'].astype('<M8[s]')
        discount_factors = np.exp(interpolator(payment_dates.astype(np.uint64)))
        self.fixed_schedule.periods['PV'] = self.fixed_schedule.periods['cashflow'] * discount_factors

        fixed_leg = self.fixed_schedule.periods['PV'].sum()

        return float_leg - fixed_leg

    def __forward_rate(self, interpolator, period):
        '''Private method for calculating the compounded forward rate for an OIS
        swap.

        The compounded forward rate is calculated as the

                                     DF[i]
                                Π [ ------- ] - 1
                                i   DF[i+1]

        Note that it achieves very speedily by calculating each forward
        rate (+ 1) for the entire date array, and then calculating the product
        of the array. Additionally, there are 3 entries for every Friday, as
        each friday should compound 3 times (no new rates on weekends).

        Arguments:
            interpolator (scipy.interpolate):   temporary interpolator object
                                                that includes the current swap
                                                maturity guess discount factor.
            period (np.recarray)            :   1 line of the swapschedule array
                                                must contain the accrual start
                                                and end dates
        '''
        start_date = period['accrual_start'].astype('<M8[s]')
        end_date = period['accrual_end'].astype('<M8[s]')
        one_day = np.timedelta64(1, 'D')
        start_day = start_date.astype(object).weekday()
        rate = 1
        first_dates = np.arange(start_date, end_date, one_day)
        # replace all Saturdays and Sundays with Fridays
        fridays = first_dates[4 - start_day::7]
        first_dates[5 - start_day::7] = fridays[:len(first_dates[5 - start_day::7])]
        first_dates[6 - start_day::7] = fridays[:len(first_dates[6 - start_day::7])]
        second_dates = first_dates + one_day
        initial_dfs = np.exp(interpolator(first_dates))
        end_dfs = np.exp(interpolator(second_dates))
        rates = (initial_dfs / end_dfs)
        rate = rates.prod() - 1
        return rate