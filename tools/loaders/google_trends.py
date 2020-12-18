import pandas as pd
import numpy as np

from datetime import datetime, timedelta
from pytrends.request import TrendReq
from typing import List, Union


def load_google_trends(kw_list: Union[List, str], start_date: str, maxstep=269, overlap=40):
    if isinstance(kw_list, str):
        kw_list = [kw_list]
        
    step = maxstep - overlap + 1
    start_date = pd.to_datetime(start_date).date()
    
    pytrend = TrendReq()
    
    # Run the first time (if we want to start from today, otherwise we need to ask for an end_date as well
    today = datetime.today().date()
    old_date = today

    # Go back in time
    new_date = today - timedelta(days=step)

    # Create new timeframe for which we download data
    timeframe = new_date.strftime('%Y-%m-%d') + ' ' + old_date.strftime('%Y-%m-%d')
    pytrend.build_payload(kw_list=kw_list, timeframe = timeframe)
    interest_over_time_df = pytrend.interest_over_time()

    ## RUN ITERATIONS
    while new_date > start_date:

        ### Save the new date from the previous iteration.
        # Overlap == 1 would mean that we start where we
        # stopped on the iteration before, which gives us
        # indeed overlap == 1.
        old_date = new_date + timedelta(days=overlap-1)

        ### Update the new date to take a step into the past
        # Since the timeframe that we can apply for daily data
        # is limited, we use step = maxstep - overlap instead of
        # maxstep.
        new_date = new_date - timedelta(days=step)

        # If we went past our start_date, use it instead
        if new_date < start_date:
            new_date = start_date

        # New timeframe
        timeframe = new_date.strftime('%Y-%m-%d') + ' ' + old_date.strftime('%Y-%m-%d')
        print('Loading %s ... ' % timeframe, end='')

        # Download data
        pytrend.build_payload(kw_list=kw_list, timeframe=timeframe)
        print('[OK]')
        
        temp_df = pytrend.interest_over_time()
        if temp_df.empty:
            raise ValueError('Google sent back an empty dataframe. Possibly there were no searches at all during the this period! Set start_date to a later date.')

        # Renormalize the dataset and drop last line
        for kw in kw_list:
            beg = new_date
            end = old_date - timedelta(days=1)

            # Since we might encounter zeros, we loop over the
            # overlap until we find a non-zero element
            for t in range(1, overlap + 1):
                #print('t = ',t)
                #print(temp_df[kw].iloc[-t])
                if temp_df[kw].iloc[-t] != 0:
                    scaling = interest_over_time_df[kw].iloc[t-1] / temp_df[kw].iloc[-t]
                    #print('Found non-zero overlap!')
                    break
                elif t == overlap:
                    print('Did not find non-zero overlap, set scaling to zero! Increase Overlap!')
                    scaling = 0

            # Apply scaling
            temp_df.loc[beg:end,kw] = temp_df.loc[beg : end, kw] * scaling
            
        interest_over_time_df = pd.concat([temp_df[:-overlap], interest_over_time_df])
        
    return interest_over_time_df.drop(columns=['isPartial'])