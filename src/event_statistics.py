import time
import pandas as pd
import numpy as np

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    '''
    Read the events.csv and mortality_events.csv files. 
    Variables returned from this function are passed as input to the metric functions.
    '''
    events = pd.read_csv(filepath + 'events.csv', parse_dates=['timestamp'])
    mortality = pd.read_csv(filepath + 'mortality_events.csv', parse_dates=['timestamp'])
    #event for pass away patient
    mortalityf = events.loc[events.patient_id.isin(mortality.patient_id), :]
    #event for alive patient
    eventsf = events.loc[~events.patient_id.isin(mortality.patient_id), :]
    return eventsf, mortalityf

def event_count_metrics(events, mortality):
    '''
    Implement to compute the event count metrics.
    Event count is defined as the number of events recorded for a given patient.
    '''
    dead_event_count = mortality.groupby("patient_id").patient_id.count()
    alive_event_count = events.groupby("patient_id").patient_id.count()
    
    avg_dead_event_count = dead_event_count.mean()
    max_dead_event_count = dead_event_count.max()
    min_dead_event_count = dead_event_count.min()
    avg_alive_event_count = alive_event_count.mean()
    max_alive_event_count = alive_event_count.max()
    min_alive_event_count = alive_event_count.min()

    return min_dead_event_count, max_dead_event_count, avg_dead_event_count, min_alive_event_count, max_alive_event_count, avg_alive_event_count

def encounter_count_metrics(events, mortality):
    '''
    Implement to compute the encounter count metrics.
    Encounter count is defined as the count of unique dates on which a given patient visited the ICU. 
    '''
    dead_event_encounter_count = mortality.groupby("patient_id").timestamp.nunique()
    alive_event_encounter_count = events.groupby("patient_id").timestamp.nunique()
    
    avg_dead_encounter_count = dead_event_encounter_count.mean()
    max_dead_encounter_count = dead_event_encounter_count.max()
    min_dead_encounter_count = dead_event_encounter_count.min()
    avg_alive_encounter_count = alive_event_encounter_count.mean()
    max_alive_encounter_count = alive_event_encounter_count.max()
    min_alive_encounter_count = alive_event_encounter_count.min()

    return min_dead_encounter_count, max_dead_encounter_count, avg_dead_encounter_count, min_alive_encounter_count, max_alive_encounter_count, avg_alive_encounter_count

def record_length_metrics(events, mortality):
    '''
    Implement to compute the record length metrics.
    Record length is the duration between the first event and the last event for a given patient. 
    '''
    dead_event_len = mortality.groupby("patient_id").timestamp.agg(np.ptp).astype('timedelta64[D]')
    alive_event_len = events.groupby("patient_id").timestamp.agg(np.ptp).astype('timedelta64[D]')
    
    avg_dead_rec_len = dead_event_len.mean()
    max_dead_rec_len = dead_event_len.max()
    min_dead_rec_len = dead_event_len.min()
    avg_alive_rec_len = alive_event_len.mean()
    max_alive_rec_len = alive_event_len.max()
    min_alive_rec_len = alive_event_len.min()

    return min_dead_rec_len, max_dead_rec_len, avg_dead_rec_len, min_alive_rec_len, max_alive_rec_len, avg_alive_rec_len

def main():
    '''
    DO NOT MODIFY THIS FUNCTION.
    '''
    # You may change the following path variable in coding but switch it back when submission.
    train_path = '../data/train/'

    # DO NOT CHANGE ANYTHING BELOW THIS ----------------------------
    events, mortality = read_csv(train_path)

    #Compute the event count metrics
    start_time = time.time()
    event_count = event_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute event count metrics: " + str(end_time - start_time) + "s"))
    print(event_count)

    #Compute the encounter count metrics
    start_time = time.time()
    encounter_count = encounter_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute encounter count metrics: " + str(end_time - start_time) + "s"))
    print(encounter_count)

    #Compute record length metrics
    start_time = time.time()
    record_length = record_length_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute record length metrics: " + str(end_time - start_time) + "s"))
    print(record_length)
    
if __name__ == "__main__":
    main()
