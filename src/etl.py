import utils
import numpy as np
import pandas as pd
from sklearn import preprocessing

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    
    '''
    Read the events.csv, mortality_events.csv and event_feature_map.csv files into events, mortality and feature_map.
    
    Return events, mortality and feature_map
    '''

    #Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    events = pd.read_csv(filepath + 'events.csv', parse_dates = ['timestamp'])
    
    #Columns in mortality_event.csv - patient_id,timestamp,label
    mortality = pd.read_csv(filepath + 'mortality_events.csv', parse_dates = ['timestamp'])

    #Columns in event_feature_map.csv - idx,event_id
    feature_map = pd.read_csv(filepath + 'event_feature_map.csv')

    return events, mortality, feature_map


def calculate_index_date(events, mortality, deliverables_path):
    
    '''
    Index date: The day on which mortality is to be predicted. Index date is evaluated as follows:
        For deceased patients: Index date is 30 days prior to the death date (timestamp fi
eld) 
        in data/train/mortality events.csv.
        For alive patients: Index date is the last event date in data/train/events.csv for
        each alive patient.
        
    Inputs are returns from read_csv function
    Return indx_date
    
    IMPORTANT:
    Save indx_date to a csv file in the deliverables folder named as etl_index_dates.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, indx_date.
    The csv file should have a header  
    '''
    
    #Last event date 
    alive_date = pd.DataFrame({'timestamp' : events.groupby('patient_id').timestamp.max()}).reset_index()
    
    #30 days prior to the death date for deceased patients
    deceased_date = mortality[["patient_id","timestamp"]]
    deceased_date.loc[:,"timestamp"] = deceased_date.loc[:,"timestamp"] - utils.timedelta(30)
    
    #Last event date for alive patients
    still_alive_date = alive_date[~alive_date.patient_id.isin(deceased_date.patient_id)]
    
    #Conbine both alive and deceased patients into one dataframe
    indx_date = pd.concat([still_alive_date, deceased_date], axis = 'rows')
    indx_date.columns = ['patient_id','indx_date']
    
    #Save indx_date to a csv file in the deliverables folder named as etl_index_dates.csv
    indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)
    
    return indx_date


def filter_events(events, indx_date, deliverables_path):
    
    '''
    Consider an observation window (2000 days) and prediction window (30 days). Remove
    the events that occur outside the observation window.

    Suggested steps:
    1. Join indx_date with events on patient_id
    2. Filter events occuring in the observation window(IndexDate-2000 to IndexDate)
    
    Inputs: events is return from read_csv function, indx_date are returns from calculate_index_date function
    Return filtered_events 
    
    IMPORTANT:
    Save filtered_events to a csv file in the deliverables folder named as etl_filtered_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header 
    '''
    
    #Join indx_date with events on patient_id
    joined_events = pd.merge(events, indx_date, on=['patient_id'], how = 'left')
    
    #Filter events occuring in the observation window(IndexDate-2000 to IndexDate)
    mask = np.logical_and(joined_events.timestamp <= joined_events.indx_date, 
                     joined_events.timestamp >= joined_events.indx_date - utils.timedelta(2000))
    filtered_events = joined_events.loc[mask, :]
    
    #Save filtered_events to a csv file in the deliverables folder named as etl_filtered_events.csv
    filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)
    
    return filtered_events


def aggregate_events(filtered_events, mortality, feature_map, deliverables_path):
    
    '''
    To create features suitable for machine learning, we will need to aggregate the events for each patient as follows:
        Sum values for diagnostics and medication events (i.e. event id starting with DIAG and DRUG).
        Count occurences for lab events (i.e. event id starting with LAB).
    Each event type will become a feature and we will directly use event id as feature name.
    
    Inputs: mortality and feature_map are returns from read_csv function, 
    filtered_events is return from filter_events function
    Return aggregated_events
    
    IMPORTANT:
    Save aggregated_events to a csv file in the deliverables folder named as etl_aggregated_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header .
    '''
    #Remove events with n/a values
    events_na = filtered_events.dropna(subset=['value'])
    
    #Join feature_map data set to get index number for event
    join_events = pd.merge(events_na, feature_map, on=['event_id'])
    
    #Split into sum and count event as specific above
    count_events = join_events[join_events.idx <= 2680]
    sum_events = join_events[join_events.idx > 2680]
    
    #Calculate feature value
    aggregated_count = count_events.groupby(["patient_id","idx"]).value.count().reset_index()
    aggregated_sum = sum_events.groupby(["patient_id","idx"]).value.sum().reset_index()
    
    #min-max normalization
    min_max_scaler = preprocessing.MinMaxScaler()
    aggregated_count['value'] = min_max_scaler.fit_transform(aggregated_count['value'].values.astype(float).reshape(-1, 1))
    aggregated_sum['value'] = min_max_scaler.fit_transform(aggregated_sum['value'].values.astype(float).reshape(-1, 1))
    
    #Combines the features into 1
    aggregated_events = pd.concat([aggregated_count, aggregated_sum]).reset_index()
    
    #rename columns
    aggregated_events = aggregated_events.rename(columns = {'idx':'feature_id','value':'feature_value'})
    
    #Save aggregated_events to a csv file in the deliverables folder named as etl_aggregated_events.csv. 
    aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', 
                             columns=['patient_id', 'feature_id', 'feature_value'], index = False)

    return aggregated_events

def create_features(events, mortality, feature_map):
    
    deliverables_path = '../deliverables/'

    #Calculate index date
    indx_date = calculate_index_date(events, mortality, deliverables_path)

    #Filter events in the observation window
    filtered_events = filter_events(events, indx_date,  deliverables_path)
    
    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(filtered_events, mortality, feature_map, deliverables_path)

    '''
    Create two dictionaries as below:
    1. patient_features :  Key - patient_id and value is array of tuples(feature_id, feature_value)
    2. mortality : Key - patient_id and value is mortality label
    '''

    patient_features =aggregated_events.groupby('patient_id').apply(lambda df: list(zip(df.feature_id.tolist(), df.feature_value.tolist()))).to_dict()
    mortality = mortality.drop('timestamp',axis = 1)
    mortality = pd.Series(mortality.label.values, index = mortality.patient_id).to_dict()
        
    return patient_features, mortality

def save_svmlight(patient_features, mortality, op_file, op_deliverable):
    
    '''
    Save in SVMLight format

    Create two files:
    1. op_file - which saves the features in svmlight format. (See instructions in Q3d for detailed explanation)
    2. op_deliverable - which saves the features in following format:
       patient_id1 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...
       patient_id2 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...  
    
    Note: Please make sure the features are ordered in ascending order, and patients are stored in ascending order as well.

    Inputs: patient_features and mortality are returns form create_features function
    '''
    deliverable1 = open(op_file, 'wb')
    deliverable2 = open(op_deliverable, 'wb')
    
    for key in sorted(patient_features):
        if key in mortality:
            line1 = "%d " %(1)
            line2 = "%d %d " %(key,1)
        else:
            line1 = "%d " %(0)
            line2 = "%d %d " %(key,0)
        pairs = ' '.join('%i:%.06f'%(i[0], i[1]) for i in sorted(patient_features[key]))
        line1 = line1 + pairs+" "
        line2 = line2 + pairs+" "
        deliverable1.write(bytes(line1+"\n",'UTF-8'))
        deliverable2.write(bytes(line2+"\n",'UTF-8'))
    deliverable1.close()
    deliverable2.close()

            
def main():
    train_path = '../data/train/'
    events, mortality, feature_map = read_csv(train_path)
    patient_features, mortality = create_features(events, mortality, feature_map)
    save_svmlight(patient_features, mortality, '../deliverables/features_svmlight.train', '../deliverables/features.train')

if __name__ == "__main__":
    main()