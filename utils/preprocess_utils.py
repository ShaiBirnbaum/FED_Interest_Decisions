import pandas as pd

def split_name_role(data, speaker_col, roles=False):

    new_data= data.copy()
    new_data['speaker_role']=""
    new_data['speaker_name']=""

    #defualt roles_list
    if roles==False:
        roles_list= ['Chairman', 'Vice Chairman', 'Governor', 'Chair', 'Vice Chair']
    else: roles_list= roles

    for role in roles_list:
        mask = new_data[speaker_col].str.contains(rf'\b{role}\b', case=False, na=False)
        new_data.loc[mask, 'speaker_role'] = role
        new_data.loc[mask, 'speaker_name'] = new_data[speaker_col].str.replace(role, '').str.strip() 

    return new_data

def map_roles(data):

    replace_dict = {'Chairman': 'Chair', 
                'Governor': 'Governor',
                'Chair': 'Chair',
                'Vice Chairman': 'Vice Chair',
                'Vice Chair': 'Vice Chair'
                }
    
    
    new_data= data
    new_data['speaker_role'] = new_data['speaker_role'].map(replace_dict)
    return new_data

def cat_interest_change(value):
    if value > 0:
        return "increase"
    elif value < 0:
        return "decrease"
    else:
        return "no_change"
    

def cat_num_interest_change(value):
    if value > 0:
        return 2
    elif value < 0:
        return 0
    else:
        return 1
    
#speeches data preprocess function
def preprocess_speeches(df):
    #spliting name and role to 2 different columns and fixing roles.
    #looks like these are the 5 different roles of speakers:
    roles_list= ['Chairman', 'Vice Chairman', 'Governor', 'Chair', 'Vice Chair']

    data= split_name_role(df, 'speaker', roles_list)
    data= map_roles(data)

    #only one date value missing
    data= data[~data['date'].isna()]

    #fixing column data types
    data['date']=pd.to_datetime(data['date'].astype('int').astype('str'))
    data['year']= data['year'].astype(int)

    #some texts contain very few words
    data=data[data['text_len']>20]

    #sort by date
    data=data.sort_values(by='date')

    #reset to new index
    data= data.reset_index(drop=True)
    return data

#fed interests data preprocess function
def preprocess_interests(df):
    #fixing column types
    usa_interest= df.copy()
    usa_interest['Previous']=usa_interest['Previous'].str.replace('%','').astype(float)
    usa_interest['Actual']=usa_interest['Actual'].str.replace('%','').astype(float)
    usa_interest['interest_date'] = pd.to_datetime(usa_interest['Release Date'], format='%b %d, %Y')

    #adding decision columns
    usa_interest['interest_change']=usa_interest['Actual']-usa_interest['Previous']
    usa_interest['decision'] = usa_interest['interest_change'].apply(cat_interest_change)
    usa_interest['decision_num'] = usa_interest['interest_change'].apply(cat_num_interest_change)

    #screening unrelevant data
    usa_interest=usa_interest[usa_interest['interest_date'] > '1996-01-01']
    usa_interest=usa_interest[~usa_interest['Actual'].isna()]
    usa_interest= usa_interest.drop(columns=(['Release Date', 'Previous']))

    return usa_interest



#### Tagging Utils

#returns a list of speeches in the requested time window from interest decision.
def texts_within_window(speech_data, decision_date, speech_date_col='date', 
                        speech_text_col='text', window_days=30):
    
    window_days = pd.DateOffset(days=window_days)
    mask1 = (speech_data[speech_date_col] >= decision_date - window_days)
    mask2 = (speech_data[speech_date_col] <= decision_date)
    out = speech_data[speech_text_col][mask1 & mask2].tolist()
    return out

#for a specific decision (row)- return th list of speeches, the decision type and date.
def process_row(row, speech_data, decision_date_col='interest_date', 
                decision_col='decision', window_days=30):
    
    decision_date = row[decision_date_col]
    decision_type = row[decision_col]
    texts = texts_within_window(speech_data, decision_date, window_days=window_days)
    return decision_type, texts, decision_date

# counts the number of texts relevant before a specific decision.
def count_texts(texts):
    return len(texts) if isinstance(texts, list) else 0

#return a texts+decisions df for all interest decisions
def tag_speeches(speech_data, interest_data, no_texts=False, window_days=30):
    result_dict = {'Decision': [], 'Texts': [], 'Decision_Date': [], 'Texts_Count': []}

    for _, row in interest_data.iterrows():
        decision_type, texts, decision_date = process_row(row, speech_data, window_days=window_days)

        if not texts and not no_texts:
            continue

        result_dict['Decision'].append(decision_type)
        result_dict['Texts'].append(texts)
        result_dict['Decision_Date'].append(decision_date)
        result_dict['Texts_Count'].append(count_texts(texts))

    return pd.DataFrame(result_dict)


# For every decision in interest_data, i need to get a list of speeches from the previous month.
# this is not good because it only merges the closest to decision speech
# tolerance = pd.to_timedelta('30d')

# tagged_df = pd.merge_asof(interest_data.sort_values(by='interest_date'), 
#                           speech_data[['text', 'date']].sort_values(by='date'), 
#                           left_on='interest_date', 
#                           right_on='date', 
#                           direction='backward', 
#                           allow_exact_matches=False, 
#                           tolerance=tolerance)


# Function to map country names
def map_country_names(country):
    country_mapping = {
        'England': 'united kingdom',
        'Australia': 'australia',
        'Japan': 'japan',
        'Canada': 'canada',
        'Sweden': 'sweden',
        'Switzerland': 'switzerland',
        'USA': 'united states',
        'Europe': 'euro area'
    }
    
    #map values needed to be mapped
    return country_mapping.get(country, country)

