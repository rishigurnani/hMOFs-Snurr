import pandas as pd
import re
import sys

args = sys.argv

feature_path = args[1]
start_str = args[2] #a string contained ONLY in the column immediately prior to the first feature column
end_str = args[3] #a string contained ONLY in the column immediately after the last feature column
normed = args[4] # 'Yes' if the PG fingerprint is ALREADY normed, 'No' if the PG fingerprint is not ALREADY normed 

y_data = pd.read_excel('/home/rgur/efrc/data_DONOTTOUCH/hMOF_allData_March25_2013.xlsx')

def flatten(A):
    rt = []
    for i in A:
        if isinstance(i,list): rt.extend(flatten(i))
        else: rt.append(i)
    return rt





def whichNan(df, non_features):
    '''
    This function checks to see which columns of df contain a Nan value
    '''
    allowed_cols = ['CH4 v/v 5.8 bar',
 '5.8 bar err.',
 'CH4 v/v 65 bar',
 '65 bar err.',
 'CH4 v/v 248 bar',
 '248 bar err.'] + non_features
    
    return [col for col in df.columns[df.isnull().any()].tolist() if col not in allowed_cols]





def ensureSimilarity(merge_df, df):
    '''
    This function ensures similarity between the merged data frame and the fingerprint data frame. 
    '''
    for col in whichNan(merge_df):
        merge_df[col] = df[col]





def geometricColNames():
    '''
    This function returns the names of all geometric features as a list
    '''
    
    return ['Dom._Pore_(ang.)', 'Max._Pore_(ang.)',
       'Void_Fraction', 'Surf._Area_(m2/g)', 'Vol._Surf._Area', 'Density']





def getPgNewCols(df):
    '''
    This function returns the new names of the PG relevant fingerprints
    '''
    tmp = df.drop(columns=non_features, index=1)
    non_norm_pg_cols = [name for name in tmp.columns if name.startswith('M')][:-1]
    #print(non_norm_pg_cols)
    if normed == "No":
        stat_cols=False
        norm_cols_batch(df, cols_to_norm=non_norm_pg_cols, stat_cols=stat_cols)
    
    return non_norm_pg_cols





def mColNames():
    '''
    This function returns the names of the features related to the metal ions as a list
    '''
    
    return ['valence_pa', 'atomic_rad_pa_(angstroms)', 'affinity_pa_(eV)', 'ionization_potential_pa_(eV)', 'electronegativity_pa']





def replacenth(string, sub, wanted, n=1):
    '''
    This function returns the string which has nth substring replaced
    '''
    where = [m.start() for m in re.finditer(sub, string)][n-1]
    before = string[:where]
    after = string[where:]
    after = after.replace(sub, wanted, 1)
    newString = before + after
    return newString





def uptakeColNames(df):
    '''
    This function returns the names of all uptake columns as a list
    '''
    
    uptake_cols = [col for col in df.columns if col.startswith('CH4 v')]
    uptake_cols = [replacenth(col, 'v', '', 2) for col in uptake_cols]
    uptake_cols = [replacenth(col, '/', '') for col in uptake_cols]
    uptake_cols = [col.replace(' ', '_') for col in uptake_cols]
    return [replacenth(col, '_', '', 3) for col in uptake_cols]

def changeColNames2(df):
    new_cols = []
    if normed == 'No':
        pg_old_cols = getPgNewCols(df)
        pg_new_cols = ['norm_' + col for col in pg_old_cols]
        df = df.drop(pg_old_cols, axis=1)    
    for col in df.columns:
        new_cols.append(col.replace(' ', '_'))
    
    df.columns = new_cols
    return df

def changeColNames(df, non_features, level=['metal', 'uptake']):
    '''
    This function changes the names of the columns of the data in df
    '''
    
    pg_new_cols = getPgNewCols(df, non_features)
    if normed == "No": #delete un-normed cols if need be
        old_pg_cols = [col[5:] for col in pg_new_cols]
        df = df.drop(old_pg_cols, axis=1)
    geo_cols = geometricColNames()
    uptake_cols = uptakeColNames(df)
    
    if ('metal' in level):
        m_cols = mColNames()
        new_cols_all = non_features + pg_new_cols + m_cols + geo_cols + uptake_cols 
    else:
        new_cols_all = non_features + pg_new_cols +  geo_cols + uptake_cols
    
    df.columns = new_cols_all
    return df




def norm_col(df, col_name, stat_cols=True):
    
    nrow = len(df)
    mean = df[col_name].mean()
    std = df[col_name].std()
    
    df['norm_' + col_name] = (df[col_name] - mean) / std
    if stat_cols:
        df['mean_' + col_name] = [mean for i in range(nrow)]
        df['std_' + col_name] = [std for i in range(nrow)]





def norm_cols_batch(df, cols_to_norm=[],level=["metal", "uptake"], stat_cols=True):
    '''
    This function normalizes all the columns specified by level
    '''
    if cols_to_norm == []:
        cols_to_norm += geometricColNames()
        if "metal" in level:
            cols_to_norm += mColNames()
        if "uptake" in level:
            cols_to_norm += [col for col in df.columns if col.startswith('CH4')]
 
    for col in cols_to_norm:
        norm_col(df, col, stat_cols)





def getNumMetal(df):
    '''
    This function returns either [] (if number of distinct metal id = 0), else 'metal' 
    '''
    if len(set(df['Metal ID'])) > 1:
        return ['metal']
    else:
        return []





def getNumUptake(df):
    '''
    This function returns either [] (if number of uptake target cols = 0), else 'uptake' 
    '''
    if len(list(filter(lambda x: 'CH4 v/v' in x, list(y_data.columns)))) > 1:
        return ['uptake']
    else:
        return []





def getLevel(df):
    '''
    This function returns the level
    Include 'metal' in level if your dataset contains multiple distinct linkers
    Include 'uptake' in level if your dataset contains multiple uptake values
    '''
    level = []
    level += getNumMetal(df)
    level += getNumUptake(df)
    return level





def makeCrystalID(df):
    df['Crystal_Id'] = df['filename'].apply(lambda x: int(x.split('_')[1]))





def getNullRows(df):
    '''
    This function returns the entire row of a df containing a null value 
    '''
    return df[df.isnull().any(axis=1)]





def main(df, force_level=False):
    '''
    This function takes merged data, df, and prepares it for ML.
    Include 'metal' in level if your dataset contains multiple distinct metals
    Include 'uptake' in level if your dataset contains multiple uptake values
    Add levels to force_level if you want to specify the level manually
    '''
    if force_level:
        level = force_level
    else:
        level = getLevel(df)
    
    df = changeColNames2(df)

    norm_cols_batch(df, level=level)

    df.to_csv('./ml_data.csv')
    
### pre-process for main
fp_data = pd.read_csv(feature_path, index_col=0)

uptake_col_names = list(filter(lambda x: 'CH4 v/v' in x, list(y_data.columns)))

col_names = flatten(['Crystal ID#', 'Dom. Pore (ang.)', 'Max. Pore (ang.)', 'Void Fraction', 
                                'Surf. Area (m2/g)', 'Vol. Surf. Area', 'Density', uptake_col_names])

fp_data['Crystal_Id'] = fp_data['filename'].apply(lambda x: int(x.split('_')[1]))

drop_cols = [col for col in fp_data if "Unnamed" in col]

try:
    fp_data = fp_data.drop(drop_cols, axis=1)
except:
    pass

try:
    fp_data = fp_data.drop('index', axis=1)
except:
    pass

ml_data = fp_data.merge(y_data[col_names],
                        left_on='Crystal_Id',right_on='Crystal ID#',how='inner')

for ind, col in enumerate(ml_data.columns):
    if start_str in col:
        start_col = ind + 1
    elif end_str in col:
        end_col = ind
        

non_features = list(ml_data.columns[:start_col]) + list(ml_data.columns[end_col:])
geo_props = ['Dom. Pore (ang.)',
 'Max. Pore (ang.)',
 'Void Fraction',
 'Surf. Area (m2/g)',
 'Vol. Surf. Area',
 'Density']
non_features = [col for col in non_features if col not in geo_props]

nan_cols = whichNan(ml_data, non_features)
if nan_cols != ['Dom. Pore (ang.)', 'Max. Pore (ang.)', 'Void Fraction', 'Surf. Area (m2/g)', 'Vol. Surf. Area', 'Density']:
    raise ValueError("Unexpected columns contain nan. %s" %nan_cols)
else:
    inds_of_null = getNullRows(ml_data[nan_cols + ['filename']]).index
    ml_data = ml_data.drop(inds_of_null,axis = 0)

### run main
main(ml_data)
