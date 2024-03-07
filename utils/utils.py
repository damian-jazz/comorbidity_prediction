import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

data_path = 'data/'
plot_path = 'plots/'

########## Data loading ##########

def load_data(case: str):
    """
    Load data for specific case: 'classification', 'regression', 'combined' or 'all'
    Returns multiple pd.Dataframe objects
    """
    # Labels
    labels_classification = pd.read_csv(data_path + 'labels_classification.csv', index_col=0)
    labels_regression = pd.read_csv(data_path + 'labels_regression.csv', index_col=0)

    # FreeSurfer features for all preprocessed subjects (n=3451) 
    freesurfer_features = pd.read_csv(data_path + 'fs_features_total.csv', index_col=0)

    # Subject-related features for all preprocessed subjects (n=3451) 
    subject_features_preprocessed = pd.read_csv(data_path + 'preprocessed_subjects.csv', index_col=0)

    # Subject-related features for different subsets depending on case
    subject_features_classification = pd.read_csv(data_path + 'subject_data_classification_subset.csv', index_col=0)
    subject_features_regression = pd.read_csv(data_path + 'subject_data_regression_subset.csv', index_col=0)
    subject_features_combined = pd.read_csv(data_path + 'subject_data_combined_subset.csv', index_col=0)

    # Preprocessing: strip empty spaces from subject ids
    labels_classification['ID'] = labels_classification['ID'].str.strip()
    labels_regression['ID'] = labels_regression['ID'].str.strip()
    freesurfer_features['ID'] = freesurfer_features['ID'].str.strip()
    subject_features_preprocessed['ID'] = subject_features_preprocessed['ID'].str.strip()
    subject_features_classification['ID'] = subject_features_classification['ID'].str.strip()
    subject_features_regression['ID'] = subject_features_regression['ID'].str.strip()
    subject_features_combined['ID'] = subject_features_combined['ID'].str.strip()

    if case == 'classification':
        subject_features = subject_features_classification
        freesurfer_features = freesurfer_features[freesurfer_features["ID"].isin(subject_features["ID"])].reset_index(drop=True)
        classification_labels = labels_classification[labels_classification["ID"].isin(subject_features["ID"])].reset_index(drop=True)
        return subject_features, freesurfer_features, classification_labels
    elif case == 'regression':
        subject_features = subject_features_regression
        freesurfer_features = freesurfer_features[freesurfer_features["ID"].isin(subject_features["ID"])].reset_index(drop=True)
        regression_labels = labels_regression[labels_regression["ID"].isin(subject_features["ID"])].reset_index(drop=True)
        return subject_features, freesurfer_features, regression_labels
    elif case == 'combined':
        subject_features = subject_features_combined
        freesurfer_features = freesurfer_features[freesurfer_features["ID"].isin(subject_features["ID"])].reset_index(drop=True)
        classification_labels = labels_classification[labels_classification["ID"].isin(subject_features["ID"])].reset_index(drop=True)
        regression_labels = labels_regression[labels_regression["ID"].isin(subject_features["ID"])].reset_index(drop=True)
        return subject_features, freesurfer_features, classification_labels, regression_labels
    elif case == 'all':
        subject_features = subject_features_preprocessed
        freesurfer_features = freesurfer_features[freesurfer_features["ID"].isin(subject_features["ID"])].reset_index(drop=True)
        classification_labels = labels_classification[labels_classification["ID"].isin(subject_features["ID"])].reset_index(drop=True)
        regression_labels = labels_regression[labels_regression["ID"].isin(subject_features["ID"])].reset_index(drop=True)
        return subject_features, freesurfer_features, classification_labels, regression_labels
    else:
        return

def load_confounders(subject_data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """
    Returns pd.Dataframe object with confounders: 'TIV', 'Age', 'Sex', 'Site', 'Field_Strength' 
    """
    C = pd.DataFrame()
    C['TIV'] = features['global_estimatedtotalintracranialvol']
    C['Age'] = subject_data['Age']
    C['Sex'] = subject_data['Sex']
    C['Site'] = subject_data['Site']
    C['Field_Strength'] = subject_data['Field_Strength']

    # Preprocess C
    C = pd.get_dummies(C, columns=['Site']) # Get 1-Hot encoding for sites
    sex_dict = {'Male': 0.0, 'Female': 1.0}
    C.loc[:,'Sex'] = C['Sex'].map(sex_dict)

    # Convert everything to float
    for col in C:
        C[col] = C[col].astype(float)

    return C

def load_feature_subset(source: pd.DataFrame, area: str, measure_list: list[str], roi_idx: int=-1) -> pd.DataFrame:
    """
    source: pd.Dataframe containing all fs data
    area: select between ('global', 'aseg', 'aparc_lh', 'aparc_rh')
    measure_list: contains measure types
    roi_idx: select all rois (-1) or specific roi (index)
    Returns pd.Dataframe object with measurements for specified scope
    """
    S = source
    X = pd.DataFrame()

    if area == 'global':
        for feature in measure_list:
            idx = S.columns.get_loc(f"global_{feature}")
            X[S.columns[idx]] = S.iloc[:,idx]
    else:
        if roi_idx == 0:  # return specific ROI (default)
            if area == 'aseg':
                for feature in measure_list:
                    start_idx = S.columns.get_loc(f"aseg_left-lateral-ventricle_{feature}")
                    offset = start_idx + roi_idx
                    X[S.columns[offset]] = S.iloc[:,offset]
            elif area == 'aparc_lh':
                for feature in measure_list:
                    l_start_idx = S.columns.get_loc(f"aparc_lh_bankssts_{feature}")
                    l_offset = l_start_idx + roi_idx
                    X[S.columns[l_offset]] = S.iloc[:,l_offset]
            elif area == 'aparc_rh':
                for feature in measure_list:
                    r_start_idx = S.columns.get_loc(f"aparc_rh_bankssts_{feature}")
                    r_offset = r_start_idx + roi_idx
                    X[S.columns[r_offset]] = S.iloc[:,r_offset]
            else:
                return         
        elif roi_idx == -1: # return all ROIs
            if area == 'aseg':
                for feature in measure_list:
                    start = S.columns.get_loc(f"aseg_left-lateral-ventricle_{feature}")
                    end = S.columns.get_loc(f"aseg_cc_anterior_{feature}")
                    X = pd.concat([X, S.iloc[:,start:end+1]], axis=1)           
            elif area == 'aparc_lh':
                for feature in measure_list:
                    l_start = S.columns.get_loc(f"aparc_lh_bankssts_{feature}")
                    l_end = S.columns.get_loc(f"aparc_lh_insula_{feature}")
                    X = pd.concat([X, S.iloc[:,l_start:l_end+1]], axis=1)
            elif area == 'aparc_rh':
                for feature in measure_list:
                    r_start = S.columns.get_loc(f"aparc_rh_bankssts_{feature}")
                    r_end = S.columns.get_loc(f"aparc_rh_insula_{feature}")
                    X = pd.concat([X, S.iloc[:,r_start:r_end+1]], axis=1)
            else:
                return
        else:
            return
    
    return X

def load_roi_labels():
    """
    Returns 2 lists: roi labels for aseg and for aparc
    """
    df = pd.read_csv(data_path + 'fs_features_total.csv', index_col=0)

    #aseg
    aseg_rois = load_feature_subset(df,'aseg',["Area_mm2"],-1)
    aseg_roi_labels = aseg_rois.columns.to_list()
    aseg_roi_labels = [label.replace('aseg_', '').replace('_Area_mm2', '') for label in aseg_roi_labels]

    # aparc
    aparc_rois = load_feature_subset(df,'aparc_lh',["volume"],-1)
    aparc_roi_labels = aparc_rois.columns.to_list()
    aparc_roi_labels = [label.replace('aparc_lh_', '').replace('_volume', '') for label in aparc_roi_labels]

    return aseg_roi_labels, aparc_roi_labels


def load_measurement_labels():
    """
    Returns 3 lists: measurement labels for global, aseg and aparc
    """
    global_measurements = ["brainsegvol", "brainsegvolnotvent", "lhcortexvol", "rhcortexvol", "cortexvol", "lhcerebralwhitemattervol", "rhcerebralwhitemattervol", "cerebralwhitemattervol", "subcortgrayvol", "totalgrayvol", "supratentorialvol", "supratentorialvolnotvent", "maskvol", "brainsegvol-to-etiv", "maskvol-to-etiv", "lhsurfaceholes", "rhsurfaceholes", "surfaceholes", "estimatedtotalintracranialvol"]
    aseg_measurements = ["Area_mm2", "max", "mean", "nvertices", "nvoxels", "std", "volume"]
    aparc_measurements = ["volume", "area", "curvind", "foldind", "gauscurv", "meancurv", "thickness", "thickness.T1", "thicknessstd"]
    
    return global_measurements, aseg_measurements, aparc_measurements


########## Helper methods ###########

def remove_zero_features(df: pd.DataFrame):
    """
    Returns pd.Dataframe and list with labels that have been removed 
    """
    zero_cols = df.columns[df.eq(0).all()]
    df = df.drop(zero_cols, axis=1)
    affected_label_names = zero_cols.tolist()
    return df, affected_label_names

def standardize(df: pd.DataFrame):
    """
    Returns pd.Dataframe that has been standardaized with StandardScaler() from sklearn
    """
    scaler = StandardScaler()
    df_standardized = df.copy()
    columns = df_standardized.columns.to_list()
    df_standardized[columns] = scaler.fit_transform(df_standardized[columns])
    return df_standardized

def deconfound_linear(subject_data: pd.DataFrame, F: pd.DataFrame) -> pd.DataFrame:
    """
    Returns pd.Dataframe with features that have been deconfounded by substracting residual obtained with LinearRegression() 
    """
    C = load_confounders(subject_data, F)
    reg = LinearRegression().fit(C, F)
    F_deconf = F - reg.predict(C)
    return F_deconf

def label_index_by_frequency_sorted(df: pd.DataFrame) -> list[int]:
    """
    Returns list of ints with indices of columns from df sorted by frequency
    """
    label_sorted_by_frequency = df.sum().sort_values(ascending=False).index.tolist()
    label_mapping = {label: index for index, label in enumerate(df.columns)}
    index_list = [label_mapping[label] for label in label_sorted_by_frequency]
    return index_list

def generate_label_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns pd.Dataframe with label statistics
    """
    new_df = pd.DataFrame({
    'Mean': df.mean(),
    'Sum': df.sum()
    })
    return new_df