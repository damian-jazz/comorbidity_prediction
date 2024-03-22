import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_auc_score, brier_score_loss, f1_score, hamming_loss, balanced_accuracy_score, accuracy_score, r2_score
from sklearn.utils import resample
from sklearn.decomposition import PCA

data_path = 'data/'

########## Loading data ##########

def load_data(case: str):
    """
    Load data for specific case: 'classification', 'classification_t1', 'regression', 'combined' or 'all'
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

    # Ids t1
    t1_include = pd.read_csv(data_path + 't1_full_shape_ids.txt', header=None, names=['ID'])

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
        freesurfer_features = freesurfer_features[freesurfer_features['ID'].isin(subject_features['ID'])].reset_index(drop=True)
        classification_labels = labels_classification[labels_classification['ID'].isin(subject_features['ID'])].reset_index(drop=True)
        return subject_features, freesurfer_features, classification_labels
    elif case == 'classification_t1':
        subject_features = subject_features_classification[subject_features_classification['ID'].isin(t1_include['ID'])].reset_index(drop=True)
        freesurfer_features = freesurfer_features[freesurfer_features['ID'].isin(t1_include['ID'])].reset_index(drop=True)
        classification_labels = labels_classification[labels_classification['ID'].isin(t1_include['ID'])].reset_index(drop=True)
        return subject_features, freesurfer_features, classification_labels
    elif case == 'regression':
        subject_features = subject_features_regression
        freesurfer_features = freesurfer_features[freesurfer_features['ID'].isin(subject_features['ID'])].reset_index(drop=True)
        regression_labels = labels_regression[labels_regression['ID'].isin(subject_features['ID'])].reset_index(drop=True)
        return subject_features, freesurfer_features, regression_labels
    elif case == 'combined':
        subject_features = subject_features_combined
        freesurfer_features = freesurfer_features[freesurfer_features['ID'].isin(subject_features['ID'])].reset_index(drop=True)
        classification_labels = labels_classification[labels_classification['ID'].isin(subject_features['ID'])].reset_index(drop=True)
        regression_labels = labels_regression[labels_regression['ID'].isin(subject_features['ID'])].reset_index(drop=True)
        return subject_features, freesurfer_features, classification_labels, regression_labels
    elif case == 'all':
        subject_features = subject_features_preprocessed
        freesurfer_features = freesurfer_features[freesurfer_features['ID'].isin(subject_features['ID'])].reset_index(drop=True)
        classification_labels = labels_classification[labels_classification['ID'].isin(subject_features['ID'])].reset_index(drop=True)
        regression_labels = labels_regression[labels_regression['ID'].isin(subject_features['ID'])].reset_index(drop=True)
        return subject_features, freesurfer_features, classification_labels, regression_labels
    else:
        return

def load_confounders(subject_data: pd.DataFrame, case='standard') -> pd.DataFrame:
    """
    subject_data: input df
    case: standard, discrete or raw
    Returns pd.Dataframe object with confounders
    """
    if case == 'discrete':
        confounders = ['Age', 'Sex', 'Site', 'Field_Strength', 'Cohort']
    elif case == 'raw':
        confounders = ['Age', 'Sex', 'Site', 'Cohort']
    else:
        confounders = ['Age', 'Sex', 'Site', 'Field_Strength']
    
    C = pd.DataFrame()
    for l in confounders:
        C[l] = subject_data[l]

    if case == 'discrete':
        C = pd.get_dummies(C, columns=['Cohort'])
        C = pd.get_dummies(C, columns=['Field_Strength'])

    if case == 'standard' or case == 'discrete':
        C = pd.get_dummies(C, columns=['Site']) # Get 1-Hot encoding for sites

        sex_dict = {'Male': 0.0, 'Female': 1.0}
        C.loc[:,'Sex'] = C['Sex'].map(sex_dict)
    
        for col in C:
            C[col] = C[col].astype(float)

    return C

def load_confounder_list() -> list[str]:
    return ['Age', 'Sex', 'EHQ_Total', 'Site', 'Field_Strength']

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

def load_roi_labels(df: pd.DataFrame):
    """
    Returns 2 lists: roi labels for aseg and for aparc
    """
    #df = pd.read_csv(data_path + 'fs_features_total.csv', index_col=0)

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

def remove_zero_features(df: pd.DataFrame, return_deleted=False):
    """
    Returns pd.Dataframe and list with labels that have been removed 
    """
    zero_cols = df.columns[df.eq(0).all()]
    df = df.drop(zero_cols, axis=1)
    affected_label_names = zero_cols.tolist()

    if return_deleted == True:
        return df, affected_label_names
    else:
        return df

def standardize(df: pd.DataFrame):
    """
    Returns pd.Dataframe that has been standardaized with StandardScaler() from sklearn
    """
    scaler = StandardScaler()
    df_standardized = df.copy()
    columns = df_standardized.columns.to_list()
    df_standardized[columns] = scaler.fit_transform(df_standardized[columns])
    return df_standardized

def deconfound_linear(C: pd.DataFrame, F: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters: confounders C and features F 
    Returns pd.Dataframe with features that have been deconfounded by substracting residual obtained with LinearRegression() 
    """
    reg = LinearRegression().fit(C, F)
    F_deconf = F - reg.predict(C)
    return F_deconf

def label_freq_sorted(df: pd.DataFrame) -> list[int]:
    """
    Returns list of ints with indices of columns from df sorted by frequency
    """
    label_sorted_by_frequency = df.sum().sort_values(ascending=False).index.tolist()
    label_mapping = {label: index for index, label in enumerate(df.columns)}
    index_list = [label_mapping[label] for label in label_sorted_by_frequency]
    return index_list

def generate_label_stats(df: pd.DataFrame, mean_ir=False) -> pd.DataFrame:
    """
    Returns pd.Dataframe with label statistics and optionally also mean_ir
    """
    diagnosis_counts = dict()
    ir_dict = dict()

    for name, _ in df.items():
        diagnosis_counts[name] = df[name].value_counts()[1]

    max_value = max(diagnosis_counts.values())

    for k,v in diagnosis_counts.items():
        ir_dict[k] = (max_value / v)

    stats = pd.DataFrame({
    'Absolute frequency': df.sum(),
    'Relative frequency': df.mean(),
    'Imbalance ratio': ir_dict.values(),
    })

    mean_ratio = np.mean(list(ir_dict.values())) 
    
    if not mean_ir:
        return stats
    else:
        return stats, mean_ratio

def pca_transform(df: pd.DataFrame, n_components=10, stats=False):
    """
    Returns pd.Dataframe that contains input data projected on PCA components
    """
    pca = PCA(n_components=n_components)
    pca = pca.fit(df)
    df_transformed = pd.DataFrame(pca.transform(df))
    
    if stats == False:
        return df_transformed
    else:
        df_transformed, pca.explained_variance_ratio_


########## Resampling ###########

def generate_oversampled_set(X: pd.DataFrame, Y: pd.DataFrame):
    """
    Returns two pd.Dataframe objects with additional datapoints
    """
    label_stats = generate_label_stats(Y)
    X_over, Y_over = X.copy(), Y.copy()
    
    for i, col in enumerate(Y):
        if col != 'Attention_Deficit_HyperactivityDisorder':
            if col != 'AnxietyDisorders':
                #print(int((label_stats.iloc[i,2]*label_stats.iloc[i,0])-label_stats.iloc[i,0]), col)
                #print(X[Y[col]==1].shape, Y[Y[col]==1].shape)

                X_select, Y_select = X[Y[col]==1], Y[Y[col]==1]
                X_select, Y_select = X_select[Y_select.iloc[:,2]==0], Y_select[Y_select.iloc[:,2]==0] # exclude ADHD
                X_select, Y_select = X_select[Y_select.iloc[:,-1]==0], Y_select[Y_select.iloc[:,-1]==0] # exclude Anxiety
                
                X_, Y_ = resample(X_select, Y_select, replace=True, n_samples=int((label_stats.iloc[i,2]*label_stats.iloc[i,0])-label_stats.iloc[i,0]), random_state=0)
                X_over = pd.concat([X_over, X_], axis=0)
                Y_over = pd.concat([Y_over, Y_], axis=0)
            else:
                X_select, Y_select = X[Y[col]==1], Y[Y[col]==1]
                
                X_, Y_ = resample(X_select, Y_select, replace=True, n_samples=int((label_stats.iloc[i,2]*label_stats.iloc[i,0])), random_state=0)
                X_over = pd.concat([X_over, X_], axis=0)
                Y_over = pd.concat([Y_over, Y_], axis=0)
        else:
            X_select, Y_select = X[Y[col]==1], Y[Y[col]==1]
            X_, Y_ = resample(X_select, Y_select, replace=True, n_samples=int((label_stats.iloc[i,2]*label_stats.iloc[i,0])), random_state=0)
            X_over = pd.concat([X_over, X_], axis=0)
            Y_over = pd.concat([Y_over, Y_], axis=0)
    
    return X_over, Y_over

def generate_undersampled_set(X: pd.DataFrame, Y: pd.DataFrame):
    """
    Returns two pd.Dataframe objects with additional datapoints
    """
    X_under, Y_under = X.copy(), Y.copy()
    
    # 1. Exclude all ADHD & Anxiety comorbidities
    Y_select = Y[Y.iloc[:,2]==1]
    Y_select = Y_select[Y_select.iloc[:,-1]==1]
    Y_under = Y_under.drop(Y_select.index, errors='ignore')
    X_under = X_under.drop(Y_select.index, errors='ignore')

    # 2. Exclude 4/5 of all ADHD-only samples
    Y_select = Y_under[(Y_under['Attention_Deficit_HyperactivityDisorder'] == 1) & (Y_under.loc[:, Y.columns != 'Attention_Deficit_HyperactivityDisorder'].sum(axis=1) == 0)]
    Y_select = Y_select.sample(frac=9/10, random_state=1)
    Y_under = Y_under.drop(Y_select.index, errors='ignore')
    X_under = X_under.drop(Y_select.index, errors='ignore')

    # 3. Exclude all ADHD and specific learning disorder samples
    Y_select = Y_under[Y_under.iloc[:,2]==1]
    Y_select = Y_select[Y_select.iloc[:,7]==1]
    
    labels_to_exclude = list(Y.columns)
    labels_to_exclude.remove('Attention_Deficit_HyperactivityDisorder')
    labels_to_exclude.remove('SpecificLearningDisorder')
    for l in labels_to_exclude:
        Y_select = Y_select[Y_select[l] == 0]
        return X_under, Y_under
    
    Y_under = Y_under.drop(Y_select.index, errors='ignore')
    X_under = X_under.drop(Y_select.index, errors='ignore')

    # 4. Exclude all ADHD and disruptive disorder samples
    Y_select = Y_under[Y_under.iloc[:,2]==1]
    Y_select = Y_select[Y_select.iloc[:,9]==1]

    labels_to_exclude = list(Y.columns)
    labels_to_exclude.remove('Attention_Deficit_HyperactivityDisorder')
    labels_to_exclude.remove('Disruptive')
    for l in labels_to_exclude:
        Y_select = Y_select[Y_select[l] == 0]
    
    Y_under = Y_under.drop(Y_select.index, errors='ignore')
    X_under = X_under.drop(Y_select.index, errors='ignore')

    return X_under, Y_under

########## Multi-label scoring ###########
    
def compute_atomic(estimator, X_test, Y_test, iteration):
        
        X_test_resampled, y_test_resampled = resample(X_test, Y_test, replace=True, n_samples=len(Y_test), random_state=0+iteration)

        Y_prob = estimator.predict_proba(X_test_resampled)
        Y_pred = estimator.predict(X_test_resampled)

        # Combine prediction probas into single ndarray
        Y_prob_merged = Y_prob[0][:,1].reshape(-1,1)
        for i in range(1, len(Y_test.columns), 1):
                Y_prob_merged = np.concatenate([Y_prob_merged, Y_prob[i][:,1].reshape(-1,1)], axis=1)
        
        # Compute brier score
        brier_w = 0
        acc_w = 0
        brier_scores = np.zeros(Y_test.shape[1])
        acc_scores = np.zeros(Y_test.shape[1])

        for i in range(Y_test.shape[1]):    
            brier_scores[i] = brier_score_loss(y_test_resampled.iloc[:,i], Y_prob_merged[:, i])
            acc_scores[i] = balanced_accuracy_score(y_test_resampled.iloc[:,i], Y_pred[:, i])
            
            brier_w += brier_scores[i] * (Y_test.iloc[:,i].sum() / Y_test.shape[0])
            acc_w += acc_scores[i] * (Y_test.iloc[:,i].sum() / Y_test.shape[0])

        # Store results
        score_dict = {
               'auprc_macro': average_precision_score(y_test_resampled, Y_prob_merged, average='macro'),
               'auprc_weighted': average_precision_score(y_test_resampled, Y_prob_merged, average='weighted'),
               'auroc_macro': roc_auc_score(y_test_resampled, Y_prob_merged, average='macro'),
               'auroc_weighted': roc_auc_score(y_test_resampled, Y_prob_merged, average='weighted'),
               'brier_macro': brier_scores.mean(),
               'brier_weighted': brier_w / Y_test.shape[1],
               'balanced_accuracy_macro': acc_scores.mean(),
               'balanced_accuracy_weighted': acc_w / Y_test.shape[1],
               'f1_micro': f1_score(y_test_resampled, Y_pred, average='micro'),
               'hamming': hamming_loss(y_test_resampled, Y_pred),
               'subset_accuracy': accuracy_score(y_test_resampled, Y_pred),
        }

        return score_dict

def compute_atomic_chain(estimator, X_test, Y_test, iteration):
        
        X_test_resampled, y_test_resampled = resample(X_test, Y_test, replace=True, n_samples=len(Y_test), random_state=0+iteration)

        Y_prob = estimator.predict_proba(X_test_resampled)
        Y_pred = estimator.predict(X_test_resampled)

        # Compute brier score
        brier_w = 0
        acc_w = 0
        brier_scores = np.zeros(Y_test.shape[1])
        acc_scores = np.zeros(Y_test.shape[1])

        for i in range(Y_test.shape[1]):    
            brier_scores[i] = brier_score_loss(y_test_resampled.iloc[:,i], Y_prob[:, i])
            acc_scores[i] = balanced_accuracy_score(y_test_resampled.iloc[:,i], Y_pred[:, i])
            
            brier_w += brier_scores[i] * (Y_test.iloc[:,i].sum() / Y_test.shape[0])
            acc_w += acc_scores[i] * (Y_test.iloc[:,i].sum() / Y_test.shape[0])

        # Store results
        score_dict = {
               'auprc_macro': average_precision_score(y_test_resampled, Y_prob, average='macro'),
               'auprc_weighted': average_precision_score(y_test_resampled, Y_prob, average='weighted'),
               'auroc_macro': roc_auc_score(y_test_resampled, Y_prob, average='macro'),
               'auroc_weighted': roc_auc_score(y_test_resampled, Y_prob, average='weighted'),
               'brier_macro': brier_scores.mean(),
               'brier_weighted': brier_w / Y_test.shape[1],
               'balanced_accuracy_macro': acc_scores.mean(),
               'balanced_accuracy_weighted': acc_w / Y_test.shape[1],
               'f1_micro': f1_score(y_test_resampled, Y_pred, average='micro'),
               'hamming': hamming_loss(y_test_resampled, Y_pred),
               'subset_accuracy': accuracy_score(y_test_resampled, Y_pred),
        }

        return score_dict


def compute_scores(fitted_estimator, X_test, Y_test, boot_iter, chain=False):

    score_dict = {
            'auprc_macro': [],
            'auprc_weighted': [],
            'auroc_macro': [],
            'auroc_weighted': [],
            'brier_macro': [],
            'brier_weighted': [],
            'balanced_accuracy_macro': [],
            'balanced_accuracy_weighted': [],
            'f1_micro': [],
            'hamming': [],
            'subset_accuracy': [],
    }

    if chain == True:
        scores = [compute_atomic_chain(fitted_estimator, X_test, Y_test, i) for i in range(boot_iter)]
    else:
        scores = [compute_atomic(fitted_estimator, X_test, Y_test, i) for i in range(boot_iter)]

    # Aggregate scores
    for k,_ in score_dict.items():
        for dict in scores:
            score_dict[k].append(dict[k])

    print(f"Mean scores with SE and 95% confidence intervals:\n")

    for k,v in score_dict.items():
        print(f"{(k + ':').ljust(30)}{np.mean(v):.2f} ({np.std(v):.2f}) [{np.percentile(v, 2.5):.2f}, {np.percentile(v, 97.5):.2f}]")

########## Univariate scoring ###########
        
def compute_univariate_scores(X_train, X_test, Y_train, Y_test, area, measure_list, metric, boot_iter):
    measure_dicts = []

    for m in measure_list:
        
        m_dict = {}
        F_train = load_feature_subset(X_train, area, [m], -1)
        F_test = load_feature_subset(X_test, area, [m], -1)
        
        for y in Y_train:
            scores = []
            
            for roi in F_train:
                if metric == 'r2':
                    model = LinearRegression(n_jobs=-1).fit(F_train[roi].to_numpy().reshape(-1,1), Y_train[y])
                elif metric == 'auroc':
                    model = LogisticRegression(max_iter=10000, n_jobs=-1).fit(F_train[roi].to_numpy().reshape(-1,1), Y_train[y])
                
                score = []             
                for i in range(boot_iter):
                    X_test_resampled, y_test_resampled = resample(F_test[roi], Y_test[y], replace=True, n_samples=len(F_test), random_state=0+i)

                    if metric == 'r2':
                        y_pred = model.predict(X_test_resampled.to_numpy().reshape(-1,1))
                        score.append(r2_score(y_test_resampled, y_pred))
                    elif metric == 'auroc':
                        y_prob = model.predict_proba(X_test_resampled.to_numpy().reshape(-1,1))[:,1]
                        score.append(roc_auc_score(y_test_resampled, y_prob))

                scores.append(np.mean(score))
                
            m_dict[y] = scores
        measure_dicts.append(m_dict)
    
    return measure_dicts

def compute_univariate_scores_global(X_train, X_test, Y_train, Y_test, measure_list, metric, boot_iter):

    score_dict = {}

    for y in Y_train:
        
        scores = [] 

        for m in measure_list:
            f_train = load_feature_subset(X_train, 'global', [m])
            f_test = load_feature_subset(X_test, 'global', [m])
            
            if metric == 'r2':
                model = LinearRegression(n_jobs=-1).fit(f_train.to_numpy().reshape(-1,1), Y_train[y])
            elif metric == 'auroc':
                model = LogisticRegression(max_iter=10000, n_jobs=-1).fit(f_train.to_numpy().reshape(-1,1), Y_train[y])    

            score = []
            for i in range(100):
                X_test_resampled, y_test_resampled = resample(f_test, Y_test[y], replace=True, n_samples=len(f_test), random_state=0+i)
                
                if metric == 'r2':
                    y_pred = model.predict(X_test_resampled.to_numpy().reshape(-1,1))
                    score.append(r2_score(y_test_resampled, y_pred))
                elif metric == 'auroc':
                    y_prob = model.predict_proba(X_test_resampled.to_numpy().reshape(-1,1))[:,1]
                    score.append(roc_auc_score(y_test_resampled, y_prob))
                else:
                    pass 
                
            scores.append(np.mean(score))

        score_dict[y] = scores
    
    return score_dict

def compute_univariate_auroc_scores(X_train, X_test, Y_train, Y_test, boot_iter):

    score_dict = {}

    for x in X_train:
        
        scores = [] 
        for y in Y_train:           
            model = LogisticRegression(max_iter=10000, n_jobs=-1).fit(X_train[x].to_numpy().reshape(-1,1), Y_train[y])    

            score = []
            for i in range(100):
                X_test_resampled, y_test_resampled = resample(X_test[x], Y_test[y], replace=True, n_samples=len(Y_test), random_state=0+i)
                y_prob = model.predict_proba(X_test_resampled.to_numpy().reshape(-1,1))[:,1]
                score.append(roc_auc_score(y_test_resampled, y_prob))

            scores.append(np.mean(score))

        score_dict[x] = scores
    
    return score_dict