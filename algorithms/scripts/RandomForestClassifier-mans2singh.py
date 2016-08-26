
# coding: utf-8

# # Create a RandomForestClassifier model to predict TP53 mutation from gene expression data in TCGA

# In[243]:

get_ipython().magic(u'matplotlib inline')
plt.style.use('seaborn-notebook')


# ## Specify model configuration

# In[244]:

# We're going to be building a 'TP53' classifier 
GENE = 'TP53'


# In[245]:

# Parameter Sweep for Hyperparameters
n_feature_kept = 5000
param_fixed = {
    'min_samples': 100,
    'min_samples_split': 4,
    'class_weight' : 'balanced'
}
param_grid = {
    'max_depth': [x for x in range(1, 10)],
    'n_estimators' : [ 10 ** x for x in  range(1,4)]
}


# *Here is some [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) regarding the classifier and hyperparameters*
# 
# *Here is some [information](https://ghr.nlm.nih.gov/gene/TP53) about TP53*

# ## Load Data

# In[246]:

if not os.path.exists('data'):
    os.makedirs('data')


# In[247]:

url_to_path = {
    # X matrix
    'https://ndownloader.figshare.com/files/5514386':
        os.path.join('data', 'expression.tsv.bz2'),
    # Y Matrix
    'https://ndownloader.figshare.com/files/5514389':
        os.path.join('data', 'mutation-matrix.tsv.bz2'),
}

for url, path in url_to_path.items():
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)


# In[248]:

get_ipython().run_cell_magic(u'time', u'', u"path = os.path.join('data', 'expression.tsv.bz2')\nX = pd.read_table(path, index_col=0)")


# In[211]:

get_ipython().run_cell_magic(u'time', u'', u"path = os.path.join('data', 'mutation-matrix.tsv.bz2')\nY = pd.read_table(path, index_col=0)")


# In[212]:

y = Y[GENE]


# In[213]:

Y.head(6)


# In[214]:

# The Series now holds TP53 Mutation Status for each Sample
y.head(6)


# In[215]:

# Here are the percentage of tumors with NF1
y.value_counts(True)


# ## Set aside 10% of the data for testing

# In[216]:

# Typically, this can only be done where the number of mutations is large enough
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
'Size: {:,} features, {:,} training samples, {:,} testing samples'.format(len(X.columns), len(X_train), len(X_test))


# ## Median absolute deviation feature selection

# In[217]:

def fs_mad(x, y):
    """    
    Get the median absolute deviation (MAD) for each column of x
    """
    scores = mad(x) 
    return scores, np.array([np.NaN]*len(scores))

# select the top features with the highest MAD
feature_select = SelectKBest(fs_mad, k=n_feature_kept)


# ## Define pipeline and Cross validation model fitting

# In[218]:

clf = RandomForestClassifier(min_samples_leaf=5, random_state=2)
# joblib is used to cross-validate in parallel by setting `n_jobs=-1` in GridSearchCV
# Supress joblib warning. See https://github.com/scikit-learn/scikit-learn/issues/6370
warnings.filterwarnings('ignore', message='Changing the shape of non-C contiguous array')
clf_grid = grid_search.GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1, scoring='roc_auc')
pipeline = make_pipeline(
    feature_select,  # Feature selection
    StandardScaler(),  # Feature scaling
    clf_grid)


# In[219]:

get_ipython().run_cell_magic(u'time', u'', u'# Fit the model (the computationally intensive part)\npipeline.fit(X=X_train, y=y_train)\nbest_clf = clf_grid.best_estimator_\nfeature_mask = feature_select.get_support()  # Get a boolean array indicating the selected features')


# In[220]:

clf_grid.best_params_


# In[221]:

best_clf


# ## Visualize hyperparameters performance

# In[222]:

def grid_scores_to_df(grid_scores):
    """
    Convert a sklearn.grid_search.GridSearchCV.grid_scores_ attribute to 
    a tidy pandas DataFrame where each row is a hyperparameter-fold combinatination.
    """
    rows = list()
    for grid_score in grid_scores:
        for fold, score in enumerate(grid_score.cv_validation_scores):
            row = grid_score.parameters.copy()
            row['fold'] = fold
            row['score'] = score
            rows.append(row)
    df = pd.DataFrame(rows)
    return df


# ## Process Mutation Matrix

# In[223]:

cv_score_df = grid_scores_to_df(clf_grid.grid_scores_)
cv_score_df.head(2)


# In[224]:

# Cross-validated performance distribution
facet_grid = sns.factorplot(x='max_depth', y='score', col='n_estimators',
    data=cv_score_df, kind='violin', size=4, aspect=1)
facet_grid.set_ylabels('AUROC');


# In[225]:

# Cross-validated performance heatmap
cv_score_mat = pd.pivot_table(cv_score_df, values='score', index='max_depth', columns='n_estimators')
ax = sns.heatmap(cv_score_mat, annot=True, fmt='.1%')
ax.set_xlabel('(n_estimators)')
ax.set_ylabel('(max_depth)');


# ## Use Optimal Hyperparameters to Output ROC Curve

# In[226]:

y_pred_train = pipeline.predict_proba(X_train)[:, 1]
y_pred_test = pipeline.predict_proba(X_test)[:, 1]

def get_threshold_metrics(y_true, y_pred):
    roc_columns = ['fpr', 'tpr', 'threshold']
    roc_items = zip(roc_columns, roc_curve(y_true, y_pred))
    roc_df = pd.DataFrame.from_items(roc_items)
    auroc = roc_auc_score(y_true, y_pred)
    return {'auroc': auroc, 'roc_df': roc_df}

metrics_train = get_threshold_metrics(y_train, y_pred_train)
metrics_test = get_threshold_metrics(y_test, y_pred_test)


# In[227]:

# Plot ROC
plt.figure()
for label, metrics in ('Training', metrics_train), ('Testing', metrics_test):
    roc_df = metrics['roc_df']
    plt.plot(roc_df.fpr, roc_df.tpr,
        label='{} (AUROC = {:.1%})'.format(label, metrics['auroc']))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Predicting TP53 mutation from gene expression (ROC curves)')
plt.legend(loc='lower right');


# ## What are the classifier coefficients?

# In[233]:

best_clf.feature_importances_


# In[236]:

feature_importances_df = pd.DataFrame(best_clf.feature_importances_.transpose(), index=X.columns[feature_mask], columns=['weight'])
feature_importances_df['abs'] = feature_importances_df['weight'].abs()
feature_importances_df = feature_importances_df.sort_values('abs', ascending=False)


# In[237]:

'{:.1%} zero coefficients; {:,} negative and {:,} positive coefficients'.format(
    (feature_importances_df.weight == 0).mean(),
    (feature_importances_df.weight < 0).sum(),
    (feature_importances_df.weight > 0).sum()
)


# In[238]:

feature_importances_df.head(10)


# The results are not surprising. TP53 is a transcription modulator and when it mutated in a tumor, the cell goes haywire. This makes finding a transcriptional signature fairly easy. Also, the genes that the classifier uses is interesting, but not necessarily novel.
# 
# 1. TP53 is a [transcription factor](https://en.wikipedia.org/wiki/Transcription_factor "TF wiki") that regulates many genes including EDA2R. Studies have linked EDA2R (or XEDAR) to [increased survival in colon cancer patients](http://www.ncbi.nlm.nih.gov/pubmed/19543321) and [losing hair as a result of chemotherapy](http://onlinelibrary.wiley.com/doi/10.1016/j.febslet.2010.04.058/full)
# 2. SPATA18 is a gene associated with spermatogenesis and is a transcription factor for TP53. It's association with TP53 was [recently discovered](http://www.ncbi.nlm.nih.gov/pubmed/21300779) in 2011.
# 3. C6orf138 (or [PTCHD4](http://www.genecards.org/cgi-bin/carddisp.pl?gene=PTCHD4)) is also a transcriptional target for TP53 and was only recently discovered in [2014 to repress hedgehog signalling](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC4239647/).
# 4. The list goes on and includes several other TP53 targets...

# ## Investigate the predictions

# In[240]:

predict_df = pd.DataFrame.from_items([
    ('sample_id', X.index),
    ('testing', X.index.isin(X_test.index).astype(int)),
    ('status', y),
    ('probability', pipeline.predict_proba(X)[:, 1]),
])
predict_df['probability_str'] = predict_df['probability'].apply('{:.1%}'.format)


# In[241]:

# Top predictions amongst negatives (potential hidden responders)
predict_df.sort_values('probability', ascending=False).query("status == 0").head(10)


# In[242]:

# Ignore numpy warning caused by seaborn
warnings.filterwarnings('ignore', 'using a non-integer number instead of an integer')

ax = sns.distplot(predict_df.query("status == 0").probability, hist=False, label='Negatives')
ax = sns.distplot(predict_df.query("status == 1").probability, hist=False, label='Positives')


# In[ ]:




# In[ ]:



