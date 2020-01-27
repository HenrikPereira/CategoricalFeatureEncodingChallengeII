# %%
import string
import numpy as np
import pandas as pd
import pandas_profiling as pdf
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import seaborn as sb
from xgboost import XGBClassifier, XGBRFClassifier

# %%
df_train = pd.read_csv(r'./Data/train.csv', index_col='id')
# df_test = pd.read_csv(r'./Data/train.csv', index_col='id')

# %%
train_profile = pdf.ProfileReport(df_train, minimal=True)
test_profile = pdf.ProfileReport(df_train, minimal=True)

train_profile.to_file(output_file=r'./Data/train_profile.html')
test_profile.to_file(output_file=r'./Data/test_profile.html')

# train_profile = pdf.ProfileReport(df_train, minimal=False)
# test_profile = pdf.ProfileReport(df_train, minimal=False)
#
# train_profile.to_file(output_file=r'./Data/train_prof_comp.html')
# test_profile.to_file(output_file=r'./Data/test_prof_comp.html')

# %%
print(100 - len(df_train.dropna()) / len(df_train) * 100, '% of rows removed when indiscriminate drop nan made...')

# %%
df_train = df_train.applymap(lambda x: np.nan if x == 'nan' else x)
# df_test = df_test.applymap(lambda x: np.nan if x == 'nan' else x)
# df_train_md = df_train.copy().fillna(df_train.median())

# F/T & N/Y --> turn to binary
df_train[['bin_3', 'bin_4']] = df_train[['bin_3', 'bin_4']].replace(['F', 'N'], 0).replace(['T', 'Y'], 1)

# low cardinality categorical --> to dummies
lc_cat = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']
for col in lc_cat:
    for i, v in enumerate(df_train[col].dropna().unique()):
        df_train[col] = df_train[col].replace(v, i)

# hashed categorical, high cardinality --> to ???
hash_cat = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']

# normal ordinal, low cardinality --> to dummies?
lc_ord = ['ord_0', 'ord_1', 'ord_2']
ord_1_dict = {'Contributor': 2, 'Grandmaster': 5, 'Novice': 1, 'Expert': 3, 'Master': 4}
ord_2_dict = {'Hot': 4, 'Warm': 3, 'Freezing': 1, 'Lava Hot': 6, 'Cold': 2, 'Boiling Hot': 5}

for k in ord_1_dict:
    df_train['ord_1'] = df_train['ord_1'].replace(k, ord_1_dict[k])
for k in ord_2_dict:
    df_train['ord_2'] = df_train['ord_2'].replace(k, ord_2_dict[k])

# ordinal alphabet based --> to dummies?
df_train['ord_5_0'] = df_train['ord_5'].str.slice(start=0, stop=1)
df_train['ord_5_1'] = df_train['ord_5'].str.slice(start=1, stop=2)
ab_ord = ['ord_3', 'ord_4', 'ord_5_0', 'ord_5_1']

for col in ab_ord:
    for i, v in enumerate(string.ascii_letters):
        df_train[col] = df_train[col].replace(v, i)

# df_train_dum = pd.get_dummies(df_train, dummy_na=False, columns=lc_cat + lc_ord + ab_ord)

# %%
df_nom_5 = df_train.copy() \
    .dropna(subset=['nom_5']) \
    .drop(columns=['target', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_5']) \
    .dropna()
le = LabelEncoder()
df_nom_5['nom_5'] = le.fit_transform(df_nom_5['nom_5'])

X_train, X_test, y_train, y_test = train_test_split(
    df_nom_5.drop(columns=['nom_5']), df_nom_5['nom_5'], test_size=0.25, random_state=123
)

dt_5 = OneVsRestClassifier(
    RandomForestClassifier(
        n_jobs=-1, max_depth=10, oob_score=True,
        class_weight='balanced', random_state=123
    ))
dt_5.fit(X=X_train, y=y_train)
y_hat = dt_5.predict(X=X_test)
print("MCC:", metrics.matthews_corrcoef(y_test, y_hat))

results_5 = pd.DataFrame(dt_5.feature_importances_.reshape(-1, 1).transpose(), columns=X_train.columns) \
    .sort_values(by=0, axis=1, ascending=False)

plt.figure(figsize=(5, 5))
plt.scatter(results_5.columns, results_5.values)
plt.xticks(rotation=90)
plt.show()

# %%
xg_5 = XGBClassifier(n_jobs=-1).fit(X_train, y_train)
y_hat = xg_5.predict(X_test)
print("MCC:", metrics.matthews_corrcoef(y_test, y_hat))

# %%
results_5 = pd.DataFrame(xg_5.feature_importances_.reshape(-1, 1).transpose(), columns=X_train.columns) \
    .sort_values(by=0, axis=1, ascending=False)

plt.figure(figsize=(5, 5))
plt.scatter(results.columns, results.values)
plt.xticks(rotation=90)
plt.show()

# %%
# df_nom_6 = df_train_dum.copy().dropna(subset=['nom_6']).drop(columns=['nom_5', 'nom_7', 'nom_8', 'nom_9']).dropna()
# df_nom_7 = df_train_dum.copy().dropna(subset=['nom_7']).drop(columns=['nom_5', 'nom_6', 'nom_8', 'nom_9']).dropna()
# df_nom_8 = df_train_dum.copy().dropna(subset=['nom_8']).drop(columns=['nom_5', 'nom_6', 'nom_7', 'nom_9']).dropna()
# df_nom_9 = df_train_dum.copy().dropna(subset=['nom_9']).drop(columns=['nom_5', 'nom_6', 'nom_7', 'nom_8']).dropna()

# rf_6 = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
# rf_7 = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
# rf_8 = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
# rf_9 = RandomForestClassifier(n_estimators=1000, n_jobs=-1)

# rf_6.fit(X=df_nom_5.drop(columns=['nom_6']), y=df_nom_5['nom_6'])
# rf_7.fit(X=df_nom_5.drop(columns=['nom_7']), y=df_nom_5['nom_7'])
# rf_8.fit(X=df_nom_5.drop(columns=['nom_8']), y=df_nom_5['nom_8'])
# rf_9.fit(X=df_nom_5.drop(columns=['nom_9']), y=df_nom_5['nom_9'])
