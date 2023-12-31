import numpy as numpy
import seaborn as sns
from matplotlib import rcParams
import matplotlib.pyplot as plt
import webbrowser
from matplotlib import ticker
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
def set_graph_params():
    rcParams['xtick.labelsize']=12
    rcParams['ytick.labelsize']=12
    rcParams['figure.figsize']=7,5
    sns.set_style('whitegrid')
def plot_Fi(importance, features, PdfFile_Fi):
    set_graph_params()
    plt.figure(figsize=(8,5))
    sns.barplot(x=features_importance.index, y=features_importance['Importance'], palette='Blues_d')
    plt.xticks(rotation=90)
    plt.set_xlabel('Feature', fontsize=14)
    plt.set_ylabel('Importance', fontsize=14)
    plt.tight_layout()
    plt.savefig(PdfFile_Fi)
def plot_Pfi(X_test, y_test, names, PdfFile_Pfi, optuna_search):
    set_graph_params()
    pfi=premutation_importance(
    estimator=optuna_search,
    X=X_test,
    y=y_test,
    scoring='neg_root_mean_squared_error',
    n_repeats=5,
    n_jobs=-1  
    )
    df_pfi=pd.DataFrame(
        data={'var_name': X_test.columns, 'importance': pfi['importances_mean']}
    ).sort_values('importance')
    fig, ax=plt.subplots(figsize=(9,6))
    ax.barh(df_pfi['var_name'], df_pfi['importance'])
    plt.xlabel('Differences', fontsize=14)
    fig.subtitle('Permuatation Importance')
    plt.xlim([0,4])
    plt.subplots_adjust(left=0.3)
    plt.savefig(PdfFile_Pfi)
def load_model(ModelName):
    with open(ModelName, mode='rb') as f:
        optuna_search=pickle.load(f)
    return optuna_search
def split_data(df, features):
    train, test=train_test_split(df, test_size=0.2, random_state=115)
    X_train=train[features]
    y_train=train['Target'].values
    X_test=test[features]
    y_test=test['Target'].values
    return X_train, y_train, x_test, y_test
base_folder=r'C:\\Users\\user\\git\\github\\py2312_fi_pls\\'
FileName=base_folder+'regression_pls.csv'
ModelName=base_folder+'rf_optuna.pickle'
PdfFile_Fi=base_folder+'pdf\\PdfFile_Fi.pdf'
PdfFile_Pfi=base_folder+'pdf\\PdfFile_Pfi.pdf'
columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','Target']
target_column='Target'
df=pd.read_csv(file_name, encoding='utf-8', engine='python', usecols=columns)
features=[c for c in df.columns if c !=target_column]
split_data(df, features)
load_model(ModelName)
pred_train=optuna_search.predict(X_train)
pred_test=optuna_search.predict(X_test)
r2_train=r2_score(y_train, pred_train)
adjusted_r2_train=1-(1-r2_train)*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)
mse_train=mean_squared_error(y_train, pred_train)
rmse_train=mse_train**0.5
r2_test=r2_score(y_test, pred_test)
adjusted_r2_test=1-(1-r2_test)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
mse_test=mean_squared_error(y_test, pred_test)
rmse_test=mse_test**0.5
set_graph_params    
sns.set_color_codes()
plt.figure()
fig, ax=plt.subplots(figsize=(7,7))
plt.xlim([-12,2])
plt.ylim([-12,2])
sns.set(font='Arial')
plt.scatter(y_train, pred_train, alpha=0.5, color='blue', label='Train')
plt.scatter(y_test, pred_test, alpha=0.5, color='green', label='Test')
plt.plot(np.linspace(-12, 2, 14), np.linspace(-12, 2, 14), 'black')
plt.xlabel('True Target', fontsize=14)
plt.ylabel('Predicted True Target', fontsize=14)
plt.title(f'Train - Adjusted R2 Score: {adjusted_r2_train:.3f}, RMSE: {rmse_train:.3f}\nTest - Adjested R2 Score: {adjusted_r2_test:.3f}, RMSE: {rmse_test:.3f}', fontsize=13)
plt.legend(fontsize=14)
plt.savefig(PdfFile)
webbrowser.open_new(PdfFile)
best_estimator=optuna_search.best_estimator_
importance=best_estimator.features_importance_
plot_Fi(importance, features, PdfFile_Fi)
plot_Pfi(X_test, y_test, features, PdfFile_Pfi)