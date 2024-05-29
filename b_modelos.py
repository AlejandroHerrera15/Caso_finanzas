import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import a_funciones as funciones
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import joblib
tabla_nuevos='https://raw.githubusercontent.com/juancamiloespana/LEA3_FIN/main/data/datos_nuevos_creditos.csv'
tabla_hist='https://raw.githubusercontent.com/juancamiloespana/LEA3_FIN/main/data/datos_historicos.csv'
tabla_nuevos=pd.read_csv(tabla_nuevos)
tabla=pd.read_csv(tabla_hist)


#sns.histplot(data=tabla_hist, x="NoPaidPerc")

#tabla_hist.info()
#tabla_hist.drop(columns=['ID','HomeOwnership', 'Education', 'MaritalStatus'],inplace=True)

#from matplotlib.pyplot import figure
#figure(figsize=(20,6))
#sns.heatmap(tabla_hist.corr(),cmap = sns.cubehelix_palette(as_cmap=True), annot = True, fmt = ".2f")

#Separación de variables explicativas con variable objetivo.
dfx=tabla.iloc[:,:-1]
dfy=tabla.iloc[:,-1]
dfx.drop(columns=["ID"],inplace=True)


cat=dfx.select_dtypes(include="object").columns
tabla[cat]
num=dfx.select_dtypes(exclude="object").columns
tabla[num]

#get_dummies
df_dum=pd.get_dummies(dfx,columns=cat)
df_dum.info()

#Escalamos las variables
scaler=StandardScaler()
scaler.fit(df_dum)
xnor=scaler.transform(df_dum)
x=pd.DataFrame(xnor,columns=df_dum.columns)
x.columns

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


#x columns contiene el dataframe ya listo para ser entrenado
#Selección de variables a traves de los modelos descritos a continuación 
mr = LinearRegression()
mdtr= DecisionTreeRegressor()
mrfr= RandomForestRegressor()
mgbr=GradientBoostingRegressor()
modelos= [ mr, mdtr, mrfr, mgbr]
var_names=funciones.sel_variables(modelos, x, dfy, threshold="2.2*mean")
var_names.shape

#Variables elegidas incialmente con el threshold 2.2 
xtrain=x[var_names] #5 variables


#Medir los modelos 
accu_x=funciones.medir_modelos(modelos,"r2",x,dfy,20) ## base con todas las variables 
accu_xtrain=funciones.medir_modelos(modelos,"r2",xtrain,dfy,20) ### base con variables seleccionadas

#Dataframe con los resultados
accu=pd.concat([accu_x,accu_xtrain],axis=1)
accu.columns=['rl', 'dt', 'rf', 'gb',
       'rl_Sel', 'dt_sel', 'rf_sel', 'gb_Sel']

#Promedio para cada modelo
np.mean(accu, axis=0)

#Gráfico de F1 score para modelos con todas las variables y modelos con variables seleccionadas
sns.boxplot(data=accu_x, palette="Set3")
sns.boxplot(data=accu_xtrain, palette="Set3")
sns.boxplot(data=accu, palette="Set3")
#en esta validación cruzada que incluye todas la variables y las variables con threshold de 2.2*mean
#se observa que los modelos DCT y RFC sostienen la misma metrica F1 score, desde aqui se sospecha
#que pueden ser los modelos elegidos

# "función" para buscar el mejor threshold que seleccina las variables para cada modelo.------------------
df_resultado = pd.DataFrame()
thres=0.1
for i in range(10):
    df_actual=0
    var_names=funciones.sel_variables(modelos, x, dfy, threshold="{}*mean".format(thres))
    xtrain=x[var_names]
    accu_xtrain=funciones.medir_modelos(modelos,"r2",xtrain,dfy,10)
    df=accu_xtrain.mean(axis=0)
    df_actual = pd.DataFrame(df, columns=['threshold {}'.format(thres)])
    df_resultado = pd.concat([df_resultado, df_actual], axis=1)
    thres+=0.15
    thres=round(thres,2)

#Gráfica de los resultados __________________________________
df=df_resultado.T
plt.figure(figsize=(10,10))
sns.lineplot(data=df)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.ylabel("r2")
plt.title("Variacion threshold")
#Como se observa los modelos DCT Y RFC no se ven afectados por la cantidad de variables selecionados por los estimadores,
#sin embargo los otros dos modelos al incrementar la exigencia en el threshold dejan caer su rendimiento

#Mejor threshold para cada modelo
df.idxmax(axis=0)
#Sin embargo los threshold con los mejores rendimientos en los modelos DTC Y RFC 
#estan exigiendo tanto a la seleccion de variables que solo aparece una o dos variables.

#Los dos modelos a tunear son random_forest y decision_tree con un trheshold de 2.2 por la media,
modelos= [ mr, mdtr, mrfr, mgbr]
var_names=funciones.sel_variables(modelos, x, dfy, threshold="0.1*mean")
var_names.shape
#Finalmente se escogen 5 variables para entrenar el modelo, se determino este número
#ya que según la gráfica presentan un desempeño casi igual al threshold con mayor rendimiento

#tabla final
xtrainf=x[var_names]
joblib.dump(xtrainf, "salidas\\xtrainf.pkl") 

#Al final se deja este threshold que da como resultado 5 variables

#Volvemos a medir el modelo pero con las 5 variables y todas las variables
#accu_x=funciones.medir_modelos(modelos,"f1",x,dfy,20) ## base con todas las variables 
accu_xtrainf=funciones.medir_modelos(modelos,"r2",xtrainf,dfy,10) ### base con variables seleccionadas
accu_xtrainf.mean()



param_grid = {
    'n_estimators': [100, 200, 300],  # Número de árboles en el bosque
    'max_depth': [5, 10, 20],  # Profundidad máxima de cada árbol
    'min_samples_split': [2, 5, 10],  # Número mínimo de muestras requeridas para dividir un nodo interno
    'min_samples_leaf': [1, 2, 4]  # Número mínimo de muestras requeridas para estar en un nodo hoja
    # Número máximo de características a considerar en cada división
     # Método de selección de muestras para el entrenamiento de cada árbol
}

rfrtuning=RandomForestRegressor()
grid_search1=GridSearchCV(rfrtuning, param_grid, scoring="r2",cv=10, n_jobs=-1)
grid_result1=grid_search1.fit(xtrainf, dfy)

pd.set_option('display.max_colwidth', 100)
resultados1=grid_result1.cv_results_
grid_result1.best_params_
pd_resultados1=pd.DataFrame(resultados1)
pd_resultados1[["params","mean_test_score"]].sort_values(by="mean_test_score", ascending=False)

rfr_final=grid_result1.best_estimator_ 
joblib.dump(rfr_final, "salidas\\rfr_final.pkl") 




param_grid2 = {
    'n_estimators': [50, 100, 200],  # Número de árboles en el ensamble
    'learning_rate': [0.01, 0.1, 0.5],  # Tasa de aprendizaje
    'max_depth': [3, 5, 7],  # Profundidad máxima de cada árbol base
    'min_samples_split': [2, 5, 10],  # Número mínimo de muestras requeridas para dividir un nodo interno
    'min_samples_leaf': [1, 2, 4]  # Número mínimo de muestras requeridas para estar en un nodo hoja
}

rfrtuning=GradientBoostingRegressor()
grid_search2=GridSearchCV(rfrtuning, param_grid2, scoring="r2",cv=10, n_jobs=-1)
grid_result2=grid_search2.fit(xtrainf, dfy)

gbr_final=grid_result2.best_estimator_ 
joblib.dump(gbr_final, "salidas\\gbr_final.pkl") 


eval=cross_val_score(rfr_final,xtrainf,dfy,cv=20,scoring="r2")
eval2=cross_val_score(gbr_final,xtrainf,dfy,cv=20,scoring="r2")

np.mean(eval)
np.mean(eval2)


#Grafiquemos las predicciones
rfr_final = joblib.load('salidas/rfr_final.pkl')
ypredrfr=rfr_final.predict(xtrainf)
sns.set_theme(style="ticks")
dict={"y":dfy,"ypredrfr":ypredrfr}
df1=pd.DataFrame(dict)
df1=df1.stack().reset_index()
df1.drop(columns=["level_0"],inplace=True)
df1.columns=["tipo","valor"]
df1["tipo"]=df1["tipo"].apply(lambda x: "real"if x=="y" else "predicho")

sns.histplot(data=df1, x="valor", hue="tipo")

#Para el gradient boosting
gbr_final = joblib.load('salidas/gbr_final.pkl')
ypredgbr=gbr_final.predict(xtrainf)
sns.set_theme(style="ticks")
dict={"y":dfy,"ypredrfr":ypredgbr}
df2=pd.DataFrame(dict)
df2=df2.stack().reset_index()
df2.drop(columns=["level_0"],inplace=True)
df2.columns=["tipo","valor"]
df2["tipo"]=df2["tipo"].apply(lambda x: "real"if x=="y" else "predicho")

sns.histplot(data=df2, x="valor", hue="tipo")

#Entrenamiento con propuesta########################################
#En la tabla de entrenamiento no se incluye el valor del prestamo
"""
rfr_final = joblib.load('salidas/rfr_final.pkl')
ypredrfr=rfr_final.predict(xtrainf)
sns.set_theme(style="ticks")
dict={"y":dfy,"ypredrfr":ypredrfr}
df1=pd.DataFrame(dict)
df1["propuesta"]=df1.apply(lambda x: x["ypredrfr"]+funcion(x),axis=1)
df1=df1.stack().reset_index()
df1.drop(columns=["level_0"],inplace=True)
df1.columns=["tipo","valor"]
df1["tipo"]=df1["tipo"].apply(lambda x: "real"if x=="y" else "predicho")

sns.histplot(data=df1, x="valor", hue="tipo")
"""
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#Metricas comparacion

# Calcula el MSE
mse = mean_squared_error(dfy, ypredrfr)
# Calcula el RMSE
rmse = np.sqrt(mse)
# Calcula el MAE
mae = mean_absolute_error(dfy, ypredrfr)
# Calcula el R2
r2 = r2_score(dfy, ypredrfr)
lista1=[mse, rmse,mae,r2]

# Calcula el MSE
mse = mean_squared_error(dfy, ypredgbr)
# Calcula el RMSE
rmse = np.sqrt(mse)
# Calcula el MAE
mae = mean_absolute_error(dfy, ypredgbr)
# Calcula el R2
r2 = r2_score(dfy, ypredgbr)
lista2=[mse,rmse,mae,r2]

lista3= ["Mean Squared Error (MSE)","Root Mean Squared Error (RMSE)","Mean Absolute Error (MAE)","R-squared (R2)"]
dict={"metrica":lista3,"rfr":lista1,"gbr":lista2}
dffinal=pd.DataFrame(dict)



#######Predicciones para tabla de nuevos########
tabla_nuevos.info()
x=tabla_nuevos.iloc[:,:-1]
y=tabla_nuevos.iloc[:,-1]

#get_dummies
df_dum=pd.get_dummies(x,columns=cat)
df_dum.info()

#Escalamos las variables
scaler=StandardScaler()
scaler.fit(df_dum)
xnor=scaler.transform(df_dum)
xtest=pd.DataFrame(xnor,columns=df_dum.columns)
xtest.drop(columns=["ID"],inplace=True)

xtrainf = joblib.load('salidas/xtrainf.pkl')

plt.hist(y) #monto de prestamo
xtest=xtest.reindex(columns=xtrainf.columns)
ypredtestrfr=rfr_final.predict(xtest)

plt.hist(ypredtestrfr, bins=50)

dict = {"ID":tabla_nuevos["ID"], "int_prev":ypredtestrfr,"NewLoanApplication":y}
dfex=pd.DataFrame(dict)
#excel["int_rc"]=excel["int_rc"].apply(lambda x: x + 0.15 )
#excel.to_excel("Predicciones.xlsx",index=False)


import math

def funcion(x):
  return (math.exp(x["NewLoanApplication"]/(np.max(x["NewLoanApplication"])*100))-1)

dfex["int_rc"]=dfex.apply(lambda x: x["int_prev"]+funcion(x),axis=1)

plt.figure(figsize=(10, 6))
plt.hist(dfex['int_prev'], bins=50, color='blue', alpha=0.5, label='int_prev')
plt.hist(dfex['int_rc'], bins=50, color='red', alpha=0.5, label='int_rc')
plt.legend()
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Histograma de intereses')
plt.show()

#Generacion de tabla
excel=dfex[["ID","int_rc"]]
excel.to_csv("Predicciones.csv",index=False)
"""
excelf=pd.concat([excel, pd.DataFrame(y)], axis=1)

def funcion(x):
    if x<50000:
        return 0.005
    elif x<100000:
        return 0.01
    elif x<150000:
        return 0.015
    elif x<200000:
        return 0.02
    elif x<250000:
        return 0.025
    elif x<300000:
        return 0.03
    else:
        return 0.04
excelf["interes_final"]=excelf.apply(lambda x: x["int_rc"]+funcion(x["NewLoanApplication"]),axis=1)



plt.figure(figsize=(10, 6))
plt.hist(excelf['interes_final'], bins=50, color='blue', alpha=0.5, label='interes_final')
plt.hist(excelf['int_rc'], bins=50, color='red', alpha=0.5, label='int_rc')
plt.legend()
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Histograma de intereses')
plt.show()
"""




