import pandas as pd
import json
from   pandas import json_normalize
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import unidecode 
import unicodedata

base = pd.read_csv('/content/PENSE2019_MICRODADOS.csv')
base.head(10)

base.rename(columns={'DEP_ADMIN':'TIPO_ESCOLA','B01001A':'SEXO','B01003':'IDADE','B01002':'ETNIA','B01021A':'ANO_ESCOLAR'}, inplace = True)
base.rename(columns={'B01006':'MORA_MAE','B01007':'MORA_PAI','B04001':'JA_FUMOU','B04013':'JA_NARGUILE','B04014':'JA_CIGARRO_ELETRONICO'}, inplace = True)
base.rename(columns={'B04015':'JA_USOU_TABACO','B04006B':'RESPONSAVEL_FUMA','B04005A':'FUMA_PROXIMO_ALUNO','B04016':'AMIGO_FUMA'}, inplace = True)

filtro_regiao = base['REGIAO'] == 3
filtro_municipio = base['MUNICIPIO_CAP'] != 0
base = base[filtro_regiao]
base = base[filtro_municipio]
base.head(10)

np.unique(base['JA_FUMOU'], return_counts=True)

base.drop(base[base['JA_FUMOU'] == -2.0].index, axis=0, inplace=True)
base.drop(base[base['JA_FUMOU'] == -1.0].index, axis=0, inplace=True)
base.drop(base[base['JA_FUMOU'] == 9.0].index, axis=0, inplace=True)

np.unique(base['JA_FUMOU'], return_counts=True)

base=base[['MUNICIPIO_CAP','TIPO_ESCOLA','SEXO','IDADE','ETNIA','ANO_ESCOLAR','MORA_MAE','MORA_PAI','JA_FUMOU','JA_NARGUILE','JA_CIGARRO_ELETRONICO','JA_USOU_TABACO','RESPONSAVEL_FUMA','FUMA_PROXIMO_ALUNO','AMIGO_FUMA']]
base.info()

def ajuste_municipio(valor):
  if valor == 3106200:
    return "BELO HORIZONTE"
  elif valor == 3205309:
    return "VITORIA"
  elif valor == 3304557:
    return "RIO DE JANEIRO"
  elif valor == 3550308:
    return "SAO PAULO"
    
def ajuste_escola(valor):
  if valor == 1:
    return "Publica"
  else:
    return "Privada"
    
def ajuste_sexo(valor):
  if valor == 1:
    return "Homem"
  elif valor == 2:
    return "Mulher"
  else:
    return "Sem Resposta"

def ajuste_idade(valor):
  if valor == 1:
    return "Menos de 13 anos"
  elif valor == 2:
    return "13 a 15 anos"
  elif valor == 3:
    return "16 a 17 anos"
  elif valor == 4:
    return "18 anos ou mais"
  else:
    return "Sem Resposta"

def ajuste_raca(valor):
  if valor == 1:
    return "Branca"
  elif valor == 2:
    return "Preta"
  elif valor == 3:
    return "Amarela"
  elif valor == 4:
    return "Parda"
  elif valor == 5:
    return "Indigena"
  else:
    return "Sem Resposta"

def ajuste_ensino_medio(valor):
  if valor == 1:
    return "6º ano"
  elif valor == 2:
    return "7º ano"
  elif valor == 3:
    return "8º ano"
  elif valor == 4:
    return "9º ano"
  elif valor == 5:
    return "1º ano ensino medio"
  elif valor == 6:
    return "2º ano ensino medio"
  elif valor == 7:
    return "3º ano ensino medio"
  else:
    return "Sem Resposta"

def ajuste_mora_mae(valor):
  if valor == 1:
    return "sim"
  else:
    return "não"

def ajuste_mora_pai(valor):
  if valor == 1:
    return "sim"
  else:
    return "não"

def ajuste_narguile(valor):
  if valor == 1:
    return "sim"
  elif valor == 2:
    return "não"
  else:
    return "Sem Resposta"

def ajuste_cigarro_eletronico(valor):
  if valor == 1:
    return "sim"
  elif valor == 2:
    return "não"
  else:
    return "Sem Resposta"

def ajuste_consumiu_tabaco(valor):
  if valor == 1:
    return "sim"
  elif valor == 2:
    return "não"
  else:
    return "Sem Resposta"

def ajuste_responsavel_fuma(valor):
  if valor == 1:
    return "Nenhum deles"
  elif valor == 2:
    return "Só meu pai ou responsável do sexo masculino"
  elif valor == 3:
    return "Só minha mãe ou responsável do sexo feminino"
  elif valor == 4:
    return "Os dois (ambos)"
  elif valor == 5:
    return "Não sei"
  else:
    return "Sem Resposta"

def ajuste_fumar_perto_7dias(valor):
  if valor == 1:
    return "Nenhum dia nos últimos 7 dias"
  elif valor == 2:
    return "1 ou 2 dias"
  elif valor == 3:
    return "3 ou 4 dias"
  elif valor == 4:
    return "5 ou 6 dias"
  elif valor == 5:
    return "Todos os dias"
  else:
    return "Sem Resposta"

def ajuste_amigo_fuma(valor):
  if valor == 1:
    return "sim"
  elif valor == 2:
    return "não"
  else:
    return "Sem Resposta"

base['MUNICIPIO_CAP'] = base['MUNICIPIO_CAP'].apply(lambda x : ajuste_municipio(x))
base['TIPO_ESCOLA'] = base['TIPO_ESCOLA'].apply(lambda x : ajuste_escola(x))
base['SEXO'] = base['SEXO'].apply(lambda x : ajuste_sexo(x))
base['IDADE'] = base['IDADE'].apply(lambda x : ajuste_idade(x))
base['ETNIA'] = base['ETNIA'].apply(lambda x : ajuste_raca(x))
base['ANO_ESCOLAR'] = base['ANO_ESCOLAR'].apply(lambda x : ajuste_ensino_medio(x))
base['MORA_MAE'] = base['MORA_MAE'].apply(lambda x : ajuste_mora_mae(x))
base['MORA_PAI'] = base['MORA_PAI'].apply(lambda x : ajuste_mora_pai(x))
base['JA_NARGUILE'] = base['JA_NARGUILE'].apply(lambda x : ajuste_narguile(x))
base['JA_CIGARRO_ELETRONICO'] = base['JA_CIGARRO_ELETRONICO'].apply(lambda x : ajuste_cigarro_eletronico(x))
base['RESPONSAVEL_FUMA'] = base['RESPONSAVEL_FUMA'].apply(lambda x : ajuste_responsavel_fuma(x))
base['FUMA_PROXIMO_ALUNO'] = base['FUMA_PROXIMO_ALUNO'].apply(lambda x : ajuste_fumar_perto_7dias(x))
base['JA_USOU_TABACO'] = base['JA_USOU_TABACO'].apply(lambda x : ajuste_consumiu_tabaco(x))
base['AMIGO_FUMA'] = base['AMIGO_FUMA'].apply(lambda x : ajuste_amigo_fuma(x))
base.head(10)

base.drop(base[base['SEXO'] == "Sem Resposta"].index, axis=0, inplace=True)
base.drop(base[base['IDADE'] == "Sem Resposta"].index, axis=0, inplace=True)
base.drop(base[base['ETNIA'] == "Sem Resposta"].index, axis=0, inplace=True)
base.drop(base[base['ANO_ESCOLAR'] == "Sem Resposta"].index, axis=0, inplace=True)
base.drop(base[base['JA_NARGUILE'] == "Sem Resposta"].index, axis=0, inplace=True)
base.drop(base[base['JA_CIGARRO_ELETRONICO'] == "Sem Resposta"].index, axis=0, inplace=True)
base.drop(base[base['JA_USOU_TABACO'] == "Sem Resposta"].index, axis=0, inplace=True)
base.drop(base[base['RESPONSAVEL_FUMA'] == "Sem Resposta"].index, axis=0, inplace=True)
base.drop(base[base['FUMA_PROXIMO_ALUNO'] == "Sem Resposta"].index, axis=0, inplace=True)
base.drop(base[base['AMIGO_FUMA'] == "Sem Resposta"].index, axis=0, inplace=True)
base_IDH = pd.read_csv('/content/base_IDH_2010.csv')

def padronizacao_acentuacao(valor):
    return unidecode.unidecode(valor)
base_IDH['Município'] = base_IDH['Município'].apply(lambda x : padronizacao_acentuacao(x))
base_IDH['Município']= base_IDH['Município'].str.upper()
base_IDH.head(10)
base_IDH[['CIDADE', 'ESTADO']] = base_IDH['Município'].str.split('(', expand=True, n=1)
base_IDH.head(10)
base_IDH = base_IDH.drop(columns=['Ranking IDHM 2010','Município','IDHM 2010','IDHM\nRenda\n2010','IDHM Longevidade 2010','ESTADO'])

def enriquecimento_IDH(valor):
  a = str(valor).strip()
  for i in base_IDH.index:
    b = str(base_IDH['CIDADE'][i]).strip()
    if a == b:
      return base_IDH['IDHM Educação 2010'][i]

base['IDH_EDUCACAO'] = base['MUNICIPIO_CAP'].apply(lambda x : enriquecimento_IDH(x))
base.head(10)
base['IDH_EDUCACAO'] = base['IDH_EDUCACAO'].str.replace(',','.').astype(float)
base.to_csv("/content/base_tabagismo_IDH.csv", index=False)

df = pd.read_csv('/content/base_tabagismo_IDH.csv')
df.head(10)

def ajuste_ja_fumou(valor):
  if valor == 1:
    return "sim"
  else:
    return "não"

df['JA_FUMOU'] = df['JA_FUMOU'].apply(lambda x : ajuste_ja_fumou(x))
df.isna().sum()
df.info()
sns.boxplot(x=df['IDH_EDUCACAO'], palette="OrRd");

def countplot_format(classe, eixox, eixoy, titulo):
  grafico = plt.subplots(figsize=(15, 13))
  grafico = sns.countplot(x=df[classe],palette = 'OrRd_r')
  grafico.set_title(f'{titulo}\n', fontsize=30); 
  grafico.set_xlabel(eixox, fontsize=10); 
  grafico.set_ylabel(eixoy, fontsize=10);
  return grafico

situacao_ja_fumou = countplot_format('JA_FUMOU', 'x', 'Quantidade', 'Quantidade de alunos que já fumaram')
plt.tick_params(left = False, right = False , labelleft = True ,labelbottom = True, bottom = False)
plt.xticks(rotation=45)
plt.show()
df['JA_FUMOU'].describe()

idade = countplot_format('IDADE', 'x', 'Quantidade', 'Idade predominante')
plt.tick_params(left = False, right = False , labelleft = True ,labelbottom = True, bottom = False)
plt.xticks(rotation=45)
plt.show()

sexo = countplot_format('SEXO', 'x', 'Quantidade', 'Sexo predominante')
plt.tick_params(left = False, right = False , labelleft = True ,labelbottom = True, bottom = False)
plt.xticks(rotation=45)
plt.show()

etnia = countplot_format('ETNIA', 'x', 'Quantidade', 'Etnia dos alunos')
plt.tick_params(left = False, right = False , labelleft = True ,labelbottom = True, bottom = False)
plt.xticks(rotation=45)
plt.show()

anoEscolar = countplot_format('ANO_ESCOLAR', 'x', 'Quantidade', 'Ano escolar predominante')
plt.tick_params(left = False, right = False , labelleft = True ,labelbottom = True, bottom = False)
plt.xticks(rotation=45)
plt.show()

respFumante = countplot_format('RESPONSAVEL_FUMA', 'x', 'Quantidade', 'Incidência de fumantes entre os responsáveis pelos alunos')
plt.tick_params(left = False, right = False , labelleft = True ,labelbottom = True, bottom = False)
plt.xticks(rotation=45)
plt.show()

amigo = countplot_format('AMIGO_FUMA', 'x', 'Quantidade', 'Incidência de amigos fumantes')
plt.tick_params(left = False, right = False , labelleft = True ,labelbottom = True, bottom = False)
plt.xticks(rotation=45)
plt.show()

def mapaDeCalor(principal, comparativo, eixoX, eixoY, titulo):
  bd = df.loc[~df[principal].isin(['NAO SE APLICA', 'NS/NR']),[principal, comparativo]]# "~" para negar
  info_adequacao = pd.pivot_table(bd, index=principal, columns=comparativo, aggfunc=len, fill_value=0)
  grafico = sns.heatmap(info_adequacao, cmap="OrRd", annot=True, fmt ='d', linewidths=.3, xticklabels=True, yticklabels=True)
  grafico.set_xlabel(eixoX, fontsize=20)
  grafico.set_ylabel(eixoY, fontsize=20)
  grafico.set_title(f'{titulo}\n', fontsize=20)

  plt.tick_params(left = False, right = False , labelleft = True ,labelbottom = True, bottom = False)
  plt.xticks(rotation=45)
  plt.gcf().set_size_inches(15, 10)
  return grafico

jaFumou_Municipio = mapaDeCalor('MUNICIPIO_CAP','JA_FUMOU', 'Ja Fumou', 'Municipio', "Alunos fumantes por municipio analisado")
plt.show()

jaFumou_escola = mapaDeCalor('TIPO_ESCOLA', 'JA_FUMOU', 'Ja Fumou', 'Tipo de escola', "Relação do tipo de escola para alunos que já fumaram")
plt.show()

jaFumou_sexo = mapaDeCalor('SEXO', 'JA_FUMOU', 'Ja Fumou', 'Sexo', "Relação entre sexo e alunos fumantes")
plt.show()

jaFumou_cor = mapaDeCalor('ETNIA', 'JA_FUMOU', 'Ja fumou', 'Etnia', "Relação entre alunos fumantes e etnia declarada pelos estudantes")
plt.show()

anoEscolar_jaFumou = mapaDeCalor('ANO_ESCOLAR', 'JA_FUMOU', 'Ja fumou', 'Série', "Relação entre a série e alunos que mais fumam")
plt.show()

narguile = mapaDeCalor('JA_NARGUILE', 'JA_FUMOU', 'Ja fumou', 'Narguile', "Relação de alunos que já utilizaram Narguile com Já fumaram cigarros")
plt.show()

cEletr = mapaDeCalor('JA_CIGARRO_ELETRONICO', 'JA_FUMOU', 'Ja fumou', 'CIGARRO ELETRÔNICO', "Relação de alunos que já utilizaram cigarro eletrônico com já fumaram cigarros")
plt.show()

ResponsavelFuma = mapaDeCalor('RESPONSAVEL_FUMA', 'JA_FUMOU', 'Ja fumou', 'Responsável Fuma', "Responsável fuma com relação a alunos que já fumaram")
plt.show()

amigo = mapaDeCalor('AMIGO_FUMA', 'JA_FUMOU', 'Ja fumou', 'Amigo Fuma', "Amigo Fuma em relação a alunos que já fumaram ")
plt.show()

idh = mapaDeCalor('IDH_EDUCACAO', 'JA_FUMOU', 'Ja fumou', 'IDH', "IDH em relação a quantidade de alunos que já experimentaram cigarros")
plt.show()

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from yellowbrick.classifier import ConfusionMatrix
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold

base_ML = pd.read_csv('/content/base_tabagismo_IDH.csv')
base_ML = base_ML.drop(columns=['MUNICIPIO_CAP'])
base_ML = pd.get_dummies(base_ML)
base_ML.head(10)

X = base_ML.drop('JA_FUMOU', axis=1)
y = base_ML['JA_FUMOU']
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X

ax = sns.countplot(x=y,palette='OrRd_r')
nr= NearMiss()
X,y = nr.fit_resample(X,y)
ax = sns.countplot(x=y,palette='OrRd_r')
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size = 0.15, random_state = 0)
X_treinamento.shape, y_treinamento.shape, X_teste.shape, y_teste.shape, X.shape, y.shape
X_df = np.concatenate((X_treinamento, X_teste), axis = 0)
X_df.shape
Y_df = np.concatenate((y_treinamento, y_teste), axis = 0)
Y_df.shape

paramentros_RF = {'criterion':['gini','entropy'],
                      'n_estimators':[10,40,100,150],
                      'min_samples_split':[2,5,10],
                      'min_samples_leaf':[1,5,10]}
gridRN = GridSearchCV(estimator = RandomForestClassifier(),param_grid=paramentros_RF)
gridRN.fit(X_df,Y_df)
melhoresParametros = gridRN.best_params_
melhorResultado = gridRN.best_score_
print(melhoresParametros)
print(melhorResultado)

paramentros_DT = {'criterion':['gini','entropy'],
                      'splitter':['best','random'],
                      'min_samples_split':[2,5,10],
                      'min_samples_leaf':[1,5,10]}
gridDT = GridSearchCV(estimator = DecisionTreeClassifier(),param_grid=paramentros_DT)
gridDT.fit(X_df,Y_df)
melhoresParametros = gridDT.best_params_
melhorResultado = gridDT.best_score_
print(melhoresParametros)
print(melhorResultado)

paramentros_RN = {'activation': ['relu','logistic','tahn'],
                  'solver': ['adam','sgd'],
                  'batch_size':[10,50]}
gridRN= GridSearchCV(estimator = MLPClassifier(),param_grid=paramentros_RN)
gridRN.fit(X_df, Y_df)
melhoresParametros = gridRN.best_params_
melhorResultado = gridRN.best_score_
print(melhoresParametros)
print(melhorResultado)


random_forest = RandomForestClassifier(criterion='entropy', min_samples_leaf= 5, min_samples_split= 2, n_estimators= 100)
random_forest.fit(X_treinamento, y_treinamento)
previsoes_RF = random_forest.predict(X_teste)
previsoes_RF
accuracy_score(y_teste, previsoes_RF)
print(classification_report(y_teste, previsoes_RF))
cm = ConfusionMatrix(random_forest)
cm.fit(X_treinamento, y_treinamento)
cm.score(X_teste, y_teste)


arvore_decisao = tree.DecisionTreeClassifier(criterion= 'gini', min_samples_leaf= 10, min_samples_split= 5, splitter= 'best')
arvore_decisao.fit(X, y)
arvore_decisao.score(X, y)
previsao_Tree = arvore_decisao.predict(X_teste)
previsao_Tree
accuracy_score(y_teste, previsao_Tree)
print(classification_report(y_teste, previsao_Tree))
cm = ConfusionMatrix(arvore_decisao)
cm.fit(X_treinamento, y_treinamento)
cm.score(X_teste, y_teste)


rede_neural = MLPClassifier(activation= 'logistic', batch_size= 50, solver= 'adam')
rede_neural.fit(X_treinamento, y_treinamento)
previsoes_RN = rede_neural.predict(X_teste)
previsoes_RN
accuracy_score(y_teste, previsoes_RN)
print(classification_report(y_teste, previsoes_RN))
cm = ConfusionMatrix(rede_neural)
cm.fit(X_treinamento, y_treinamento)
cm.score(X_teste, y_teste)

resultadosVC_RF =[]
for i in range(30):
  kfold = KFold(n_splits=10 , shuffle=True, random_state=i)
  RF = RandomForestClassifier(criterion='entropy', min_samples_leaf= 5, min_samples_split= 2, n_estimators= 100)
  scores = cross_val_score(RF,X_df, Y_df, cv= kfold )
  resultadosVC_RF.append(scores.mean())
resultadosVC_RF

resultadosVC_arvore =[]
for i in range(30):
  kfold = KFold(n_splits=10 , shuffle=True, random_state=i)
  DT = DecisionTreeClassifier(criterion= 'gini', min_samples_leaf= 10, min_samples_split= 5, splitter= 'best')
  scores = cross_val_score(DT,X_df, Y_df, cv= kfold )
  resultadosVC_arvore.append(scores.mean())
resultadosVC_arvore

resultadosVC_RN =[]
for i in range(30):
  kfold = KFold(n_splits=10 , shuffle=True, random_state=i)
  RN = MLPClassifier(activation= 'logistic', batch_size= 50, solver= 'adam')
  scores = cross_val_score(RN,X_df, Y_df, cv= kfold )
  resultadosVC_RN.append(scores.mean())
resultadosVC_RN

resultados = pd.DataFrame({'Random_Forest':resultadosVC_RF, 'Arvore':resultadosVC_arvore, 'Rede_Neural':resultadosVC_RN})
resultados.to_csv('/content/resultados_cross_validation.csv')
resultados.head(10)
resultados.describe()
(resultados.std()/resultados.mean())*100