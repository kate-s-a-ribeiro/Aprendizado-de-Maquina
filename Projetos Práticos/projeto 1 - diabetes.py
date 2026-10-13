import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score

# primeiro, lemos o .csv com as informações utilizando pandas
data = pd.read_csv('./py/Aprendizado de Máquina/Datasets/diabetes.csv', index_col=False, sep=',')


# agora, vamos substituir os valores com zero
# substituiremos pela mediana, pois a mediana é menos sensível a outliers
data["Glucose"] = data["Glucose"].replace(0, data.loc[data["Glucose"] != 0, "Glucose"].median())
data["BloodPressure"] = data["BloodPressure"].replace(0, data.loc[data["BloodPressure"] != 0, "BloodPressure"].median())
data["SkinThickness"] = data["SkinThickness"].replace(0, data.loc[data["SkinThickness"] != 0, "SkinThickness"].median())
data["Insulin"] = data["Insulin"].replace(0, data.loc[data["Insulin"] != 0, "Insulin"].median())
data["BMI"] = data["BMI"].replace(0, data.loc[data["BMI"] != 0, "BMI"].median())

# aqui, nós pegamos nesse mesmo csv algumas colunas
x = data.drop("Outcome", axis=1)   # aqui, nós pegamos todos os dados, menos o resultado (se a pessoa está doente ou não)
y = data['Outcome']                # e aqui nós pegamos somente o resultado

# daí agora vamos treinar
# vamos separar em 80% treino e 20% teste
x_treino, x_teste, y_treino, y_teste = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# agora a gente cria e treina o modelo, utilizando RandomForest
# n_estimators = número de árvores de decisão. 100 dá, 300 é melhor
# max_depth=None significa que as árvores podem ser mais profundas
# class_weight='balanced' significa que errar doente é mais grave, então ele vai errar menos doentes
modelo = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, class_weight='balanced')
modelo.fit(x_treino, y_treino)

# agora, a gente vai fazer as previsões
# primeiro de tudo, podemos usar o predict:
#previsoes = modelo.predict()
# de automático, ele tem um treshold de 0.5
# isso significa que se um paciente tiver 51% de chance de estar doente, então ele está doente
# se ele tiver 49% de chance de estar doente, então está saudável

# assim, podemos alterar esse treshold
# podemos alterá-lo, por exemplo, para 40% (0.4) ou 30% (0.3)
# isso significa que, acima de 30%, já será considerado doente

# a exemplo:
probabilidade = modelo.predict_proba(x_teste)[:,1]  # aqui estamos pegando somente a possibilidade do paciente estar doente
previsoes = (probabilidade >= 0.5).astype(int)      # e definimos que, se for acima de 0.3, vai ser 1 (doente); abaixo, será 0

# o que isso significa?
# isso significa que, com um treshold de 0.5, só vamos dizer que está doente se tivermos certeza absoluta
# com um treshold menor, vamos detectar muito mais doentes, mas também vamos dar muitos alarmes falsos pra pessoas saudáveis

print(f"Acurácia: {accuracy_score(y_teste, previsoes):.2f}")

print("\nRelatório de Classificação:")
print(classification_report(y_teste, previsoes))

# aqui nós vamos ver um score que independe do treshold
# se o valor for maior que 0.8, tá bom
# acima de 0.9, tá excelente
# e quanto mais próximo de 0.5, então mais próximo de um chute é
auc = roc_auc_score(y_teste, probabilidade)
print(f"AUC-ROC: {auc:.2f}")
