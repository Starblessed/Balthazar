import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

from scipy import stats
import statistics as stt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import PolynomialFeatures
baln = sns.diverging_palette(0, 180, as_cmap=True)
anti_baln = sns.diverging_palette(180, 0, as_cmap=True)
meses = "janeiro fevereiro março abril maio junho julho agosto setembro outubro novembro dezembro".split()
limite = 1000

class CustomModel(keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


# Adiciona a coluna "condição" ao dataframe, representando a balneabilidade da praia.
def add_propria(dataframe):
    dataframe["INEA"] = np.where(dataframe["Colimetria"] < limite, 1, -1)
    return dataframe


# Cria um sub-dataframe com colunas "Mes", "Semana", "Colimetria" e "INEA" para a praia selecionada.
def amostrar(praia):
    data = df[["Ano", "Mes", "Semana", praia]]
    data.rename(columns={praia:"Colimetria"}, inplace=True)
    add_propria(data)
    return data


# Define a função que gera a coluna "Semana" em um DataFrame.
def set_weeks(dataframe):
    week_list = []
    week_iter = 0
    for entry in dataframe["Data"]:
        if entry.day in [1, 2, 3, 4, 5, 6, 7]:
            week_iter = 1
        elif entry.day in [8, 9, 10, 11, 12, 13, 14]:
            week_iter = 2
        elif entry.day in [15, 16, 17, 18, 19, 20, 21]:
            week_iter = 3
        elif entry.day in [22, 23, 24, 25, 26, 27, 28]:
            week_iter = 4
        elif entry.day in [29, 30, 31]:
            week_iter = 5
        week_list.append(week_iter)
    return np.array(week_list)


# Define a função que gera um mapa de calor de colimetria da praia escolhida.
def coli_hm(praia, dataframe, ax):
    pivotted = dataframe.pivot_table(index="Mes", columns="Semana", values=praia.upper())
    pivotted.values[1][4] = np.nan
    hm = sns.heatmap(pivotted, cmap=anti_baln, vmin=0, vmax=limite,
                cbar_kws={'label': 'Qnt. Média de Col. Termotolerantes (NMP)'}, ax=ax)
    return hm

# Define dados beta em pré-processamento.
def beta_data_gen(praia):
    dataframe = df[["Mes", "Semana", praia.upper()]]
    return dataframe


# Faz o pré-processamento dos dados, preparando-os para alimentar o modelo em fase de treinamento e a posteriori.
def gerar_dados(praia):
    mes = praia["Mes"].tolist()
    semana = praia["Semana"].tolist()
    c_inea = praia["INEA"].tolist()
    feats = [mes, semana]
    feats = np.array(feats).transpose().tolist()
    return feats, c_inea


# Treina o modelo com os dados informados, na praia escolhida, e prevê a situação da mesma no mês e semana informados.
def train_model(ft, lbl, n_praia, pred_m, pred_s):
    X_train, X_test, y_train, y_test = tts(ft, lbl, test_size=0.2)

    inputs = keras.Input(shape=(2,))
    outputs = keras.layers.Dense(units=1, activation='sigmoid')(inputs)
    model = CustomModel(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                  loss="mse",
                  metrics=[keras.metrics.mean_squared_error])

    model.fit(X_train, y_train, epochs=100)
    model.evaluate(X_test, y_test)

    '''yhat = model.predict(np.array([[pred_m], [pred_s]]).transpose().tolist())
    raw_value = round(yhat.tolist()[0][0], 2)
    f_pred = round(raw_value)
    print(f'{yhat} - {raw_value} - {f_pred}')

    condicao = ''
    pa = 0
    match f_pred:
        case (-1):
            condicao = 'Imprópria para banho'
            pa = abs(raw_value)
        case 0:
            condicao = 'Imprópria para banho'
            pa = 1 - raw_value
        case 1:
            condicao = 'Própria para banho'
            pa = raw_value
    print('----------------------------------------')
    print(f'Situação da Praia {n_praia} na {pred_s}a semana de {meses[pred_m - 1]}: \n'
          f'Condição: {condicao}. \n'
          f'Precisão aproximada: {pa*100}%.')
    print('----------------------------------------')'''
    q_mes = []
    q_semana = []
    q_preds = []

    for month in range(1, 13):
        for week in range(1, 6):
            q_mes.append(month)
            q_semana.append(week)
            yhat = model.predict(np.array([[month], [semana]]).transpose().tolist())
            prediction = round(yhat.tolist()[0][0])
            print(prediction)
            q_preds.append(round(prediction))
    qmes = np.array(q_mes)
    qsemana = np.array(q_semana)
    qpreds = np.array(q_preds)
    pred_dict = {"Mes": qmes, "Semana": qsemana, "Predict": qpreds}
    pred_df = pd.DataFrame(pred_dict)
    pivoteado = pred_df.pivot_table(index="Mes", columns="Semana", values="Predict")
    figure, (ax1, ax2) = plt.subplots(1, 2)
    sns.heatmap(pivoteado, cmap=baln, vmin=0, vmax=1, ax=ax1)
    coli_hm(n_praia, df, ax2)
    plt.show()





    '''model = tf.keras.Sequential()
    model.add(keras.layers.Dense(units=2, input_shape=(2,), activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.3), loss='mse')

    model.fit(ft, lbl, epochs=100)'''
    # model.evaluate(X_test, y_test, batch_size=3)'''


# Importa os dados do arquivo .csv
df = pd.read_csv('dados_colimetria_zo_zs .csv')

# Converte as datas para o tipo 'datetime' e adiciona as colunas Ano, Mês e Semana
df["Data"] = pd.to_datetime(df["Data"])
df["Ano"] = df["Data"].dt.year
df["Mes"] = df["Data"].dt.month
df["Semana"] = set_weeks(df)

# Remove as linhas que possuem pelo menos 1 valor nulo e converte as colunas Ano, Mês, Semana e Colimetria para int.
df = df.dropna()
df = df.astype({'Ano': 'int', 'Mes': 'int', 'Semana': 'int', 'PN000': 'int'})


# Recorta do DataFrame somente as entradas do período com todas as praias dos boletins do INEA
df.drop(df[df["Ano"] < 2016].index, inplace=True)
df.drop(df[df["Ano"] == 2023].index, inplace=True)

# Atribui um dataframe amostrado na praia especificada a uma variável.
prainha = amostrar("PN000")
vermelha = amostrar("VR000")
botafogo0 = amostrar("BT000")
quebra_mar = amostrar("BD010")
flamengo4 = amostrar("FL004")
flamengo0 = amostrar("FL000")
urca = amostrar("UR000")
gavea = amostrar("GV001")
pepino = amostrar("PP010")
pontal0 = amostrar("PS000")
leblon = amostrar("LB000")
ipanema = amostrar("IP003")
pepe = amostrar("BD009")
leme = amostrar("LM002")
# Define os dados de treinamento e previsão, posteriormente treinando o modelo e executando uma previsão.
mes = 3
semana = 2
features, label = gerar_dados(leme)
train_model(features, label, "LM002", mes, semana)

