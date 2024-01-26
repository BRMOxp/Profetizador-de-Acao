import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly

# Carregando os dados
ticker = input("Digite o código da ação desejada: ")
dados = yf.Ticker(ticker).history("2y")

# Preparando os dados para o treinamento
treinamento = dados.reset_index()
treinamento["Date"] = treinamento["Date"].dt.tz_localize(None)
treinamento = treinamento[['Date', 'Close']]
treinamento.columns = ['ds', 'y']

# Treinamento do modelo
modelo = Prophet()
modelo.fit(treinamento)
periodo = modelo.make_future_dataframe(periods=90)
previsoes = modelo.predict(periodo)

# Gerando o gráfico de previsões
plot_plotly(modelo, previsoes, xlabel = "período", ylabel="valor").show()