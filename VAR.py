import pandas as pd
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import datetime
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Валовое накопление основного капитала GFCF
# Индекс потребительских цен CPI
# Коэффициент безработицы
symbols = ["RUSGFCFQDSMEI", "RUSCPIALLMINMEI", "LMUNRRTTRUM156S"]

start = datetime.datetime(2003, 12, 1)
end = datetime.datetime(2024, 12, 1)
df = web.DataReader(symbols, "fred", start, end)
print(df)

# Переведем данные в квартальные
df000 = df.resample("Q", convention="start").last().dropna()
df000.plot(subplots=True, layout=(3, 1), figsize=(14, 10), sharex=False, sharey=False)
plt.show()

# логарифмируем данные и вычисляем разности
# VAR предполагает, что входной временной ряд стационарный
# ряд переводится в стационарный путем вычисления разностей. В данном случае разности первого порядка, скорее всего, будут достаточными, поскольку тренд постоянный
data = np.log(df000).diff().dropna()
# анализируем структуру ряда динамики на основе автокорреляций и частных автокорреляций
fig, axes = plt.subplots(nrows=3, ncols=3, sharex=False)
for i in range(len(symbols)):
    plot_acf(df000.values[:, i], ax=axes[i, i], label=symbols[i])
    axes[i, i].set_title(symbols[i])

axes[1, 0].xcorr(df000.values[:, 1], df000.values[:, 0])
axes[2, 0].xcorr(df000.values[:, 2], df000.values[:, 0])
axes[2, 1].xcorr(df000.values[:, 2], df000.values[:, 1])
axes[0, 1].xcorr(df000.values[:, 0], df000.values[:, 1])
axes[0, 2].xcorr(df000.values[:, 0], df000.values[:, 2])
axes[1, 2].xcorr(df000.values[:, 1], df000.values[:, 2])
plt.show()

# Модель VAR с порядком 2
model = VAR(data)
results = model.fit(2)
results.summary()

# строим график автокорреляций в остатках
results.plot_acorr()
plt.show()

# прогноз на 10 кварталов
lag_order = results.k_ar
results.forecast(data.values[-lag_order:], 10)
results.plot_forecast(10)
plt.show()

# Функции импульсного ответа

from statsmodels.tsa.vector_ar.irf import IRAnalysis

irf = results.irf()

irf.plot(orth=True)
plt.show()

irf.plot(impulse="RUSGFCFQDSMEI", orth=True)
plt.show()

irf.plot(impulse="RUSCPIALLMINMEI", orth=True)
plt.show()

irf.plot(impulse="LMUNRRTTRUM156S", orth=True)
plt.show()
