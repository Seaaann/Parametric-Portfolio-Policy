import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as st
import sympy as sp
import datetime
import scipy as sp

sns.set_style("whitegrid")

df = pd.read_csv(
    "/Users/sean/Desktop/Github/Parametric-Portfolio-Policy/price.csv"
).set_index("Date")
print(df)
df_price = pd.pivot_table(df, values="adjusted", index=df.index, columns=df["symbol"])
mktcap = pd.pivot_table(df, values="mktcap", index=df.index, columns=df["symbol"])

have_null = df_price.columns[df_price.isna().any()]
have_null.append(df_price.columns[df_price.isna().any()])
df_price = df_price.drop(columns=have_null)
ret = np.log(df_price).diff()
ret = ret.iloc[1::, :]

have_null = mktcap.columns[mktcap.isna().any()]
have_null.append(mktcap.columns[mktcap.isna().any()])
mktcap = mktcap.drop(columns=have_null)
mktcap = mktcap.iloc[1::, :]

m12 = ret.rolling(12).sum()
m12 = m12.iloc[11::, :]
ret = ret.iloc[12::, :]
mktcap = mktcap.iloc[11::, :]

ret_rowmean = ret.mean(axis=1)
mktcap_rowmean = mktcap.mean(axis=1)
m12_rowmean = m12.mean(axis=1)
ret_std = ret.std(axis=1)
mkt_std = mktcap.std(axis=1)
m12_std = m12.std(axis=1)
d = {
    "Mean(ret)": ret_rowmean,
    "Mean(mktcap)": mktcap_rowmean,
    "Mean(m12)": m12_rowmean,
    "Sd(ret)": ret_std,
    "Sd(mktcap)": mkt_std,
    "Sd(m12_std)": m12_std,
}
DF = pd.DataFrame(d)
print(DF.iloc[1::, :])

fig = plt.figure(figsize=(12, 8))
DF = DF.reset_index()
for i in range(1, 7):
    plt.subplot(2, 3, i)
    DF.iloc[:, i].plot()
plt.show()

fig = plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(ret.iloc[:, 0:10])
plt.ylabel("monthly returns")
plt.subplot(2, 2, 2)
plt.plot(m12.iloc[:, 0:10])
plt.ylabel("12 month returns")
plt.subplot(2, 2, 3)
plt.plot(mktcap.iloc[:, 0:10])
plt.ylabel("market captalization")
plt.show()


def Scale(y, c=True, sc=True):

    """
    If ‘scale’ is
    ‘TRUE’ then scaling is done by dividing the (centered) columns of
    ‘x’ by their standard deviations if ‘center’ is ‘TRUE’, and the
    root mean square otherwise.  If ‘scale’ is ‘FALSE’, no scaling is
    done.

    The root-mean-square for a (possibly centered) column is defined
    as sqrt(sum(x^2)/(n-1)), where x is a vector of the non-missing
    values and n is the number of non-missing values.  In the case
    ‘center = TRUE’, this is the same as the standard deviation, but
    in general it is not.
    """
    x = y.copy()

    if c:
        x -= x.mean()
    if sc and c:
        x /= x.std()
    elif sc:
        x /= np.sqrt(x.pow(2).sum().div(x.count() - 1))
    return x


scaled_mktcap = pd.DataFrame(Scale(mktcap.T))
scaled_m12 = pd.DataFrame(Scale(m12.T))
scaled_mktcap = scaled_mktcap.T
scaled_mktcap = scaled_mktcap.iloc[0:109, :]
scaled_m12 = scaled_m12.T
scaled_m12 = scaled_m12.iloc[0:109, :]


def PPS(x, wb, nt, ret, m12, mktcap, rr):
    wi = wb + nt * (x[0] * m12 + x[1] * mktcap)
    wret = (wi * ret).sum(axis=1)
    ut = ((1 + wret) ** (1 - rr)) / (1 - rr)
    u = -(ut.mean())
    return u


Scaled_m12 = scaled_m12.reset_index()
Scaled_m12 = Scaled_m12.drop("Date", axis=1)
Scaled_mktcap = scaled_mktcap.reset_index()
Scaled_mktcap = Scaled_mktcap.drop("Date", axis=1)
Ret = ret.reset_index()
Ret = Ret.drop("Date", axis=1)

nt = wb = 1 / np.shape(ret)[1]
rr = 5

res_save = []
weights = []
x0 = np.array([0, 0])
for i in range(0, 60):
    opt = sp.optimize.minimize(
        PPS,
        x0,
        method="BFGS",
        args=(
            wb,
            nt,
            Ret.iloc[0 : 48 + i, :],
            Scaled_m12.iloc[0 : 48 + i, :],
            Scaled_mktcap.iloc[0 : 48 + i, :],
            rr,
        ),
    )
    print("The {} window".format(i + 1))
    print("The value:", opt["x"])
    res_save.append(opt["x"])
    w = wb + nt * (
        opt["x"][0] * Scaled_m12.iloc[i + 48, :]
        + opt["x"][1] * Scaled_mktcap.iloc[i + 48, :]
    )
    print(w)
    weights.append(w)

index = ret.index[49:110]
char_df = pd.DataFrame(res_save, index=index, columns=["m12", "mktcap"])

fig = plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
char_df["m12"].plot()
plt.title("m12")
plt.ylabel("Parameter Value")
plt.subplot(2, 1, 2)
char_df["mktcap"].plot()
plt.title("mktcap")
plt.ylabel("Parameter Value")
plt.show()

weights = pd.DataFrame(weights)
ret_fit = (weights * Ret.tail(60)).sum(axis=1)
ret_EW = (nt * Ret.tail(60)).sum(axis=1)
acc_fit = ret_fit.cumsum()
acc_fitvalue = acc_fit.values
acc_fitvalue = acc_fitvalue[1::]
acc_EW = ret_EW.cumsum()
acc_EWvalue = acc_EW.values

acc_ret = ret.tail(60).cumsum()
top100 = (np.argsort(-acc_ret.tail(1))).iloc[0, :].values[0:100]

acc_top100 = acc_ret.iloc[:, top100].mean(axis=1)
acc_top100value = acc_top100.values

acc_df = pd.DataFrame(index=index)
acc_df["opt"] = acc_fitvalue
acc_df["EW"] = acc_EWvalue
acc_df["top100"] = acc_top100value
acc_df.plot(figsize=(12, 8))
plt.ylabel("Cumulative Return")
plt.show()