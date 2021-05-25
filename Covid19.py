

# IMPORT ALL THE REQUIRED LIBRARY
from fbprophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium import plugins
from sklearn.metrics import mean_squared_error

# DEFINE THE DATASETS
df_India = pd.read_csv(
    r'S:\Data Science\project_No_1\project_No_1\covid_19_india.csv')
India_coord = pd.read_excel(
    r'S:\Data Science\project_No_1\project_No_1\Indian Coordinates.xlsx')

# PRINT THE DATASETS
print(df_India.info())
print(df_India.head())
print(df_India.tail())
print(df_India.dtypes)
print(India_coord.info())
print(India_coord.head())

# TAKE CARE OF MISSING DATA


def replace_dash_with_zeros(inp):
    return int(inp.replace("-", "0"))


df_India.drop(['Sno'], axis=1, inplace=True)
df_India['Date'] = pd.to_datetime(df_India['Date'], format="%d/%m/%y")
df_India['ConfirmedIndianNational'] = df_India['ConfirmedIndianNational'].apply(
    replace_dash_with_zeros)
df_India['ConfirmedForeignNational'] = df_India['ConfirmedForeignNational'].apply(
    replace_dash_with_zeros)
df_India.sort_values("Confirmed", ascending=False, inplace=True)
print(df_India)

df_India.loc[df_India["ConfirmedForeignNational"] == "-", :]
list(zip(df_India.columns, df_India.dtypes, df_India.isna().sum()))
print(
    f'We have data available from : {df_India.Date.min()} to {df_India.Date.max()}')
df_India.groupby(["State/UnionTerritory", "Date"]).sum()
States = df_India['State/UnionTerritory'].unique().tolist()
print(States)
States.remove("Cases being reassigned to states")
States.remove("Unassigned")
print(States)
len(States)

# Merging Data Frames
df_final_India = pd.DataFrame()
dates = pd.DataFrame({"Date": pd.date_range(
    df_India.Date.min(), df_India.Date.max())})
for state in States:
    all_dates_df = pd.merge(dates,
                            df_India.loc[df_India['State/UnionTerritory']
                                         == state, :], on="Date",
                            how="left")
    all_dates_df['State/UnionTerritory'] = state
    all_dates_df = all_dates_df.fillna(0)
    all_dates_df['New Cases'] = all_dates_df['Confirmed'] - \
        all_dates_df['Confirmed'].shift(1)
#     print(state)
#     display(all_dates_df.loc[all_dates_df['New Cases'] <  0,:])
    df_final_India = pd.concat([df_final_India, all_dates_df], axis=0)
print("Finally we have a data of Size: ", df_final_India.shape)
print(df_final_India.head())

df_final_India.dropna(inplace=True)
print(df_final_India.shape)
del df_final_India['Time']
del df_final_India['ConfirmedIndianNational']
del df_final_India['ConfirmedForeignNational']
print(df_final_India)
df_final_India.groupby(["State/UnionTerritory", "Date"]).sum()
df_final_India = df_final_India.groupby(
    ["State/UnionTerritory", "Date"]).sum().reset_index()
print(df_final_India)

# Statewise Covid19 Status in India


def plot_pie(active, cured, death, title):
    labels = ['Active', 'Recovered', 'Died']
    sizes = [active, cured, death]
    color = ['#66b3ff', 'green', 'red']
    explode = []

    for i in labels:
        explode.append(0.05)

    plt.pie(sizes, labels=labels, autopct='%1.1f%%',
            startangle=9, explode=explode, colors=color)
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')

    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.title(title + 'COVID-19 Cases', fontsize=20)
    plt.axis('equal')
    plt.tight_layout()


total_cases_india = 0
cured_cases_india = 0
death_cases_india = 0
active_cases_india = 0
state_df = pd.DataFrame()

for state in States:
    one_state_df = df_final_India.loc[df_final_India['State/UnionTerritory'] == state, :]
    state_df = pd.concat(
        [state_df, pd.DataFrame(one_state_df.iloc[-1, :]).T], axis=0)
    total_cases = one_state_df['Confirmed'].values[-1]
    cured = one_state_df['Cured'].values[-1]
    deaths = one_state_df['Deaths'].values[-1]
    active = total_cases - cured - deaths
    plot_pie(active, cured, deaths, state)
    total_cases_india += total_cases
    cured_cases_india += cured
    death_cases_india += deaths
    active_cases_india += active
state_df.reset_index(inplace=True, drop=True)
print(state_df)
f, ax = plt.subplots(figsize=(12, 28))
data = state_df[['State/UnionTerritory', 'Confirmed', 'Cured', 'Deaths']]
data.sort_values('Confirmed', ascending=False, inplace=True)
sns.set_color_codes("pastel")
sns.barplot(x="Confirmed", y="State/UnionTerritory",
            data=data, label="Total", color="red")
sns.set_color_codes("muted")
sns.barplot(x="Cured", y="State/UnionTerritory",
            data=data, label="Cured", color="green")
ax.legend(ncol=5, loc="lower right", frameon=True)
ax.set(ylabel="", xlabel="Cases")
i = 0
for p in ax.patches:
    x = p.get_x() + p.get_width() + 3
    y = p.get_y() + p.get_height()/2
    if i <= len(States):
        ax.annotate(" "*10 + str(int(p.get_width())), (x, y))
    else:
        ax.annotate(int(p.get_width()), (x, y))

    i += 1

    f, ax = plt.subplots(figsize=(12, 28))
data = state_df[['State/UnionTerritory', 'Confirmed', 'Cured', 'Deaths']]
data.sort_values('Confirmed', ascending=False, inplace=True)
sns.set_color_codes("pastel")
sns.barplot(x="Confirmed", y="State/UnionTerritory",
            data=data, label="Total", color="red")
sns.set_color_codes("muted")
sns.barplot(x="Cured", y="State/UnionTerritory",
            data=data, label="Cured", color="green")
ax.legend(ncol=5, loc="lower right", frameon=True)
ax.set(ylabel="", xlabel="Cases")
total = total_cases_india
i = 0
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_width()/total)
    x = p.get_x() + p.get_width() + 3
    y = p.get_y() + p.get_height()/2
    if i <= len(States):
        ax.annotate(" "*10 + str(percentage), (x, y))
    else:
        ax.annotate(percentage, (x, y))

    i += 1
    print("Total infected cases in India: ", total_cases_india)
print("Total cured cases in India: ", cured_cases_india)
print("Total active cases in India: ", active_cases_india)
print("Total death cases in India: ", death_cases_india)
plot_pie(active_cases_india, cured_cases_india, death_cases_india, "India")

# VISUALISING THE SPREADS GEOGRAPHICALLY

India_coord.rename(
    columns={"Name of State / UT": "State/UnionTerritory"}, inplace=True)

set(India_coord['State/UnionTerritory'].values).symmetric_difference(
    set(state_df['State/UnionTerritory'].values))

India_coord['State/UnionTerritory'] = India_coord['State/UnionTerritory'].str.strip()
state_df['State/UnionTerritory'] = state_df['State/UnionTerritory'].str.strip()

set(India_coord['State/UnionTerritory'].values).symmetric_difference(
    set(state_df['State/UnionTerritory'].values))

India_coord.loc[India_coord.shape[0]] = ['Gujarat', '22.2587', '71.1924']
print(India_coord)
set(India_coord['State/UnionTerritory'].values).symmetric_difference(
    set(state_df['State/UnionTerritory'].values))

India_coord['State/UnionTerritory'] = np.where(India_coord['State/UnionTerritory'] == "Andaman And Nicobar",
                                               "Andaman and Nicobar Islands", India_coord['State/UnionTerritory'])
India_coord['State/UnionTerritory'] = np.where(India_coord['State/UnionTerritory'] == "Union Territory of Jammu and Kashmir",
                                               "Jammu and Kashmir", India_coord['State/UnionTerritory'])
India_coord['State/UnionTerritory'] = np.where(India_coord['State/UnionTerritory'] == "Union Territory of Ladakh",
                                               "Ladakh", India_coord['State/UnionTerritory'])
India_coord['State/UnionTerritory'] = np.where(India_coord['State/UnionTerritory'] == "Orissa",
                                               "Odisha", India_coord['State/UnionTerritory'])
India_coord['State/UnionTerritory'] = np.where(India_coord['State/UnionTerritory'] == "Dadra And Nagar Haveli",
                                               "Dadar Nagar Haveli", India_coord['State/UnionTerritory'])

set(India_coord['State/UnionTerritory'].values).symmetric_difference(
    set(state_df['State/UnionTerritory'].values))

df_full = pd.merge(India_coord, state_df,
                   on='State/UnionTerritory').reset_index(drop=True)
df_full

map = folium.Map(location=[20, 70], zoom_start=4, tiles='Stamenterrain')

for lat, lon, value, name in zip(df_full['Latitude'], df_full['Longitude'], df_full['Confirmed'], df_full['State/UnionTerritory']):
    folium.CircleMarker([lat, lon], radius=value*0.0015, popup=('<strong>State</strong>: ' + str(name).capitalize() +
                                                                '<br>''<strong>Total Cases</strong>: ' + str(value) + '<br>'), color='red', fill_color='red', fill_opacity=0.3).add_to(map)
map

map = folium.Map(location=[20, 70], zoom_start=4, tiles='OpenStreetMap')

for lat, lon, value, name in zip(df_full['Latitude'], df_full['Longitude'], df_full['Confirmed'], df_full['State/UnionTerritory']):
    folium.CircleMarker([lat, lon], radius=value*0.0015, popup=('<strong>State</strong>: ' + str(name).capitalize() +
                                                                '<br>''<strong>Total Cases</strong>: ' + str(value) + '<br>'), color='red', fill_color='red', fill_opacity=0.3).add_to(map)
map

map = folium.Map(location=[20, 70], zoom_start=4, tiles='Stamenwatercolor')

for lat, lon, value, name in zip(df_full['Latitude'], df_full['Longitude'], df_full['Confirmed'], df_full['State/UnionTerritory']):
    folium.CircleMarker([lat, lon], radius=value*0.0015, popup=('<strong>State</strong>: ' + str(name).capitalize() +
                                                                '<br>''<strong>Total Cases</strong>: ' + str(value) + '<br>'), color='red', fill_color='red', fill_opacity=0.3).add_to(map)
map

# Lets check the trend of the virus

df_daywise_India = df_final_India.groupby(
    "Date")['Confirmed', 'Cured', 'Deaths', "New Cases"].sum().reset_index()
df_daywise_India


fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_daywise_India['Date'], y=df_daywise_India['Confirmed'], mode='lines+markers', name='Total Cases'))
fig.update_layout(title_text='Trend of Coronavirus Cases in India (Cumulative cases)',
                  plot_bgcolor='rgb(230, 230, 230)')
fig.show()

fig = px.bar(df_daywise_India, x="Date", y="New Cases",
             barmode='group', height=400)
fig.update_layout(title_text='Coronavirus Cases in India on daily basis',
                  plot_bgcolor='rgb(230, 230, 230)')
fig.show()

fig = px.bar(df_daywise_India, x="Date", y="Confirmed", color='Confirmed', orientation='v', height=600,
             title='Confirmed Cases in India', color_discrete_sequence=px.colors.cyclical.IceFire)

'''Colour Scale for plotly
https://plot.ly/python/builtin-colorscales/
'''

fig.update_layout(plot_bgcolor='rgb(230, 230, 230)')
fig.show()


# Forecasting Using fbprophet

df = df_daywise_India.iloc[:-1, ]
df_train = df.loc[df['Date'] <= "2020-05-23", :]
df_test = df.loc[df['Date'] > "2020-05-23", :]

confirmed_train = df_train[['Date', 'Confirmed']]
confirmed_test = df_test[['Date', 'Confirmed']]

deaths_train = df_train[['Date', 'Deaths']]
deaths_test = df_test[['Date', 'Deaths']]

recovered_train = df_train[['Date', 'Cured']]
recovered_test = df_test[['Date', 'Cured']]

confirmed_train.columns = ['ds', 'y']
confirmed_train.tail()

m = prophet()
m.fit(confirmed_train)
future = m.make_future_dataframe(periods=5, freq="D")
future.tail(5)
forecast = m.predict(future)
forecast

result_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5)
result_df['Actual'] = confirmed_test['Confirmed']
result_df

trace0 = go.Scatter(
    x=result_df['ds'],
    y=result_df['Actual'],
    mode='lines+markers',
    name='Actuals',
    line=dict(color='#dd0000', shape='linear'),
    opacity=0.3,
    connectgaps=True
)
trace1 = go.Scatter(
    x=result_df['ds'],
    y=result_df['yhat'],
    name='Predicted',
    mode='lines+markers',
    marker=dict(
        size=10,
        color='#44dd00'),
    opacity=0.3
)
data = [trace0, trace1]
layout = go.Layout(
    yaxis=dict(
        title="Results for Prophet (Total Cases)"
    )
)
fig = go.Figure(data=data, layout=layout)
fig.show()

recovered_train.columns = ['ds', 'y']
recovered_train.tail()

m = Prophet()
m.fit(recovered_train)
future = m.make_future_dataframe(periods=5, freq="D")
future.tail(5)

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5)
result_df = forecast.tail(5)
result_df['Actual'] = recovered_test['Cured']
result_df

trace0 = go.Scatter(
    x=result_df['ds'],
    y=result_df['Actual'],
    mode='lines+markers',
    name='Actuals',
    line=dict(color='#dd0000', shape='linear'),
    opacity=0.3,
    connectgaps=True
)
trace1 = go.Scatter(
    x=result_df['ds'],
    y=result_df['yhat'],
    name='Predicted',
    mode='lines+markers',
    marker=dict(
        size=10,
        color='#44dd00'),
    opacity=0.3
)
data = [trace0, trace1]
layout = go.Layout(
    yaxis=dict(
        title="Results for Prophet (Recovered)"
    )
)
fig = go.Figure(data=data, layout=layout)
fig.show()

deaths_train.columns = ['ds', 'y']
deaths_train.tail()

m = Prophet(seasonality_mode='multiplicative')
m.fit(deaths_train)
future = m.make_future_dataframe(periods=5, freq="D")
future.tail(5)

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5)
result_df = forecast.tail(5)
result_df['Actual'] = deaths_test['Deaths']
result_df

trace0 = go.Scatter(
    x=result_df['ds'],
    y=result_df['Actual'],
    mode='lines+markers',
    name='Actuals',
    line=dict(color='#dd0000', shape='linear'),
    opacity=0.3,
    connectgaps=True
)
trace1 = go.Scatter(
    x=result_df['ds'],
    y=result_df['yhat'],
    name='Predicted',
    mode='lines+markers',
    marker=dict(
        size=10,
        color='#44dd00'),
    opacity=0.3
)
data = [trace0, trace1]
layout = go.Layout(
    yaxis=dict(
        title="Results for Prophet (Death)"
    )
)
fig = go.Figure(data=data, layout=layout)
fig.show()

confirmed['day'] = confirmed['ds'].dt.day
confirmed['month'] = confirmed['ds'].dt.month
confirmed['year'] = confirmed['ds'].dt.year
# del confirmed['ds']
