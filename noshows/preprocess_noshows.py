import pandas as pd
import datetime

dataset = pd.read_csv("data/noshows.csv")

# convert M e F para numeros
dataset['Gender'] = dataset['Gender'].replace('F',1)
dataset['Gender'] = dataset['Gender'].replace('M',0)

dataset['No-show'] = dataset['No-show'].replace('No', 0)
dataset['No-show'] = dataset['No-show'].replace('Yes', 1)
dataset.rename(columns={'No-show':'class'},inplace=True)

dataset.AppointmentDay = pd.to_datetime(pd.Series(dataset.AppointmentDay))
dataset.AppointmentDay = dataset.AppointmentDay.dt.week

dataset.ScheduledDay = pd.to_datetime(pd.Series(dataset.ScheduledDay))
dataset.ScheduledDay = dataset.ScheduledDay.dt.week

dataset.to_csv("data/base_noshows.csv", index=False)

