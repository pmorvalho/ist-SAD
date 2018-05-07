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
# dataset.AppointmentDay = dataset.AppointmentDay.dt.dayofyear
dataset.AppointmentDay = dataset.AppointmentDay.dt.week

dataset.ScheduledDay = pd.to_datetime(pd.Series(dataset.ScheduledDay))
# dataset.ScheduledDay = dataset.ScheduledDay.dt.dayofyear
dataset.ScheduledDay = dataset.ScheduledDay.dt.week


dataset.drop("PatientId",axis=1,inplace=True)
dataset.drop("AppointmentID",axis=1,inplace=True)
dataset.drop("Neighbourhood",axis=1,inplace=True)

dataset.to_csv("data/base_noshows.csv", index=False)
# dataset.to_csv("data/base__dayofyear_noshows.csv", index=False)

