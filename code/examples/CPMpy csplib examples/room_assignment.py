# %%
import pandas as pd  # Data handling
from cpmpy import *  # CPMpy constraint solving library

# %%
"""
# Read room requests from CSV (formerly: from Excell sheet)
"""

# %%
df = pd.read_csv("room_assignment.csv", parse_dates=[0,1])
df.head()

# %%
"""
# Model the room assignment problem

* All requests must be assigned to one out of 5 rooms (same room during entire period).
* Some requests already have a room pre-assigned.
* A room can only serve one request at a time.
"""

# %%
def model_rooms(df, max_rooms, verbose=True):
    n_requests = len(df)

    # All requests must be assigned to one out of the rooms (same room during entire period).
    requestvars = intvar(0, max_rooms-1, shape=(n_requests,))

    m = Model()

    # Some requests already have a room pre-assigned
    for idx, row in df.iterrows():
        if not pd.isna(row['room']):
            m += (requestvars[idx] == int(row['room']))

    # A room can only serve one request at a time.
    # <=> requests on the same day must be in different rooms
    for day in pd.date_range(min(df['start']), max(df['end'])):
        overlapping = df[(df['start'] <= day) & (day < df['end'])]
        if len(overlapping) > 1:
            m += AllDifferent(requestvars[overlapping.index])
            
    return (m, requestvars)

# %%
# solve ours
(m, requestvars) = model_rooms(df, max_rooms=5)

sat = m.solve()
print(sat, m.status())

# %%
"""
# Visualize the room calendar
"""

# %%
# use Plotly's excellent gantt chart support
import plotly.express as px

df['Room Number'] = requestvars.value()  # value in the solution
df['Room Number'].fillna(0, inplace=True)
fig = px.timeline(df, x_start="start", x_end="end", y="Room Number", color=df.index)
fig.show()