# %%
import time
import random
import requests
from datetime import datetime, timedelta

import numpy as np
import pandas as pd  # Data handling
from tqdm.auto import tqdm
import plotly.express as px

from cpmpy import *  # CPMpy constraint solving library

# %%
"""
# Read room requests from CSV (formerly: from Excell sheet)
"""

# %%
#url = "https://raw.githubusercontent.com/CPMpy/cpmpy/master/examples/room_assignment.csv"
#df = pd.read_csv(url, parse_dates=[0,1])
#df.head()

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
"""
# Randomly generate an assignment
First generate a full schedule of `n_rooms` and days from `day_start` to `day_end`, where each request is between `req_min` and `req_max` days, and with a gap between `gap_min` and `gap_max`.
"""

# %%
n_rooms = 50
day_start = datetime.strptime("2022-07-01", "%Y-%m-%d")
day_end = datetime.strptime("2022-08-31", "%Y-%m-%d")
req_min = 1
req_max = 7
gap_min = 0
gap_max = 1

data = [] # (start, end, room, Room Number)
for room in range(n_rooms):
    #print(room)
    day_cur = day_start
    while day_cur < day_end:
        # generate request
        n_days = random.uniform(req_min, req_max)
        day_cur_end = min([day_cur + timedelta(days=n_days), day_end])
        data.append( (day_cur, day_cur_end, room) )
        day_cur = day_cur_end
        
        # generate gap
        n_gap = random.uniform(gap_min, gap_max)
        day_cur = min([day_cur + timedelta(days=n_gap), day_end])

df = pd.DataFrame(data, columns=["start","end","room"]).sample(frac=1).reset_index(drop=True) # shuffle
fig = px.timeline(df, x_start="start", x_end="end", y="room", color=df.index)
fig.show()
df['room'] = np.nan

# solve ours
rooms = n_rooms
(m, requestvars) = model_rooms(df, max_rooms=n_rooms)

sat = m.solve()
print(sat, m.status())

# %%
"""
# Model the room assignment problem

* All requests must be assigned to one out of 5 rooms (same room during entire period).
* Some requests already have a room pre-assigned.
* A room can only serve one request at a time.
"""

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

# %%
# Incremental implementation
class IncrRooms:
    def __init__(self, n_rooms, solvername="ortools", restart=False, verbose=True):
        self.n_rooms = n_rooms
        self.verbose = verbose
        
        self.df = None
        self.vars = []
        
        self.restart = restart
        if self.restart:
            # restart from scratch each time
            self.solvername = solvername
            self.s = Model()
        else:
            # use incremental solver
            self.s = SolverLookup.get(solvername)
    
    def addRequest(self, reqdf):
        # single request at a time only
        assert(len(reqdf.shape) == 1) # so single dim
        
        # make var of this request
        reqvar = intvar(0, self.n_rooms-1)
        
        # fixed room?
        if ~np.isnan(reqdf['room']):
            self.s += ( reqvar == int(reqdf['room']) )
            
        # requests on same day must be in separate rooms
        if self.df is not None:
            orders_idx = set()
            #  a bit silly with the loop, but I trust it more
            for date in pd.date_range(reqdf['start'], reqdf['end'], closed="left"):
                orders_idx.update([int(idx) for idx in self.df[(self.df['start'] <= date) & (date < self.df['end'])].index])
            #orders = self.df[(self.df['start'] <= reqdf['start']) & (reqdf['start'] < self.df['end']) | # l overlap
            #                 (reqdf['start'] <= self.df['start']) & (self.df['end'] < reqdf['end']) | # mid
            #                 (self.df['start'] < reqdf['end']) & (self.df['end'] <= reqdf['end'])]  # r overlap
            if len(orders_idx) > 1:
                self.s += [reqvar != self.vars[idx] for idx in orders_idx]
        
        # store things
        self.vars.append(reqvar)
        if self.df is None:
            self.df = reqdf.to_frame().T
        else:
            self.df = self.df.append(reqdf.to_frame().T)
    
    def solve(self):
        if self.restart:
            return self.s.solve(self.solvername)
        else:
            return self.s.solve()

    
#sheet, rooms = ("P-Large-02 (59 ROOMS)",60) #("P-Small-01-(4 ROOMS)",5) #("P-Large-01-(59 ROOMS)",60)
#rooms = 80
#df = pd.read_csv("room_assignment.csv", parse_dates=[0,1])
df_shuffle = df.sample(frac=1).reset_index(drop=True)

"""
print("Stand-alone, 200:")
(m, ordervars) = model_rooms(df_shuffle.iloc[:200,:], rooms, verbose=False)
s = SolverLookup.get('ortools', m)
if not s.solve():
        print("Got unsat?")
print("\t", s.status())
"""

# %%
print("Incr, ort (model is incremental but solver restarts from scratch):")
t0 = time.time()
ir = IncrRooms(rooms, "ortools")
logr_ort = [] # (nr_orders, runtime)
for i in tqdm( range(0,min([600,len(df_shuffle)])) ):
    ir.addRequest(df_shuffle.iloc[i,:])
    st = ir.solve()
    if not st:
        print("Error: model is UNSAT?")
        break
    logr_ort.append((i, ir.s.status().runtime))
print(f"{i}: {st} {ir.s.status()}", time.time()-t0)

# %%
print("Incr, grb:")
t0 = time.time()
ir = IncrRooms(rooms, "gurobi")
logr_grb = [] # (nr_orders, runtime)
for i in tqdm( range(0,min([600,len(df_shuffle)])) ):
    ir.addRequest(df_shuffle.iloc[i,:])
    st = ir.solve()
    if not st:
        print("Error: model is UNSAT?")
        break
    logr_grb.append((i, ir.s.status().runtime))
print(f"{i}: {st} {ir.s.status()}", time.time()-t0)

# %%
l1 = pd.DataFrame(logr_ort, columns=["nr", "Non-incremental"])
l2 = pd.DataFrame(logr_grb, columns=["nr", "Incremental"])
l = l1.merge(l2, on="nr")
l.plot(x="nr")

# %%
