# %%
from cpmpy import *
import numpy as np

# %%


# %%
model = Model()

gr,bl,og,ye,gy = boolvar(shape=5)

model += (12*gr + 2*bl + 1*og + 4*ye + 1*gy <= 15)

model.maximize(4*gr + 2*bl + 1*og + 10*ye + 2*gy)


model.solve()

# %%
print(gr.value(), bl.value(), og.value(), ye.value(), gy.value())

# %%


# %%
v = boolvar()
print(v)

# %%
v = boolvar(shape=5)
print(v)

# %%
v = intvar(1,9, shape=5)
print(v)

# %%
m = intvar(1,9, shape=(3,3))
print(m)

# %%
t = boolvar(shape=(2,3,4))
print(t)

# %%
puzzle_start = np.array([
    [0,3,6],
    [2,4,8],
    [1,7,5]])

(dim,dim2) = puzzle_start.shape
assert (dim == dim2), "puzzle needs square shape"
n = dim*dim2 - 1 # e.g. an 8-puzzle

# State of puzzle at every step
K = 20
x = intvar(0,n, shape=(K,dim,dim), name="x")
print(x)

# %%


# %%
x,y,z = intvar(1,9, shape=3)
print( x + y )

# %%
print( x * y )

# %%
print( abs(x - y) )

# %%
a = intvar(1,9, shape=5, name="a")
print( sum(a) )

# %%
c = ( abs(sum(a) - (x+y)) == z )
print( c )

# %%
type(c)

# %%
c.name

# %%
c.args[1]

# %%
type(c.args[0])

# %%
c.args[0].name

# %%


# %%
x = boolvar(shape=(4,4), name="x")
print(x)

# %%
print(x[0,:])

# %%
print(x[:,0])

# %%
print(x[:,1:-1])

# %%


# %%
x = boolvar(shape=4, name="x")
print(x)

# %%
sel = np.array([True, False, True, False])
print(x[sel])

# %%
print(x[np.arange(4) % 2 == 0])

# %%
idx = [1,3]
print(x[idx])

# %%


# %%
x = intvar(1,9, shape=3, name="x")
y = intvar(1,9, shape=3, name="y")
print(x + y)

# %%
print(x == [1,2,3])

# %%


# %%
print(x == 1)

# %%


# %%
import numpy as np
from cpmpy import *

e = 0 # value for empty cells
given = np.array([
    [e, e, e,  2, e, 5,  e, e, e],
    [e, 9, e,  e, e, e,  7, 3, e],
    [e, e, 2,  e, e, 9,  e, 6, e],

    [2, e, e,  e, e, e,  4, e, 9],
    [e, e, e,  e, 7, e,  e, e, e],
    [6, e, 9,  e, e, e,  e, e, 1],

    [e, 8, e,  4, e, e,  1, e, e],
    [e, 6, 3,  e, e, e,  e, 8, e],
    [e, e, e,  6, e, 8,  e, e, e]])


# Variables
puzzle = intvar(1,9, shape=given.shape, name="puzzle")

# %%
model = Model()
n = given.shape[0]

# Constraints on rows and columns
for i in range(n):
    model += AllDifferent([puzzle[i,j] for j in range(n)])
    model += AllDifferent([puzzle[j,i] for j in range(n)])

# Constraints on blocks
for i in range(0,9, 3):
    for j in range(0,9, 3):
        model += AllDifferent([puzzle[r,c]
                               for r in range(i,i+3)
                               for c in range(j,j+3)])

# Constraints on values (cells that are not empty)
for r in range(n):
    for c in range(n):
        if given[r,c] != e:
            model += puzzle[r,c] == given[r,c]

model.solve()

# %%
model = Model()


# Constraints on rows and columns
model += [AllDifferent(row) for row in puzzle]
model += [AllDifferent(col) for col in puzzle.T]


# Constraints on blocks
for i in range(0,9, 3):
    for j in range(0,9, 3):
        model += AllDifferent(puzzle[i:i+3, j:j+3])

        
        
# Constraints on values (cells that are not empty)
model += [puzzle[given!=e] == given[given!=e]]




model.solve()

# %%


# %%
jobs_data = cpm_array([ # (job, machine) = duration
    [3,2,2], # job 0
    [2,1,4], # job 1
    [0,4,3], # job 2 (duration 0 = not used)
])
max_dur = sum(jobs_data.flat)

n_jobs, n_machines = jobs_data.shape
all_jobs = range(n_jobs)
all_machines = range(n_machines)


# Variables
start_time = intvar(0, max_dur, shape=(n_machines,n_jobs), name="start")
end_time = intvar(0, max_dur, shape=(n_machines,n_jobs), name="stop")

# %%
from itertools import combinations

# %%
model = Model()
# end = start + dur
for j in all_jobs:
    for m in all_machines:
        model += (end_time[m,j] == start_time[m,j] + jobs_data[j,m])

# Precedence constraint per job
for j in all_jobs:
    for m1,m2 in combinations(all_machines,2): # [0,1,2]->[(0,1),(0,2),(1,2)]
        model += (end_time[m1,j] <= start_time[m2,j])

# No overlap constraint: one starts before other one ends
for m in all_machines:
    for j1,j2 in combinations(all_jobs, 2):
        model += (start_time[m,j1] >= end_time[m,j2]) | \
                 (start_time[m,j2] >= end_time[m,j1])

# Objective: makespan
makespan = Maximum([end_time[m,j] for m in all_machines for j in all_jobs])
model.minimize(makespan)
model.solve()

# %%
model = Model()
# end = start + dur
model += (end_time == start_time + jobs_data.T)



# Precedence constraint per job
for m1,m2 in combinations(all_machines,2):
    model += (end_time[m1,:] <= start_time[m2,:])

    
# No overlap constraint: one starts before other one ends
for j1,j2 in combinations(all_jobs, 2):
    model += (start_time[:,j1] >= end_time[:,j2]) | \
             (start_time[:,j2] >= end_time[:,j1])

    
# Objective: makespan
makespan = max(end_time)
model.minimize(makespan)
model.solve()

# %%
model = Model()
# FOR THE EXPERTS: NOT A CONSTRAINT!
end_time = start_time + jobs_data.T

# Precedence constraint per job
for m1,m2 in combinations(all_machines,2): # [(0,1), (0,2), (1,2)]
    model += (end_time[m1,:] <= start_time[m2,:])

# No overlap constraint: one starts before other one ends
for j1,j2 in combinations(all_jobs, 2):
    model += (start_time[:,j1] >= end_time[:,j2]) | \
             (start_time[:,j2] >= end_time[:,j1])

# Objective: makespan, NOT A CP VARIABLE!
makespan = max(end_time)
model.minimize(makespan)
model.solve()

# %%
print("Makespan:",makespan.value())
print("Schedule:")
grid = -8*np.ones((n_machines, makespan.value()), dtype=int)

for j in all_jobs:
    for m in all_machines:
        grid[m,start_time[m,j].value():end_time[m,j].value()] = j
print(grid)

# %%
from PIL import Image, ImageDraw, ImageFont

# based on Alexander Schiendorfer's https://github.com/Alexander-Schiendorfer/cp-examples
def visualize_scheduling(start, end):
    nMachines, nJobs = start.shape
    makespan = max(end.value())
    
    # Draw solution
    # Define start location image & unit sizes
    start_x, start_y = 30, 40
    pixel_unit = 50
    pixel_task_height = 100
    vert_pad = 10

    imwidth, imheight = makespan * pixel_unit + 2 * start_x, start_y + start_x + nMachines * (pixel_task_height + vert_pad)

    # Create new Image object
    img = Image.new("RGB", (imwidth, imheight), (255, 255,255))

    # Create rectangle image
    img1 = ImageDraw.Draw(img)

    # Get a font
    try:
        myFont = ImageFont.truetype("arialbd.ttf", 20)
    except:
        myFont = ImageFont.load_default()

    # Draw makespan label
    center_x, center_y = imwidth / 2, start_y / 2
    msg =  f"Makespan: {makespan}"
    w, h = img1.textsize(msg, font=myFont)
    img1.text((center_x - w / 2, center_y - h / 2), msg, fill="black", font=myFont)

    task_cs = ["#4bacc6", "#f79646", "#9bbb59"]
    task_border_cs = ["#357d91", "#b66d31", "#71893f"]

    # Draw three rectangles for machines
    machine_upper_lefts = []
    for i in range(nMachines):
        start_m_x, start_m_y = start_x, start_y + i * (pixel_task_height + vert_pad)
        end_m_x, end_m_y = start_m_x + makespan * pixel_unit, start_m_y + pixel_task_height
        machine_upper_lefts += [(start_m_x, start_m_y)]

        shape = [(start_m_x, start_m_y), (end_m_x, end_m_y)]
        img1.rectangle(shape, fill ="#d3d3d3")

    # Draw tasks for each job
    inner_sep = 5
    for j in range(nJobs):
        job_name = str(j)
        for m in range(nMachines):
            if start[m,j].value() == end[m,j].value():
                continue # skip
            start_m_x, start_m_y = machine_upper_lefts[m]

            start_rect_x, start_rect_y = start_m_x + start[m,j].value() * pixel_unit, start_m_y + inner_sep
            end_rect_x, end_rect_y = start_m_x + end[m,j].value() * pixel_unit, start_m_y + pixel_task_height - inner_sep

            shape = [(start_rect_x, start_rect_y), (end_rect_x, end_rect_y)]
            img1.rectangle(shape, fill=task_cs[j], outline=task_border_cs[j])

            # Write a label for each task of each job
            msg =  f"{job_name}"
            text_w, text_h = img1.textsize(msg, font=myFont)
            center_x, center_y = (start_rect_x + end_rect_x) / 2, (start_rect_y + end_rect_y) / 2
            img1.text((center_x - text_w / 2, center_y - text_h / 2), msg, fill="white", font=myFont)

    img.show()
visualize_scheduling(start_time, end_time)

# %%


# %%
# '0' is empty spot
puzzle_start = np.array([
    [3,7,5],
    [1,6,4],
    [8,2,0]])
puzzle_end = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,0]])

def n_puzzle(puzzle_start, puzzle_end, K):
    print("Max steps:", K)
    m = Model()

    (dim,dim2) = puzzle_start.shape
    assert (dim == dim2), "puzzle needs square shape"
    n = dim*dim2 - 1 # e.g. an 8-puzzle

    # State of puzzle at every step
    x = intvar(0,n, shape=(K,dim,dim), name="x")

    
    
    # Start state constraint
    m += (x[0] == puzzle_start)

    # End state constraint
    m += (x[-1] == puzzle_end)

    # define neighbors = allowed moves for the '0'
    def neigh(i,j):
        # same, left,right, down,up, if within bounds
        for (rr, cc) in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]:
            if 0 <= i+rr and i+rr < dim and 0 <= j+cc and j+cc < dim:
                yield (i+rr,j+cc)

    # Transition: define next (t) based on prev (t-1) + invariants
    for t in range(1, K):
        # Invariant: in each step, all cells are different
        m += AllDifferent(x[t])

        # Invariant: only the '0' position can move
        m += ((x[t-1] == x[t]) | (x[t-1] == 0) | (x[t] == 0))
        
        # for each position, determine reachability of the '0' position
        for i in range(dim):
            for j in range(dim):
                m += (x[t,i,j] == 0).implies(any(x[t-1,r,c] == 0 for r,c in neigh(i,j)))

    return (m,x)

# %%
(m,x) = n_puzzle(puzzle_start, puzzle_end, 10)
m.solve()
print(m.status())

# %%
(m,x) = n_puzzle(puzzle_start, puzzle_end, 100)
m.solve()
print(m.status())

# %%
K0 = 5
step = 3

(m,x) = n_puzzle(puzzle_start, puzzle_end, K0)
while not m.solve():
    print(m.status())
    K0 = K0 + step
    (m,x) = n_puzzle(puzzle_start, puzzle_end, K0)

print(m.status())

# %%
(m,x) = n_puzzle(puzzle_start, puzzle_end, 14)
m.solve()
m.status()

# %%
(m,x) = n_puzzle(puzzle_start, puzzle_end, 20)

m.solve()
base_runtime = m.status().runtime
print("Runtime with default params", base_runtime)

from cpmpy.solvers import CPM_ortools, param_combinations

all_params = {'cp_model_probing_level': [0,1,2,3],
              'linearization_level': [0,1,2],
              'symmetry_level': [0,1,2]}

configs = [] # (runtime, param)
for params in param_combinations(all_params):
    s = CPM_ortools(m)
    print("Running", params, end='\r')
    s.solve(time_limit=base_runtime*1.05, **params) # timeout of 105% of base_runtime
    configs.append( (s.status().runtime, params) )

best = sorted(configs)[0]
print("\nFastest in", round(best[0],4), "seconds, config:", best[1])

# %%
(m,x) = n_puzzle(puzzle_start, puzzle_end, 100)
m = CPM_ortools(m)
m.solve(**best[1])
print(m.status())

# %%
