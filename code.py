import numpy as np
from numpy import linalg as LA
from math import sqrt, sin, cos
from Tkinter import *
import time
import functools
import heapq


#Drawing parameters
pixelsize = 1024
framedelay = 30
drawVels = True
QUIT = False
paused = False
step = False
circles = []
velLines = []
gvLines = []


#Initalize parameters to run a simulation
# the simulation time step
dt = 0.1 
scenarioFile='hallway_agents.csv'
# exort the simulation?
doExport = True
# the simulated agents
agents = []
# keep track of the agents' traces
trajectories = []
# keep track of simulation iterations 
ittr = 0
#how many time steps we want to simulate
maxIttr = 500  
# simuation time      
globalTime = 0 
# is the agent close to its goal?
goalRadiusSq = 1 
# have all agents reached their goals
reachedGoals = False
# number of neighbors
NeighborNum = 10
# threshold of neighbor distance
NeighborDist = 5

gama = 8.5


#=======================================================================================================================
# read a scenario
#=======================================================================================================================
def readScenario(fileName, scalex=1., scaley=1.):
    if fileName=='hallway_agents.csv':
        scalex = 0.75
        scaley = 2
    
    # it may be better to define an Agent class, here I'm using a lazy approach 
    fp = open(fileName, 'r')
    lines = fp.readlines()
    fp.close()
    for line in lines:
        tokens = line.split(',')
        id = int(tokens[0]) # the id of the agent
        gid = int(tokens[1]) # the group id of the agent
        pos = np.array([float(tokens[2]), float(tokens[3])]) # the position of the agent 
        vel = [np.zeros(2)] # the velocity of the agent
        goal = np.array([float(tokens[4]), float(tokens[5])]) # the goal of the agent
        prefspeed = float(tokens[6]) # the preferred speed of the agent
        gvel = goal-pos # the goal velocity of the agent
        gvel = gvel/(sqrt(gvel.dot(gvel)))*prefspeed       
        maxspeed = float(tokens[7]) # the maximum sped of the agent
        radius = float(tokens[8]) # the radius of the agent
        agents.append([id, gid, pos, vel, gvel, goal, radius, prefspeed, maxspeed, False])    
    
    # define the boundaries of the environment
    positions = [row[2] for row in agents]
    goals = [row[5] for row in agents]
    x_min =	min(np.amin(np.array(positions)[:,0]), np.amin(np.array(goals)[:,0]))*scalex - 2.
    y_min =	min(np.amin(np.array(positions)[:,1]), np.amin(np.array(goals)[:,1]))*scaley - 2.
    x_max =	max(np.amax(np.array(positions)[:,0]), np.amax(np.array(goals)[:,0]))*scalex + 2.
    y_max =	max(np.amax(np.array(positions)[:,1]), np.amax(np.array(goals)[:,1]))*scaley + 2.

    return x_min, x_max, y_min, y_max 


#=======================================================================================================================
# initialize the agents 
#=======================================================================================================================
def initWorld(canvas):
    print ("")
    print ("Simulation of Agents on a flat 2D torus.")
    print ("Agents do not avoid collisions")
    print ("Green Arrow is Goal Velocity, Red Arrow is Current Velocity")
    print ("SPACE to pause, 'S' to step frame-by-frame, 'V' to turn the velocity display on/off.")
    print ("")
       
    colors = ["white","blue","yellow", "#FAA"]
    for agent in agents:
        circles.append(canvas.create_oval(0, 0, agent[6], agent[6], fill=colors[agent[1]%4])) # color the disc of an agenr based on its group id
        velLines.append(canvas.create_line(0,0,10,10,fill="red"))
        gvLines.append(canvas.create_line(0,0,10,10,fill="green"))

      
#=======================================================================================================================
# draw the agents
#=======================================================================================================================
def drawWorld():    
    for i in range(len(agents)):
        agent = agents[i]
        if not agent[-1]:
            canvas.coords(circles[i],world_scale*(agent[2][0]- agent[6] - world_xmin), world_scale*(agent[2][1] - agent[6] - world_ymin), world_scale*(agent[2][0] + agent[6] - world_xmin), world_scale*(agent[2][1] + agent[6] - world_ymin))
            canvas.coords(velLines[i],world_scale*(agent[2][0] - world_xmin), world_scale*(agent[2][1] - world_ymin), world_scale*(agent[2][0]+ agent[6]*agent[3][0] - world_xmin), world_scale*(agent[2][1] + agent[6]*agent[3][1] - world_ymin))
            canvas.coords(gvLines[i],world_scale* (agent[2][0] - world_xmin), world_scale*(agent[2][1] - world_ymin), world_scale*(agent[2][0]+ agent[6]*agent[3][0] - world_xmin), world_scale*(agent[2][1] + agent[6]*agent[3][1] - world_ymin))
            if drawVels:
                canvas.itemconfigure(velLines[i], state="normal")
                canvas.itemconfigure(gvLines[i], state="normal")
            else:
                canvas.itemconfigure(velLines[i], state="hidden")
                canvas.itemconfigure(gvLines[i], state="hidden")


#=======================================================================================================================
# keyboard events
#=======================================================================================================================                        
def on_key_press(event):
    global paused, step, QUIT, drawVels

    if event.keysym == "space":
        paused = not paused
    if event.keysym == "s":
        step = True
        paused = False
    if event.keysym == "v":
        drawVels = not drawVels
    if event.keysym == "Escape":
        QUIT = True


def _cmp(x, y):
    if y[1] - x[1] < 0:
        return -1
    elif y[1] - x[1] == 0:
        return 0
    return 1

#=======================================================================================================================
# Find the K nearest neighbors 
#=======================================================================================================================                        
def find_k_nighbors(agent, NeighborNum, NeighborDist):
    neighagents = []
    for j in range(len(agents)):
        if agent[0] != agents[j][0]:
            if LA.norm(agent[2] - agents[j][2]) < NeighborDist:
                neighagents.append([agents[j][0], LA.norm(agent[2] - agents[j][2]), agents[j][2], agents[j][3], agents[j][6]])

    sortedneibors = sorted(neighagents, cmp = _cmp)

    if len(sortedneibors) > NeighborNum:
        sortedKneibors = sortedneibors[:NeighborNum]
    else:
        sortedKneibors = sortedneibors
    return sortedKneibors


#=======================================================================================================================
# Find the admissable velocity
#=======================================================================================================================
def adVelocity(agent):
    agentAdVel = []       
    num = 100

    t = np.random.uniform(0.0, 2.0*np.pi, num)
    r = agent[-2] * np.sqrt(np.random.uniform(0.0, 1.0, num))
    x = r * np.cos(t)
    y = r * np.sin(t)
    for VelNum in range(num):
        agentAdVel.append([x[VelNum],y[VelNum]])
    return agentAdVel


#=======================================================================================================================
# Time to collision
#=======================================================================================================================
def timetocollision(agent, neiborKagents, canVel):
    tau = np.zeros(len(neiborKagents))
    discr = [0]*len(neiborKagents)
    radiussum = [0]*len(neiborKagents)
    relaposition = [0]*len(neiborKagents)
    c = [0]*len(neiborKagents)
    relavelocity = [0]*len(neiborKagents)
    a = [0]*len(neiborKagents)
    b = [0]*len(neiborKagents)
    for j in range(len(neiborKagents)):
        radiussum[j] = agent[6] + neiborKagents[j][4]
        relaposition[j] = agent[2] - neiborKagents[j][2]
        relavelocity[j] = canVel - neiborKagents[j][3]
        a[j] = np.dot(relavelocity[j], relavelocity[j])
        b[j] = np.dot(relaposition[j], relavelocity[j])
        c[j] = np.dot(relaposition[j], relaposition[j]) - radiussum[j]*radiussum[j]
        # agents are colliding
        if c[j] < 0:
            if b[j] < 0:
                # agents are moving away
                tau[j] = 100000
            else:
                # agents are continuing colliding
                tau[j] = 0.000001
        else:
            discr[j] = b[j]*b[j] - a[j]*c[j]
            if discr[j] <= 0:
                tau[j] = 100000
            else:
                tau[j] = c[j]/(-b[j] + sqrt(discr[j]))
                if tau[j] < 0:
                    tau[j] = 100000
    return tau


#=======================================================================================================================
# find fit velocity
#=======================================================================================================================
def fitVelocity(gama,agent,neiborKagents):
    costtoNeibor = []
    for canVel in adVelocity(agent):
        tau = timetocollision(agent, neiborKagents, canVel)
        costvalue = LA.norm(agent[4] - canVel) + gama/min(tau)
        costtoNeibor.append([costvalue, canVel])
    mincost = min(costtoNeibor, key = lambda t: t[0])
    print costtoNeibor
    print mincost[1]
    return np.array(mincost[1])


#=======================================================================================================================
# update the simulation 
#=======================================================================================================================
def updateSim(dt):
    global reachedGoals
    for agent in agents:
        if not agent[-1]:
            #compute the goal velocity, here we assume it's the same as the goal velocity
            agent[3] = agent[4]

    reachedGoals = True
    for agent in agents:
        if not agent[-1]:
            neiborKagents_ = find_k_nighbors(agent, NeighborNum, NeighborDist)
            if neiborKagents_ != []:
                if np.min(timetocollision(agent, neiborKagents_, agent[3])) != 100000:
                    agent[3] = fitVelocity(gama,agent,neiborKagents_)

            agent[2] += agent[3]*dt   #update the position
            gvel = agent[5] - agent[2]
            distGoalSq = gvel.dot(gvel)
            # goal has been reached
            if distGoalSq < goalRadiusSq: 
                agent[-1] = True
            else: 
                #compute the goal velocity for the next time step
                gvel = gvel/sqrt(distGoalSq)*agent[7]
                agent[4] = gvel
                reachedGoals = False


#=======================================================================================================================
# simulate and draw frames 
#=======================================================================================================================                        
def drawFrame(dt):
    global start_time,step,paused,ittr,globalTime

    if reachedGoals or ittr > maxIttr or QUIT: #Simulation Loop
        print("%s itterations ran ... quitting"%ittr)
        win.destroy()
    else:
        elapsed_time = time.time() - start_time
        start_time = time.time()
        if not paused:
            updateSim(dt)
            ittr += 1
            globalTime += dt
            for agent in agents:
                if not agent[-1]:
                    trajectories.append([agent[0], agent[1], agent[2][0], agent[2][1], agent[3][0], agent[3][1], agent[6], globalTime])

        drawWorld()
        if step == True:
            step = False
            paused = True    
        
        win.title('Multi-Agent Navigation')
        win.after(framedelay,lambda: drawFrame(dt))
  

#=======================================================================================================================
# Main execution of the code
#=======================================================================================================================
world_xmin, world_xmax, world_ymin, world_ymax = readScenario(scenarioFile)
world_width = world_xmax - world_xmin
world_height = world_ymax - world_ymin
world_scale = pixelsize/world_width

# set the visualizer
win = Tk()
# keyboard interaction
win.bind("<space>",on_key_press)
win.bind("s",on_key_press)
win.bind("<Escape>",on_key_press)
win.bind("v",on_key_press)
# the drawing canvas
canvas = Canvas(win, width=pixelsize, height=pixelsize*world_height/world_width, background="#666")
canvas.pack()
initWorld(canvas)
start_time = time.time()
# the main loop of the program
win.after(framedelay, lambda: drawFrame(dt))
mainloop()
if doExport:
    header = "id,gid,,x,y,v_x,v_y,radius,time"
    exportFile = scenarioFile.split('.csv')[0] + "_sim.csv"
    np.savetxt(exportFile, trajectories, delimiter=",", fmt='%d,%d,%f,%f,%f,%f,%f,%f', header=header, comments='')
