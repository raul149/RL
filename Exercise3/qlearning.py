import gym
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb

np.random.seed(123)

env = gym.make('CartPole-v0')
#env = gym.make('LunarLander-v2')
env.seed(321)

episodes = 20000
test_episodes = 10
num_of_actions = 2

# Reasonable values for Cartpole discretization
discr = 16
x_min, x_max = -2.4, 2.4
v_min, v_max = -3, 3
th_min, th_max = -0.3, 0.3
av_min, av_max = -4, 4

'''# For LunarLander, use the following values:
#         [  x     y  xdot ydot theta  thetadot cl  cr
#s_min = [ -1.2  -0.3  -2.4  -2  -6.28  -8       0   0 ]
#s_max = [  1.2   1.2   2.4   2   6.28   8       1   1 ]
x_min= -1.2
x_max= 1.2
ymin=-0.3
ymax=1.2
vx_min= -2.4
vx_max= 2.4
vymin=-2.0
vymax=2.0
th_min=-6.28
th_max=6.28
av_min=-8
av_max=8
'''


# Parameters
gamma = 0.98
alpha = 0.1
target_eps = 0.1
a = int(20000*target_eps/(1-target_eps))
initial_q = 0  # T3: Set to 50

# Create discretization grid
x_grid = np.linspace(x_min, x_max, discr)
v_grid = np.linspace(v_min, v_max, discr)
th_grid = np.linspace(th_min, th_max, discr)
av_grid = np.linspace(av_min, av_max, discr)


q_grid = np.zeros((discr, discr, discr, discr, num_of_actions)) + initial_q

'''# Create discretization grid
x_grid = np.linspace(x_min, x_max, discr)
y_grid = np.linspace(ymin, ymax, discr)
vx_grid = np.linspace(vx_min, vx_max, discr)
vy_grid = np.linspace(vymin, vymax, discr)
th_grid = np.linspace(th_min, th_max, discr)
av_grid = np.linspace(av_min, av_max, discr)


q_grid = np.zeros((discr, discr, discr, discr, discr, discr, 2 , 2, num_of_actions)) + initial_q'''

def find_nearest(array, value):
    return np.argmin(np.abs(array - value))

##CARTPOLE
def get_cell_index(state):
    x = find_nearest(x_grid, state[0])
    v = find_nearest(v_grid, state[1])
    th = find_nearest(th_grid, state[2])
    av = find_nearest(av_grid, state[3])
    return x, v, th, av
##LUNAR
    '''
def get_cell_index(state):
    x = find_nearest(x_grid, state[0])
    y = find_nearest(y_grid, state[1])
    vx = find_nearest(vx_grid, state[2])
    vy = find_nearest(vy_grid, state[3])
    th = find_nearest(th_grid, state[4])
    av = find_nearest(av_grid, state[5])
    return x, y, vx, vy, th, av
'''
###CARTPOLE
def get_action(state, q_values, greedy=False):
    # TODO: Implement epsilon-greedy
    x,v,th,av = get_cell_index(state)
    if np.random.random()>epsilon or greedy==True:
        action= np.argmax(q_values[x,v,th,av])
    else:
        action = int(np.random.random()*num_of_actions)
    #raise NotImplementedError("Implement epsilon-greedy")
    return action
####LUNAR
'''def get_action(state, q_values, greedy=False):
    # TODO: Implement epsilon-greedy
    x,y,vx,vy,th,av = get_cell_index(state)
    cl=int(state[6])
    cr=int(state[7])
    if np.random.random()>epsilon or greedy==True:
        action= np.argmax(q_values[x,y,vx,vy,th,av,cl,cr])
    else:
        #print('here')
        action = int(np.random.random()*num_of_actions)
    #raise NotImplementedError("Implement epsilon-greedy")
    return action'''
    
#CARTPOLE
def update_q_value(old_state, action, new_state, reward, done, q_array):
    # TODO: Implement Q-value update
    x,v,th,av = get_cell_index(old_state)
    nx,nv,nth,nav = get_cell_index(new_state)
    newaction=get_action(new_state, q_grid, True)
    if not done:
        q_grid[x,v,th,av,action]=q_array[x,v,th,av,action]+alpha*(reward+gamma*q_array[nx,nv,nth,nav,newaction]-q_array[x,v,th,av,action])
        #print(q_array[x,v,th,av,action]+alpha*(reward+gamma*np.max(q_array[nx,nv,nth,nav])-q_array[x,v,th,av,action]))
    else:
        q_grid[x,v,th,av,action]=q_array[x,v,th,av,action]+alpha*(reward+gamma*0-q_array[x,v,th,av,action])

    
#LUNAR
'''def update_q_value(old_state, action, new_state, reward, done, q_array):
    # TODO: Implement Q-value update
    x,y,vx,vy,th,av = get_cell_index(old_state)
    cl = int(old_state[6])
    cr = int(old_state[7])
    nx,ny,nvx,nvy,nth,nav = get_cell_index(new_state)
    ncl = int(new_state[6])
    ncr = int(new_state[7])    
    newaction=get_action(new_state, q_grid, True)
    if not done:
        q_grid[x,y,vx,vy,th,av,cl,cr,action]=q_array[x,y,vx,vy,th,av,cl,cr,action]+alpha*(reward+gamma*q_array[nx,ny,nvx,nvy,nth,nav,ncl,ncr,newaction]-q_array[x,y,vx,vy,th,av,cl,cr,action])
        #print(q_array[x,v,th,av,action]+alpha*(reward+gamma*np.max(q_array[nx,nv,nth,nav])-q_array[x,v,th,av,action]))
    else:
        q_grid[x,y,vx,vy,th,av,cl,cr,action]=q_array[x,y,vx,vy,th,av,cl,cr,action]+alpha*(reward+gamma*0-q_array[x,y,vx,vy,th,av,cl,cr,action])   
'''

# Training loop
ep_lengths, epl_avg = [], []
for ep in range(episodes+test_episodes):
    test = ep > episodes
    state, done, steps = env.reset(), False, 0
    #epsilon = 0.000
    epsilon = a/(a+ep)
    #print(epsilon)
    # T1: GLIE/constant, T3: Set to 0
    while not done:
        action = get_action(state, q_grid, greedy=test)
        new_state, reward, done, _ = env.step(action)
        if not test:
            update_q_value(state, action, new_state, reward, done, q_grid)
        else:
            env.render()
        state = new_state
        steps += 1
    ep_lengths.append(steps)
    epl_avg.append(np.mean(ep_lengths[max(0, ep-500):]))
    if ep % 200 == 0:
        print("Episode {}, average timesteps: {:.2f}".format(ep, np.mean(ep_lengths[max(0, ep-200):])))
# Save the Q-value array
np.save("q_values.npy", q_grid)  # TODO: SUBMIT THIS Q_VALUES.NPY ARRAY

# Calculate the value function
values = np.zeros(q_grid.shape[:-1])  # TODO: COMPUTE THE VALUE FUNCTION FROM THE Q-GRID
for i in range(discr-1):
    for j in range(discr-1):
        for k in range(discr-1):
            for l in range(discr-1):
                values[i][j][k][l]= np.max(q_grid[i,j,k,l])
#print(values)    
np.save("value_func.npy", values)  # TODO: SUBMIT THIS VALUE_FUNC.NPY ARRAY


# Plot the heatmap
# TODO: Plot the heatmap here using Seaborn or Matplotlib
#plt.imshow(np.mean(values, axis=(1, 3)))
sb.heatmap(np.mean(values,xticklabels=x_grid,yticklabels=th_grid)


plt.show()
# Draw plots
plt.plot(ep_lengths)
plt.plot(epl_avg)
plt.legend(["Episode length", "500 episode average"])
plt.title("Episode lengths")
plt.show()

