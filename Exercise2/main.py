# Copyright 2020 (c) Aalto University - All Rights Reserved
# ELEC-E8125 - Reinforcement Learning Course
# AALTO UNIVERSITY
#
#############################################################


import numpy as np
from time import sleep
from sailing import SailingGridworld

epsilon = 10e-4  # TODO: Use this criteria for Task 3

# Set up the environment
env = SailingGridworld(rock_penalty=-2)
value_est = np.zeros((env.w, env.h))
env.draw_values(value_est)


if __name__ == "__main__":
    # Reset the environment
    state = env.reset()

    # Compute state values and the policy
    # TODO: Compute the value function and policy (Tasks 1, 2 and 3)
    value_est, policy = np.zeros((env.w, env.h)), np.zeros((env.w, env.h))
    converge=np.zeros((env.w, env.h))
    convergepolicy=np.zeros((env.w, env.h))
    finished=0
    #Task 1-2 (Comment line for Task 3 onwards)
    #for iteration in range(100):
    #Task 3-4  (Comment line for Task 1-2, Uncomment 3-4)  
    while finished==0:
    
    #Calculate for each action the value
        #Create a copy of the value_est, we will use it for keeping the values from state k-1.
        value_est_old=value_est.copy()
        policy_old=policy.copy()
        #Loop for all the x,y values in the matrix, to update their value
        for i in range(env.w):
            for j in range(env.h):
                #Create a list to store the value of each action for a specific state
                setof_actions_values=[]
                #Enter every state, and check eery possible action to take there, and the probabilities of the next state
                for Transition in env.transitions[i,j]:
                    value_s=0
                    for nstate,reward,done,prob in Transition:
                            if not done:
                                value_s=value_s+prob*(reward+0.9*value_est_old[nstate])
                            else:
                                value_s=value_s+prob*(reward+0.9*0)
                    #We store each value and add it up to the summatory for the specific action            
                    setof_actions_values.append(value_s)
                #We put in the value estimation matrix the maximum value.
                value_est[i][j]= np.max(setof_actions_values)
                #We estimate the difference between previos value function estimation and now. If bigger than epsilon, then we put a 1.
                if (abs(value_est[i][j]-value_est_old[i][j])<epsilon):
                    converge[i][j]=int(0)
                else:
                    converge[i][j]=1
                #We put in the policy matrix the argument that maximizes the value(the action)
                policy[i][j]= np.argmax(setof_actions_values)
                
                #We can also, do it with plicy see the difference, if it changes, in this case we just have to put a random number
                if (abs(policy[i][j]-policy_old[i][j])<epsilon):
                    convergepolicy[i][j]=int(0)
                else:
                    convergepolicy[i][j]=1
        #print(convergepolicy)
        #If all the elements in converge matrix are =, then we are finished (TASK3)
        if np.count_nonzero(converge)==0:
            #print('Converged Value Function')
            #Once the Value Function has converged, then finished=1, and we exit the loop.
            finished=1
        #If want to see when does policy converge, activate following!   
        #if np.count_nonzero(convergepolicy)==0:
            #print('Converged Policy')
        #Clear text after each iteration
        env.clear_text
    
        #Show the values and the policy
        env.draw_values(value_est)
        env.draw_actions(policy)
        env.render()
        sleep(1)

    # Save the state values and the policy
    fnames = "values.npy", "policy.npy"
    np.save(fnames[0], value_est)
    np.save(fnames[1], policy)
    print("Saved state values and policy to", *fnames)

    # Run a single episode
    # TODO: Run multiple episodes and compute the discounted returns (Task 4)
    #Create a list to store all discounted rewards, po will be the power of the eqution, and dreward the discoutned reward
    accrew=[]
    for episode in range(1000):
        done = False
        po=0
        dreward=0
        while not done:
        # Select a random action
        # TODO: Use the policy to take the optimal action (Task 2)
            action = policy[state]

        # Step the environment, we calculate the discounted reward for each action, and we sum with the previous step.
            state, reward, done, _ = env.step(action)
            dreward=reward*0.9**po
            po=po+1
        # Render and sleep
            env.render()
            sleep(0.01)
        #Once an episode is donde(Hitting rocks, Reaching harbour), then append to the list, to further calculate mean and std.deviation
        accrew.append(dreward)
        state = env.reset()
        print(episode)
    meanreward = np.mean(accrew)
    stdeviation= np.std(accrew)
    print(meanreward)
    print(stdeviation)
