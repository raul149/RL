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
env = SailingGridworld(rock_penalty=-10)
value_est = np.zeros((env.w, env.h))
env.draw_values(value_est)


if __name__ == "__main__":
    # Reset the environment
    state = env.reset()

    # Compute state values and the policy
    # TODO: Compute the value function and policy (Tasks 1, 2 and 3)
    value_est, policy = np.zeros((env.w, env.h)), np.zeros((env.w, env.h))
    iteration=0
    converge=np.zeros((env.w, env.h))
    index=0
    while iteration<100:
    #Calculate for each action the value
        i=0
        j=0
        value_estlive=value_est
        while j<env.h:
            while i<env.w:           
                setof_actions_values=[]
                actionstaken=[]
                for Transition in env.transitions[i,j]:
                    value_s=0
                    for nstate,reward,done,prob in Transition:
                            #print (i,j)
                            #print(prob)
                            #print(reward)
                            #print (value_est[nstate])
                            if not done:
                                value_s=value_s+prob*(reward+0.9*value_est[nstate])
                            else:
                                value_s=value_s+prob*(reward+0.9*0)
                            #print('value_s')
                        #We store each value and add it up to the summatory for the specific action            
                    setof_actions_values.append(value_s)
                    actionstaken.append(env.transitions[i,j])
                    #print(arg.Transition)
                    
                    #print(set_actions_values)
                value_estlive[i][j]= np.max(setof_actions_values)
                if abs(value_estlive[i][j]-value_est[i][j])<epsilon:
                    converge[i][j]=0
                else:
                    converge[i][j]=1
                #index=setof_actions_values.index(value_estlive[i][j])
                #print(index)
                policy[i][j]= np.argmax(setof_actions_values)
                i=i+1
            i=0
            j=j+1
        if np.count_nonzero(policy)==0:
            print('Converged')
            print(iteration+1)
        #value_est=value_estlive
        print()
        print(iteration)
        iteration=iteration+1
        #print(iteration)
        #print(value_est)
    
        #Clear text after each iteration
    env.clear_text
    
        #Show the values and the policy
    env.draw_values(value_est)
    print(policy)
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
    done = False
    while not done:
        # Select a random action
        # TODO: Use the policy to take the optimal action (Task 2)
        action = policy[state]

        # Step the environment
        state, reward, done, _ = env.step(action)

        # Render and sleep
        env.render()
        sleep(0.5)

