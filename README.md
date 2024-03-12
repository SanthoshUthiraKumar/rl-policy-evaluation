# POLICY EVALUATION

## AIM
To develop a Python program to evaluate the given policy by maximizing its cumulative reward while dealing with slippery terrain.

## PROBLEM STATEMENT
The Bandit Slippery Walk problem is a Reinforcement Learning (RL) problem in which the agent must learn to navigate a slippery environment to reach the goal state.

1. we are tasked with creating an RL agent to solve the "Bandit Slippery Walk" problem.

2. The environment consists of Seven states representing discrete positions the agent can occupy.

3. The agent must learn to navigate this environment while dealing with the challenge of slippery terrain.

4. Slippery terrain introduces stochasticity in the agent's actions, making it difficult to predict the outcomes of its actions accurately.

## STATE
The environment has 7 states:

Two Terminal States: G: The goal state & H: A hole state.Five Transition states / Non-terminal States including S: The starting state.

## Actions
The agent can take two actions: R (move right) and L (move left). 

The transition probabilities for each action are as follows:

50% chance that the agent moves in the intended direction.
33.33% chance that the agent stays in its current state.
16.66% chance that the agent moves in the opposite direction.
For example, if the agent is in state S and takes the "R" action, then there is a 50% chance that it will move to state 4, a 33.33% chance that it will stay in state S, and a 16.66% chance that it will move to state 2.

## REWARD
The agent receives a reward of +1 for reaching the goal state and a reward of 0 for all other states.

## GRAPHICAL REPRESENTATION
![Graph](https://github.com/SanthoshUthiraKumar/rl-policy-evaluation/assets/119477975/d873dc35-2aa6-4507-af81-2f2dc6081d1f)

## FORMULA
![Formula](https://github.com/SanthoshUthiraKumar/rl-policy-evaluation/assets/119477975/0ee3a958-ad5f-4767-9dd1-820ddee90651)

## POLICY EVALUATION FUNCTION
```
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)
    # Write your code here to evaluate the given policy
    while True:
      V = np.zeros(len(P))
      for s in range(len(P)):
        for prob, next_state, reward, done in P[s][pi(s)]:
          V[s] += prob * (reward + gamma *  prev_V[next_state] * (not done))
      if np.max(np.abs(prev_V - V)) < theta:
        break
      prev_V = V.copy()
    return V

# Code to evaluate the first policy
V1 = policy_evaluation(pi_1, P)
print_state_value_function(V1, P, n_cols=7, prec=5)

# Code to evaluate the second policy
V2 = policy_evaluation(pi_2, P)
print_state_value_function(V2, P, n_cols=7, prec=5)

# Comparing the two policies
# Compare the two policies based on the value function using the above equation and find the best policy

V1

V2

V1>=V2

if(np.sum(V1>=V2)==7):
  print("The first policy has the better policy")
elif(np.sum(V2>=V1)==7):
  print("The second policy has the better policy")
else:
  print("Both policies have their merits.")
```
## OUTPUT:
### POLICY 1
Policy 1
![269647740-21e5577c-7484-4877-9fab-de8f5c558c8c](https://github.com/SanthoshUthiraKumar/rl-policy-evaluation/assets/119477975/dc4fde19-3e21-4f52-89d2-28d2ef9a5242)
State Value Function
![269648127-21b6c0e1-df38-49f7-b7ca-532c46134365](https://github.com/SanthoshUthiraKumar/rl-policy-evaluation/assets/119477975/434d5672-01f1-4704-ac7e-ff1fe4d81772)
Evaluation
![269648443-69411a03-182b-4a06-a7c4-560f4b49e2ea](https://github.com/SanthoshUthiraKumar/rl-policy-evaluation/assets/119477975/9ee23523-cb1d-403c-98cb-f71e2207938c)

### POLICY 2
Policy 2
![269648607-e618b051-422e-4569-8096-d149f6b7a9b0](https://github.com/SanthoshUthiraKumar/rl-policy-evaluation/assets/119477975/5de398e4-9e11-488e-9b4d-39ac91632d42)
State Value Function
![269648699-12fc42db-8f6f-44e1-9900-5640bc1b759e](https://github.com/SanthoshUthiraKumar/rl-policy-evaluation/assets/119477975/a0033b1d-5e70-49c6-8fd0-557b2b8a7336)
Evaluation
![269648870-c26a46e3-9d89-469c-a3f1-0d31f32919cd](https://github.com/SanthoshUthiraKumar/rl-policy-evaluation/assets/119477975/dc4c9944-151a-4691-81aa-9336852daba9)

### COMPARISON
![269649656-fa9e0edf-65d6-4b7b-9efa-1fe0ccc64de3](https://github.com/SanthoshUthiraKumar/rl-policy-evaluation/assets/119477975/6c913c25-4491-43c4-8c06-5ccecad72e6f)

### CONCLUSION
![269649736-f517b4bf-e39b-4265-bc26-876fcbf1cabf](https://github.com/SanthoshUthiraKumar/rl-policy-evaluation/assets/119477975/d942eb6d-4a86-496a-aebe-f3a61a2f776c)

## RESULT:
Thus, This program will evaluate the given policy in the Bandit Slippery Walk environment and predict the expected reward of the policy.
