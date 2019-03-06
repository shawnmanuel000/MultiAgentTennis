# Project Report

[image1]: Rewards.png "Rewards plot"

This report outlines the learning algorithm used to train the agent in the Tennis.ipynb notebook as well as the performance of the agent as it was trained.

### Learning Algorithm

The agent was trained using a multi-agent group of DDPG (Actor Critic) agents each implemented with neural networks. The learning algorithm uses two networks, an actor network for calculating the action to take in the given state, and a critic network for approximates the expected reward, `Q`, from the actor's action, `a`, in a state, `s`, according to the Bellman equation which is given as:

**`Q(s,a) = r + gamma * Q(s')`**

where `r` is the reward received from the action and `s'` is the next state.

At each timestep, the critic is trained first on the experiences to converge its estimates of the actor's value towards the q target of the Bellman equation. The actor is then trained according to the advantage of its predicted action compared to the critic's valuation of that action. Both networks had a similar structure of 3 hidden layers with 100 hidden nodes. Then a learning rate of 0.00025 was chosen to make small steps towards updating the network. Training the network was enhanced with experience replay with a replay buffer having size of 100000 and sampling 50 experience tuples each training step. This was to balance the lower learning rate and remove any temporal correlation between training samples.

Finally, the training process used a noise process to add inertial noise to each agent's actions with a starting noise scale of 1.0 at the start and then decaying at a rate of 0.995 to keep exploring for the first 200 episodes until exploration reached the minimum proportion. This was to ensure that the agent could still benefit from long term exploration.

### Plot of Rewards

The agent was trained for 1500 episodes where the total reward for each episode (blue) and the average reward over the last 100 episodes (orange) is shown in the figure below. The x axis is the number of episodes and the y axis is the score. It can be seen that the agent doesn't increase its score much over the first 1100 episodes, but then it learns how to pass the ball over the net and manages to reach a maximum average reward of 1.5 before becoming unstable and falling in performance.

![Rewards plot][image1]

### Future Work

From the plot, the agent's performance is fairly steady until it nears the maximum reward of 2.65 and then it becomes unstable. For future work, the stability of the agent can be improved by introducing batch normalization to avoid corrupting weights and looking into other more stable actor critic algorithms like Proximal Policy Optimization.