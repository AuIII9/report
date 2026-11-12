class QLearningAgent:
    def __init__(self, states, actions, gamma=0.9, alpha=0.5, epsilon=0.3,
                 epsilon_decay=0.995, epsilon_min=0.05, seed=None):
        self.states, self.actions = list(states), list(actions)
        self.gamma, self.alpha = gamma, alpha
        self.epsilon, self.epsilon_decay, self.epsilon_min = epsilon, epsilon_decay, epsilon_min
        self.q = {(s,a): 0.0 for s in states for a in actions}

    def learn(self, episodes, mode='fixed'):
        for episode in episodes:
            for transition in episode:
                self.update(*transition)

    def update(self, state, action, reward, next_state, done):
        target = reward if done else reward + self.gamma*max(self.q[next_state,a] for a in self.actions)
        self.q[state,action] += self.alpha*(target-self.q[state,action])

    def get_q_value(self, state, action):
        return float(self.q[state,action])

    def get_policy(self, state):
        order = [a for a in ['w','g','i1','i2'] if a in self.actions]
        order += [a for a in self.actions if a not in order]
        return max(order, key=lambda a: self.q[state,a])
