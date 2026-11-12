class ModelBasedAgent:
    def __init__(self, states, actions, gamma=0.9, seed=None):
        self.states, self.actions, self.gamma = list(states), list(actions), gamma
        self.learn_model([])

    def learn_model(self, episodes):
        self.counts = {(s,a): 0 for s in self.states for a in self.actions}
        self.p = {k: dict.fromkeys(self.states, 0.0) for k in self.counts}
        self.r = dict.fromkeys(self.counts, 0.0)
        for episode in episodes:
            for s,a,r,ns,done in episode:
                self.counts[s,a] += 1
                self.p[s,a][ns] += 1
                self.r[s,a] += r
        for k,n in self.counts.items():
            self.p[k] = {s: c/n if n else 1/len(self.states) for s,c in self.p[k].items()}
            self.r[k] = self.r[k]/n if n else 0.0
        self.v = dict.fromkeys(self.states, 0.0)

    def _q(self, s, a):
        # Entering B1 ends the episode; reset may nevertheless start at B1.
        return self.r[s,a] + self.gamma * sum(p*self.v[ns] for ns,p in self.p[s,a].items() if ns != 'B1')

    def run_value_iteration(self, theta=1e-4, max_iterations=1000):
        for i in range(max_iterations):
            new = {s: max(self._q(s,a) for a in self.actions) for s in self.states}
            delta = max(abs(new[s]-self.v[s]) for s in self.states)
            self.v = new
            self.iterations = i+1
            if delta < theta:
                break

    def get_transition(self, state, action):
        return dict(self.p[state,action])

    def get_reward(self, state, action):
        return float(self.r[state,action])

    def get_value(self, state):
        return float(self.v[state])

    def get_policy(self, state):
        order = [a for a in ['w','g','i1','i2'] if a in self.actions]
        order += [a for a in self.actions if a not in order]
        return max(order, key=lambda a: self._q(state,a))
