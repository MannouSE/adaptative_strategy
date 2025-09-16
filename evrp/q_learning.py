def get_best_action(Q, state, actions):
    if state not in Q or not Q[state]:
        return actions[0]
    return max(Q[state].items(), key=lambda kv: kv[1])[0]

def update(Q, state, action, reward, next_state, alpha, gamma, actions):
    if state not in Q: Q[state] = {}
    if action not in Q[state]: Q[state][action] = 0.0
    next_max = 0.0
    if next_state in Q and Q[next_state]:
        next_max = max(Q[next_state].values())
    Q[state][action] = (1 - alpha) * Q[state][action] + alpha * (reward + gamma * next_max)

def decay_epsilon(eps, eps_min, decay):
    new_eps = eps * decay
    return eps_min if new_eps < eps_min else new_eps
