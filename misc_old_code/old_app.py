def print_env_status(drill: Drill, workplan: Workplan, last_action: Action, reward: float):
    str = "life: %d,\taction: %s,\treward: %.02f,\tworkplan: %s" % (
        drill.remainingLife, last_action, reward, workplan.pretty_print())
    print(str)


def main():
    MAX_CHANGES = 10
    print("Running simulation for a maximum of %d drill changes" % MAX_CHANGES)
    random.seed()  # Make sure random numbers are actually random
    workplan = Workplan()
    drill = Drill()
    agent = Agent()
    cum_reward = 0

    i = 0  # Simualtion iteration counter
    while i < MAX_CHANGES:
        # Determine the next action
        action = agent.act(drill, workplan)
        # Calculate reward
        reward = 0
        if drill.remainingLife <= 0:
            reward = -100
            drill.change_drillbit()  # Emergency change
        elif action == Action.DRILL or action == Action.SKIP:
            reward = 1
        elif action == Action.CHANGE:
            holes_curr = workplan.next().num_holes
            reward = -0.1 * drill.remainingLife - holes_curr
        cum_reward += reward
        # Print status
        print_env_status(drill, workplan, action, reward)
        # Perform the action
        match action:
            case Action.CHANGE:
                drill.change_drillbit()
                i += 1
            case Action.SKIP:
                workplan.pop()
                pass
            case Action.DRILL:
                drill.do_drill_action(workplan.pop())

    print("Simulation ended after %d drill changes" % MAX_CHANGES)
    print("Cumulative reward: %.02f" % cum_reward)

