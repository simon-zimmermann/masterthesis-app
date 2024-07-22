from DrillExperimentQ import DrillExperimentQ


def drillExperimentTrain():
    experiment = DrillExperimentQ()

    experiment.eval_performance(print_demo=False)
    experiment.train(n_episodes=10000, print_plot=True)
    experiment.eval_performance(print_demo=True)

    experiment.save_agent("agent1")


def drillExperimentLoad():
    experiment = DrillExperimentQ("agent1")
    experiment.eval_performance(print_demo=True)


if __name__ == "__main__":
    drillExperimentTrain()
    # drillExperimentLoad()
