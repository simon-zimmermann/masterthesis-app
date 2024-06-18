from DrillExperimentQ import DrillExperimentQ


def drillExperiment():
    experiment = DrillExperimentQ()

    experiment.eval_performance(print_demo=False)
    experiment.train(n_episodes=50000, print_plot=True)
    experiment.eval_performance(print_demo=True)





if __name__ == "__main__":
    drillExperiment()
