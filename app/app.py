from DrillExperiment import DrillExperiment


def drillExperiment():
    #logger.info("Initial evaluation")
    experiment = DrillExperiment()
    experiment.eval_perf(n_episodes=1)
    experiment.print_metrics()
    #experiment.print_dataset(100)

    #experiment.train(1000)
    experiment.train(10000)
    experiment.plot_training_data()

    experiment.eval_perf(1)
    experiment.print_metrics()
    #experiment.print_dataset(100)





if __name__ == "__main__":
    drillExperiment()
