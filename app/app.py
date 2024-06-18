from DrillExperiment import DrillExperiment


def drillExperiment():
    experiment = DrillExperiment()
    experiment.eval_perf(10)
    experiment.print_metrics()
    
    experiment.train(10000)
    experiment.plot_training_data()

    experiment.eval_perf(10)
    experiment.print_metrics()
    experiment.print_dataset(100)





if __name__ == "__main__":
    drillExperiment()
