from mlflow import log_metric
from random import choice

metric_names = ["cpu", "ram", "disk"]
percents = [i for i in range(0, 100)]

for i in range(20):
    log_metric(choice(metric_names), choice(percents))
