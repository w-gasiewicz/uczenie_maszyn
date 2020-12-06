import numpy as np
import strlearn as sl
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

classes = [1, 2, 3]


stream = sl.streams.StreamGenerator(n_chunks=200,
                                    chunk_size=500,
                                    n_classes=2,
                                    n_drifts=1,
                                    n_features=50,
                                    random_state=12345,
                                    weights=[0.1, 0.9]
                                    )


stream_ttt = sl.streams.StreamGenerator(n_chunks=200,
                                    chunk_size=500,
                                    n_classes=2,
                                    n_drifts=1,
                                    n_features=50,
                                    random_state=12345,
                                    weights=[0.1, 0.9]
                                    )


clfs = [
    #sl.ensembles.AWE(GaussianNB(), n_estimators=10),
    sl.ensembles.SEA(GaussianNB(), n_estimators=10),
    sl.ensembles.SEA(GaussianNB(), n_estimators=20),
    #MLPClassifier(hidden_layer_sizes=(50), max_iter= 1000),
    #sl.ensembles.OOB(GaussianNB(), n_estimators=10)
]
clf_names = [
    "SEA",
    "SEA gaus"#,
    #"MLP",
    #"OOB"
]

# Nazwy metryk
metrics_names = ["Recall",
                "BAC",
                "Precision",
                "F1 score",
                "G-mean"]

# Wybrana metryka
metrics = [sl.metrics.recall,
            sl.metrics.balanced_accuracy_score,
            sl.metrics.precision,
            sl.metrics.f1_score,
            sl.metrics.geometric_mean_score_1]

# Inicjalizacja ewaluatora
evaluator = sl.evaluators.Prequential(metrics)
evaluator_ttt = sl.evaluators.TestThenTrain(metrics)

# Uruchomienie
evaluator.process(stream, clfs)
evaluator_ttt.process(stream_ttt, clfs)

# Rysowanie wykresu
fig, ax = plt.subplots(1, len(metrics), figsize=(24, 8))
for m, metric in enumerate(metrics):
    ax[m].set_title(metrics_names[m])
    ax[m].set_ylim(0, 1)
    for i, clf in enumerate(clfs):
        ax[m].plot(evaluator.scores[i, :, m], label=clf_names[i])
        ax[m].plot(evaluator_ttt.scores[i, :, m], label=clf_names[i])
    plt.ylabel("Metric")
    plt.xlabel("Chunk")
    ax[m].legend()
plt.show()