import flwr as fl
import csv

def weighted_average(metrics):
    """根據每個 Client 的資料量計算加權平均 Accuracy"""

    accuracies = [
        num_examples * metrics["accuracy"]
        for num_examples, metrics in metrics
    ]

    examples = [
        num_examples
        for num_examples, _ in metrics
    ]

    return {
        "accuracy": sum(accuracies) / sum(examples)
    }

# build FebAvg Strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,

    min_fit_clients=5,
    min_evaluate_clients=5,
    min_available_clients=5,

    evaluate_metrics_aggregation_fn=weighted_average,
)


if __name__ == "__main__":

    # start Server
    history = fl.server.start_server(
        server_address="127.0.0.1:8080",

        config=fl.server.ServerConfig(
            num_rounds=10
        ),

        strategy=strategy
    )

    # =========================
    # save result
    # =========================

    with open("results.csv", "w", newline="") as f:

        writer = csv.writer(f)

        # CSV 第一列
        writer.writerow([
            "round",
            "loss",
            "accuracy"
        ])

        # 取得 Loss
        loss_history = dict(
            history.losses_distributed
        )

        # 取得 Accuracy
        acc_history = dict(
            history.metrics_distributed["accuracy"]
        )

        # 找出所有 Round
        rounds = sorted(
            set(loss_history.keys()) |
            set(acc_history.keys())
        )

        # 寫入每一輪
        for r in rounds:

            writer.writerow([
                r,
                loss_history.get(r, ""),
                acc_history.get(r, "")
            ])

    print("Results saved to results.csv")