import flwr as fl


# =========================
# 1. Evaluation
# =========================

def evaluate(
    server_round,
    parameters,
    config
):

    return 0.0, {
        "accuracy": 0.0
    }


# =========================
# 2. FedAvg Strategy
# =========================

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,

    min_fit_clients=5,
    min_evaluate_clients=5,
    min_available_clients=5,

    evaluate_fn=evaluate,
)


# =========================
# 3. Start Server
# =========================

if __name__ == "__main__":

    fl.server.start_server(
        server_address="127.0.0.1:8080",

        config=fl.server.ServerConfig(
            num_rounds=10
        ),

        strategy=strategy
    )
