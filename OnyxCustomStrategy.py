from typing import List, Dict, Tuple
import flwr as fl
from flwr.common.typing import Scalar, Config
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.fedavg import FedAvg
from logging import INFO
from flwr.common.logger import log

class OnyxCustomStrategy(FedAvg):


    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        fit_ins_list = super().configure_fit(server_round, parameters, client_manager)

        for client, fit_ins in fit_ins_list:
            client_properties = client.get_properties(dict())
            #print(f"OnyxCustomStrategy [Server] Client ID: {client.cid}, Properties: {client_properties}")
            log(INFO,"OnyxCustomStrategy Client ID (%s) Properties (%s).", client.cid,client_properties)
        return fit_ins_list

    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        evaluate_ins_list = super().configure_evaluate(server_round, parameters, client_manager)

        # Request properties from clients
        for client, evaluate_ins in evaluate_ins_list:
            client_properties = client.get_properties({"round": server_round})
            print(f"OnyxCustomStrategy [Server] Client ID: {client.cid}, Properties: {client_properties}")

        return evaluate_ins_list