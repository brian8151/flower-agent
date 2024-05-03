import logging
from typing import Optional, Tuple, Dict

from flwr.server.server import Server
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import Strategy, FedAvg
from flwr.common import Parameters, Scalar, EvaluateResultsAndFailures, FitResultsAndFailures, History


class OnyxFlowerServer(Server):
    """Custom Flower server with additional logging."""

    def __init__(
        self,
        client_manager: ClientManager,
        strategy: Optional[Strategy] = None,
    ) -> None:
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.logger = logging.getLogger("OnyxFlowerServer")
        logging.basicConfig(level=logging.INFO)
        self.logger.info("OnyxFlowerServer initialized")

    def fit(self, num_rounds: int, timeout: Optional[float]) -> Tuple[History, float]:
        """Run federated learning for a number of rounds."""
        self.logger.info(f"Starting fit for {num_rounds} rounds with timeout {timeout}")
        result = super().fit(num_rounds, timeout)
        self.logger.info("Fit completed")
        return result

    def evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ]:
        """Evaluate model."""
        self.logger.info(f"Starting evaluation round {server_round} with timeout {timeout}")
        result = super().evaluate_round(server_round, timeout)
        self.logger.info("Evaluation round completed")
        return result

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Fit model for a single round."""
        self.logger.info(f"Starting fit round {server_round} with timeout {timeout}")
        result = super().fit_round(server_round, timeout)
        self.logger.info("Fit round completed")
        return result

    def disconnect_all_clients(self, timeout: Optional[float]) -> None:
        """Disconnect all clients."""
        self.logger.info("Disconnecting all clients")
        super().disconnect_all_clients(timeout)
        self.logger.info("All clients disconnected")