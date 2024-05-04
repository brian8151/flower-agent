from flwr.common.logger import log
from flwr.server.client_proxy  import ClientProxy
from flwr.server.criterion  import Criterion
from flwr.server.client_manager  import ClientManager
import threading
from typing import Dict, List, Optional
from logging import INFO
from flwr.common.logger import log
import random
class OnyxClientManager(ClientManager):
    """Provides a pool of available clients with custom logging."""

    def __init__(self) -> None:
        self.clients: Dict[str, ClientProxy] = {}
        self._cv = threading.Condition()

    def __len__(self) -> int:
        """Return the number of available clients.

        Returns
        -------
        num_available : int
            The number of currently available clients.
        """
        return len(self.clients)

    def num_available(self) -> int:
        """Return the number of available clients."""
        num_clients = len(self)
        log(INFO, "Number of available ONYX clients: (%s).", num_clients)
        for cid, client in self.clients.items():
            client_info = f"GrpcClientProxy - CID: {cid}, Node ID: {client.node_id}, Bridge Info: {client.bridge}"
            log(INFO, "Client Info: %s", client_info)
        return num_clients
    def num_available(self) -> int:
        """Return the number of available clients.

        Returns
        -------
        num_available : int
            The number of currently available clients.
        """
        num_clients = len(self)
        log(INFO, "Number of available ONYX clients: (%s).", num_clients)
        for cid, client in self.clients.items():
            log(INFO, "Client ID (%s), Client Info: (%s).", cid, client)
        return num_clients

    def wait_for(self, num_clients: int, timeout: int = 86400) -> bool:
        """Wait until at least `num_clients` are available.

        Blocks until the requested number of clients is available or until a
        timeout is reached. Current timeout default: 1 day.

        Parameters
        ----------
        num_clients : int
            The number of clients to wait for.
        timeout : int
            The time in seconds to wait for, defaults to 86400 (24h).

        Returns
        -------
        success : bool
        """
        with self._cv:
            return self._cv.wait_for(
                lambda: len(self.clients) >= num_clients, timeout=timeout
            )

    def register(self, client: ClientProxy) -> bool:
        """Register Flower ClientProxy instance.

        Parameters
        ----------
        client : flwr.server.client_proxy.ClientProxy

        Returns
        -------
        success : bool
            Indicating if registration was successful. False if ClientProxy is
            already registered or can not be registered for any reason.
        """
        log(INFO, "[ONYX Client register]")
        if client.cid in self.clients:
            return False

        self.clients[client.cid] = client
        with self._cv:
            self._cv.notify_all()

        return True

    def unregister(self, client: ClientProxy) -> None:
        """Unregister Flower ClientProxy instance.

        This method is idempotent.

        Parameters
        ----------
        client : flwr.server.client_proxy.ClientProxy
        """
        if client.cid in self.clients:
            del self.clients[client.cid]

            with self._cv:
                self._cv.notify_all()

    def all(self) -> Dict[str, ClientProxy]:
        """Return all available clients."""
        return self.clients

    def sample(
            self,
            num_clients: int,
            min_num_clients: Optional[int] = None,
            criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        log(INFO, "Sample a number of ONYX Flower ClientProxy instances  (%s).", num_clients)
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]

        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []

        sampled_cids = random.sample(available_cids, num_clients)
        return [self.clients[cid] for cid in sampled_cids]