import ntplib
import socket

# --- Subsystem Classes ---

class _ServerPoolManager:
    def __init__(self, custom_server_ip: str = None):
        if custom_server_ip:
            self._servers = [custom_server_ip]
        else:
            self._servers = ["0.pool.ntp.org", "1.pool.ntp.org", "2.pool.ntp.org"]
        self._server_count = len(self._servers)

    def get_servers(self):
        return self._servers

class _NtpClient:
    """A wrapper for the ntplib client."""
    def __init__(self, timeout=2):
        self._ntp_client = ntplib.NTPClient()
        self._timeout = timeout

    def fetch_time(self, server: str, port: int) -> float:
        """Requests time from a single NTP server on a specific port."""
        try:
            response = self._ntp_client.request(server, port=port, version=3, timeout=self._timeout)
            return response.tx_time
        except Exception as e:
            raise IOError(f"Failed to get time from {server}:{port}") from e

# --- The Public Facade Class ---

class TimeBrokerFacade:
    """Provides a simple, unified interface to get synchronized time."""
    def __init__(self, ntp_server_ip: str, port: int = 123):
        """
        Initializes the facade.
        
        Args:
            ntp_server_ip (str): The IP address or hostname of the NTP server.
            port (int, optional): The server port. Defaults to 123 (standard NTP).
        """
        if not ntp_server_ip:
            raise ValueError("An ntp_server_ip must be provided.")
        
        self._pool_manager = _ServerPoolManager(custom_server_ip=ntp_server_ip)
        self._ntp_client = _NtpClient()
        self._port = port

    def get_synchronized_time(self) -> float:
        """
        Gets synchronized time from the configured server.
        """
        server = self._pool_manager.get_servers()[0]
        print(f"--> [Facade] Attempting to sync with '{server}:{self._port}'...")
        try:
            timestamp = self._ntp_client.fetch_time(server, port=self._port)
            print(f"Success! Received time from '{server}'.")
            return timestamp
        except (socket.gaierror, ntplib.NTPException, socket.timeout, IOError) as e:
            print(f"Failed to connect: {e}")
            raise IOError(f"Critical: Could not get time from NTP server '{server}'.") from e
