import os
import unittest
from unittest import mock

from cryosparc_client import ensure_cryosparc_no_proxy


class TestCryoSPARCClientProxyBypass(unittest.TestCase):
    def test_proxy_bypass_preserves_existing_entries_and_adds_cryosparc_hosts(self):
        with mock.patch.dict(os.environ, {"NO_PROXY": "example.com", "no_proxy": "intranet.local"}, clear=False):
            ensure_cryosparc_no_proxy("127.0.0.1")

            no_proxy = os.environ["NO_PROXY"].split(",")
            lower_no_proxy = os.environ["no_proxy"].split(",")

            self.assertIn("example.com", no_proxy)
            self.assertIn("intranet.local", lower_no_proxy)
            self.assertIn("127.0.0.1", no_proxy)
            self.assertIn("localhost", no_proxy)
            self.assertIn("admin", lower_no_proxy)
            self.assertIn("172.16.1.2", lower_no_proxy)


if __name__ == "__main__":
    unittest.main()
