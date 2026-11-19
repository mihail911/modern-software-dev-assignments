import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dispatcher import signing  # noqa: E402


class SigningTest(unittest.TestCase):
    def test_known_value(self):
        body = b"The quick brown fox jumps over the lazy dog"
        self.assertEqual(
            signing.sign("key", body),
            "sha256=f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8",
        )

    def test_verify_accepts_own_signature(self):
        body = b'{"id":"evt_1","amount":"12.50"}'
        self.assertTrue(signing.verify("s3cret", body, signing.sign("s3cret", body)))

    def test_verify_rejects_anything_else(self):
        body = b'{"id":"evt_1","amount":"12.50"}'
        header = signing.sign("s3cret", body)
        self.assertFalse(signing.verify("s3cret", body.replace(b"12.50", b"92.50"), header))
        self.assertFalse(signing.verify("other", body, header))
        self.assertFalse(signing.verify("s3cret", body, header[len("sha256="):]))
        self.assertFalse(signing.verify("s3cret", body, None))


if __name__ == "__main__":
    unittest.main()
