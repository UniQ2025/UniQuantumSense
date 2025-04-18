from cryptography.fernet import Fernet
import unittest

class TestEncryption(unittest.TestCase):
    def setUp(self):
        self.key = Fernet.generate_key()
        self.fernet = Fernet(self.key)

    def test_encryption_decryption(self):
        data = b"Sensitive data"
        encrypted = self.fernet.encrypt(data)
        decrypted = self.fernet.decrypt(encrypted)
        self.assertEqual(data, decrypted)

    def test_invalid_decryption(self):
        with self.assertRaises(Exception):
            self.fernet.decrypt(b"invalid_encrypted_data")

if __name__ == '__main__':
    unittest.main()                   