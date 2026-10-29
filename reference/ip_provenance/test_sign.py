"""Detached owner-key signing tests; transient keys only, never committed."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from reference.ip_provenance.freeze import canonical_bytes
from reference.ip_provenance.sign import sign_detached, verify_detached


class TestIPOwnerSignature(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.manifest = self.dir / "manifest.json"
        self.manifest.write_bytes(b'{"example":"frozen"}\n')
        self.priv = self.dir / "offline-owner-private.pem"
        self.pub = self.dir / "independently-trusted-public.pem"
        key = Ed25519PrivateKey.generate()
        self.priv.write_bytes(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ))
        self.pub.write_bytes(key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        ))

    def test_valid_owner_signature_and_wrong_identity_rejected(self):
        # Mock only the costly Git inventory/anchor verification; the crypto
        # signature is real Ed25519 over the canonical exact manifest bytes.
        with patch("reference.ip_provenance.sign._verified_manifest",
                   return_value=self.manifest.read_bytes()):
            receipt = sign_detached(self.manifest, self.priv)
            path = self.dir / "signature.json"
            path.write_bytes(canonical_bytes(receipt))
            self.assertTrue(verify_detached(self.manifest, path, self.pub))
            # An unrelated public key must not verify the attestation.
            wrong = Ed25519PrivateKey.generate().public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            wrong_path = self.dir / "wrong-public.pem"
            wrong_path.write_bytes(wrong)
            with self.assertRaises(ValueError):
                verify_detached(self.manifest, path, wrong_path)

    def test_signed_manifest_mutation_and_signature_tamper_rejected(self):
        signed = self.manifest.read_bytes()
        with patch("reference.ip_provenance.sign._verified_manifest", side_effect=lambda path: path.read_bytes()):
            receipt = sign_detached(self.manifest, self.priv)
            path = self.dir / "signature.json"
            path.write_bytes(canonical_bytes(receipt))
            self.assertTrue(verify_detached(self.manifest, path, self.pub))
            self.manifest.write_bytes(b'{"example":"modified"}\n')
            with self.assertRaises(ValueError):
                verify_detached(self.manifest, path, self.pub)
            self.manifest.write_bytes(signed)
            receipt["signature_base64"] = "AAAA"
            path.write_bytes(canonical_bytes(receipt))
            with self.assertRaises(ValueError):
                verify_detached(self.manifest, path, self.pub)


if __name__ == "__main__":
    unittest.main()
