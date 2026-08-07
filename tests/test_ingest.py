import gzip
import unittest
import zlib

from src.ingestion.ingest import SecRequestError, _decode_response_body


class DecodeResponseBodyTests(unittest.TestCase):
    def test_decodes_gzip_json(self) -> None:
        payload = b'{"0":{"ticker":"AAPL"}}'

        self.assertEqual(_decode_response_body(gzip.compress(payload), "gzip", "utf-8"), payload.decode())

    def test_decodes_zlib_and_raw_deflate_json(self) -> None:
        payload = b'{"0":{"ticker":"AAPL"}}'
        raw_deflate = zlib.compressobj(wbits=-zlib.MAX_WBITS)

        self.assertEqual(_decode_response_body(zlib.compress(payload), "deflate", "utf-8"), payload.decode())
        self.assertEqual(
            _decode_response_body(raw_deflate.compress(payload) + raw_deflate.flush(), "deflate", "utf-8"),
            payload.decode(),
        )

    def test_rejects_unsupported_content_encoding(self) -> None:
        with self.assertRaisesRegex(SecRequestError, "unsupported Content-Encoding"):
            _decode_response_body(b"{}", "br", "utf-8")


if __name__ == "__main__":
    unittest.main()
