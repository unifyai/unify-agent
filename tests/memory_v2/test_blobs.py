from unify.memory_v2.blobs import BlobStore


def test_put_get_roundtrip_and_idempotent(tmp_path):
    b = BlobStore(tmp_path / "blobs")
    s1 = b.put(b"hello")
    s2 = b.put(b"hello")
    assert s1 == s2 and len(s1) == 64 and b.get(s1) == b"hello"


def test_cap_text_small_and_large(tmp_path):
    b = BlobStore(tmp_path / "blobs")
    assert b.cap_text("abc", 10) == {"text": "abc"}
    big = "é" * 100
    out = b.cap_text(big, 20)
    assert out["bytes"] == len(big.encode()) and b.get(out["blob"]).decode() == big
    assert len(out["excerpt"].encode()) <= 20
