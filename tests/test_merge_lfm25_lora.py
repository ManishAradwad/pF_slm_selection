from scripts.merge_lfm25_lora import _tree_hash


def test_tree_hash_can_exclude_existing_merge_manifest(tmp_path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "merge_manifest.json").write_text("old manifest", encoding="utf-8")
    nested = tmp_path / "tokenizer"
    nested.mkdir()
    (nested / "tokenizer.json").write_text("{}", encoding="utf-8")

    hashes = _tree_hash(
        tmp_path,
        excluded_relative_paths=frozenset({"merge_manifest.json"}),
    )

    assert set(hashes) == {"model.safetensors", "tokenizer/tokenizer.json"}
    assert hashes == _tree_hash(
        tmp_path,
        excluded_relative_paths=frozenset({"merge_manifest.json"}),
    )
