from pathlib import Path

from map_builder.project import ImageIndexer, ProjectStore


def test_image_indexer_adds_updates_and_marks_missing(tmp_path: Path) -> None:
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.png"
    ignored = tmp_path / "notes.txt"
    first.write_bytes(b"not a real image yet")
    second.write_bytes(b"also not a real image")
    ignored.write_text("ignore me")

    store = ProjectStore.open(tmp_path)
    try:
        indexer = ImageIndexer(read_dimensions=False)
        summary = indexer.index_folder(tmp_path, store)
        assert summary.num_found == 2
        assert summary.num_new == 2
        assert {image.rel_path for image in store.list_images()} == {"first.jpg", "second.png"}

        image = store.list_images()[0]
        store.set_image_ignored(image.id, True)
        first.write_bytes(b"changed")
        second.unlink()

        summary = indexer.index_folder(tmp_path, store)
        assert summary.num_found == 1
        assert summary.num_updated == 1
        assert summary.num_missing == 1

        by_path = {image.rel_path: image for image in store.list_images()}
        assert by_path["first.jpg"].missing is False
        assert by_path["second.png"].missing is True
        assert store.get_image(image.id).ignored is True  # type: ignore[union-attr]
    finally:
        store.close()


def test_image_indexer_marks_missing_image_present_when_reappears(tmp_path: Path) -> None:
    image_path = tmp_path / "image.bmp"
    image_path.write_bytes(b"initial")
    store = ProjectStore.open(tmp_path)
    try:
        indexer = ImageIndexer(read_dimensions=False)
        indexer.index_folder(tmp_path, store)
        image_path.unlink()
        indexer.index_folder(tmp_path, store)
        assert store.list_images()[0].missing is True

        image_path.write_bytes(b"returned")
        indexer.index_folder(tmp_path, store)
        assert store.list_images()[0].missing is False
    finally:
        store.close()
