from pathlib import Path

import pylake

from examples.generate_ctd_examples import main as generate_examples


def test_generated_ctd_files_are_readable(tmp_path):
    directory = generate_examples(tmp_path)
    expected = {
        "example.rsk": ("RBR", 20),
        "example.tob": ("Sea & Sun", 20),
        "example_datalakes.json": ("DataLakes", 3),
        "example_kor.csv": ("KOR", 20),
    }

    for name, (source, time_size) in expected.items():
        path = Path(directory) / name
        dataset = pylake.read(path)
        assert dataset.attrs["source"] == source
        assert dataset.sizes["time"] == time_size
