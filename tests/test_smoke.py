import bot


def test_package_imports() -> None:
    assert bot.__version__ == "0.1.0"
