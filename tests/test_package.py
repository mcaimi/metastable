import subprocess
import sys

import pytest

import metastable
from metastable import exceptions


def test_version():
    assert metastable.__version__ == "0.1.0"


@pytest.mark.parametrize("name", [n for n in metastable.__all__ if n != "__version__"])
def test_every_public_name_resolves(name):
    assert getattr(metastable, name) is not None


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        metastable.nope


def test_dir_lists_lazy_exports():
    assert "create_generator" in dir(metastable)
    assert "MergePipeline" in dir(metastable)


def test_lazy_export_is_same_object_as_module_attribute():
    from metastable.pipelines.registry import create_generator
    assert metastable.create_generator is create_generator


def test_import_does_not_load_diffusers():
    code = "import sys, metastable; print('diffusers' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


@pytest.mark.parametrize("exc", [
    exceptions.CheckpointLoadError,
    exceptions.MergeError,
    exceptions.ConfigurationError,
    exceptions.PipelineError,
])
def test_exception_hierarchy(exc):
    assert issubclass(exc, exceptions.MetastableError)
    assert issubclass(exc, Exception)
    assert getattr(metastable, exc.__name__) is exc
