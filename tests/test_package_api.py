"""Tests for the package's public surface and its PEP 562 lazy loader.

``simpleEMS/__init__.py`` maintains two hand-written structures -- ``__all__``
and ``_NAMES_BY_MODULE`` -- that must agree. Nothing in the code enforces
that, so a name added to one and not the other either becomes unimportable or
silently disappears from ``from simpleEMS import *``. These tests are the
enforcement.
"""

import importlib
import sys

import pytest

import simpleEMS


@pytest.fixture
def isolated_modules():
    """Restore ``sys.modules`` after a test that purges ``simpleEMS`` from it.

    Re-importing a package leaves *new* class objects behind while other test
    modules still hold references to the originals, so ``isinstance`` and
    ``type(a) is type(b)`` start failing for objects that are, by every other
    measure, the same type. Snapshotting and restoring keeps that contained.
    """
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == "simpleEMS" or name.startswith("simpleEMS.")
    }
    try:
        yield
    finally:
        for name in [
            n for n in sys.modules if n == "simpleEMS" or n.startswith("simpleEMS.")
        ]:
            del sys.modules[name]
        sys.modules.update(saved)


class TestNameTables:
    def test_all_and_lazy_table_agree(self):
        """Every exported name must be resolvable, and every resolvable name
        must be exported."""
        assert set(simpleEMS.__all__) == set(simpleEMS._NAME_TO_MODULE)

    def test_all_has_no_duplicates(self):
        assert len(simpleEMS.__all__) == len(set(simpleEMS.__all__))

    def test_lazy_table_is_flattened_correctly(self):
        """``_NAME_TO_MODULE`` is built by flattening ``_NAMES_BY_MODULE``."""
        expected = {
            name: module
            for module, names in simpleEMS._NAMES_BY_MODULE.items()
            for name in names
        }

        assert expected == simpleEMS._NAME_TO_MODULE

    def test_no_name_is_claimed_by_two_modules(self):
        all_names = [
            name for names in simpleEMS._NAMES_BY_MODULE.values() for name in names
        ]

        assert len(all_names) == len(set(all_names))


class TestLazyImport:
    @pytest.mark.needs_csxcad
    @pytest.mark.parametrize("name", sorted(simpleEMS.__all__))
    def test_every_exported_name_resolves(self, name):
        """Catches a name pointed at the wrong module, or one that was
        renamed in its owning module without updating the table."""
        assert getattr(simpleEMS, name) is not None

    @pytest.mark.needs_csxcad
    @pytest.mark.parametrize("name", sorted(simpleEMS.__all__))
    def test_resolved_name_matches_its_owning_module(self, name):
        module_name = simpleEMS._NAME_TO_MODULE[name]
        module = importlib.import_module(f".{module_name}", "simpleEMS")

        assert getattr(simpleEMS, name) is getattr(module, name)

    @pytest.mark.needs_csxcad
    def test_access_caches_into_module_globals(self, monkeypatch):
        """The loader writes the value into ``globals()`` so later lookups
        skip ``__getattr__`` entirely."""
        # an earlier test may already have resolved (and cached) this name
        monkeypatch.delitem(vars(simpleEMS), "MicrostripLine", raising=False)
        assert "MicrostripLine" not in vars(simpleEMS)

        resolved = simpleEMS.MicrostripLine

        assert vars(simpleEMS)["MicrostripLine"] is resolved

    def test_unknown_attribute_raises_attribute_error(self):
        # The name is held in a variable so this reads as a real lookup rather
        # than a discarded attribute expression.
        missing = "NotARealName"

        with pytest.raises(AttributeError, match="has no attribute"):
            getattr(simpleEMS, missing)

    def test_error_message_names_the_package_and_attribute(self):
        missing = "DefinitelyMissing"

        with pytest.raises(AttributeError) as excinfo:
            getattr(simpleEMS, missing)

        message = str(excinfo.value)
        assert "simpleEMS" in message
        assert "DefinitelyMissing" in message

    def test_importing_package_does_not_pull_in_submodules(self, isolated_modules):
        """The whole point of the lazy loader: ``import simpleEMS`` must stay
        cheap and must not drag in gmsh/cadquery/pyvista."""
        for module in list(sys.modules):
            if module.startswith("simpleEMS"):
                del sys.modules[module]

        importlib.import_module("simpleEMS")

        heavy = [
            "simpleEMS.sim_tools",
            "simpleEMS.patch_antenna",
            "simpleEMS.fem_backend",
            "simpleEMS.fdtd_mesh",
        ]
        assert [m for m in heavy if m in sys.modules] == []


class TestMetadata:
    def test_version_matches_packaging_metadata(self):
        """``bump-my-version`` rewrites the version in two files; they drift
        if one of the ``[[tool.bumpversion.files]]`` entries is wrong."""
        from importlib.metadata import PackageNotFoundError, version

        try:
            installed = version("simpleEMS")
        except PackageNotFoundError:
            pytest.skip("simpleEMS is not installed as a distribution")

        assert simpleEMS.__version__ == installed

    def test_version_is_pep440_triple(self):
        parts = simpleEMS.__version__.split(".")

        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)

    def test_declares_author_and_licence(self):
        assert simpleEMS.__author__
        assert simpleEMS.__license__ == "AGPL-3.0-or-later"
