import pytest

from determined.experimental import Determined, ModelSortBy
from tests import config as conf
from tests import experiment as exp
from tests.cluster.test_users import log_out_user


@pytest.mark.e2e_cpu
def test_model_registry() -> None:
    exp_id = exp.run_basic_test(
        conf.fixtures_path("mnist_pytorch/const-pytorch11.yaml"),
        conf.tutorials_path("mnist_pytorch"),
        None,
    )

    log_out_user()  # Ensure that we use determined credentials.

    d = Determined(conf.make_master_url())
    mnist = None
    objectdetect = None
    tform = None

    existing_models = [m.name for m in d.get_models(sort_by=ModelSortBy.NAME)]

    try:
        # Create a model and validate twiddling the metadata.
        mnist = d.create_model("mnist", "simple computer vision model", labels=["a", "b"])
        assert mnist.metadata == {}

        mnist.add_metadata({"testing": "metadata"})
        db_model = d.get_model(mnist.name)
        # Make sure the model metadata is correct and correctly saved to the db.
        assert mnist.metadata == db_model.metadata
        assert mnist.metadata == {"testing": "metadata"}

        # Confirm we can look up a model by its ID
        db_model = d.get_model_by_id(mnist.model_id)
        assert db_model.name == "mnist"
        db_model = d.get_model(mnist.model_id)
        assert db_model.name == "mnist"

        # Confirm DB assigned username
        assert db_model.username == "determined"

        mnist.add_metadata({"some_key": "some_value"})
        db_model = d.get_model(mnist.name)
        assert mnist.metadata == db_model.metadata
        assert mnist.metadata == {"testing": "metadata", "some_key": "some_value"}

        mnist.add_metadata({"testing": "override"})
        db_model = d.get_model(mnist.name)
        assert mnist.metadata == db_model.metadata
        assert mnist.metadata == {"testing": "override", "some_key": "some_value"}

        mnist.remove_metadata(["some_key"])
        db_model = d.get_model(mnist.name)
        assert mnist.metadata == db_model.metadata
        assert mnist.metadata == {"testing": "override"}

        mnist.set_labels(["hello", "world"])
        db_model = d.get_model(mnist.name)
        assert mnist.labels == db_model.labels
        assert db_model.labels == ["hello", "world"]

        # confirm patch does not overwrite other fields
        mnist.set_description("abcde")
        db_model = d.get_model(mnist.name)
        assert db_model.metadata == {"testing": "override"}
        assert db_model.labels == ["hello", "world"]

        # overwrite labels to empty list
        mnist.set_labels([])
        db_model = d.get_model(mnist.name)
        assert db_model.labels == []

        # archive and unarchive
        assert mnist.archived is False
        mnist.archive()
        db_model = d.get_model(mnist.name)
        assert db_model.archived is True
        mnist.unarchive()
        db_model = d.get_model(mnist.name)
        assert db_model.archived is False

        # Register a version for the model and validate the latest.
        checkpoint = d.get_experiment(exp_id).top_checkpoint()
        model_version = mnist.register_version(checkpoint.uuid)
        assert model_version.model_version == 1

        latest_version = mnist.get_version()
        assert latest_version is not None
        assert latest_version.checkpoint.uuid == checkpoint.uuid

        latest_version.set_name("Test 2021")
        db_version = mnist.get_version()
        assert db_version is not None
        assert db_version.name == "Test 2021"

        latest_version.set_notes("# Hello Markdown")
        db_version = mnist.get_version()
        assert db_version is not None
        assert db_version.notes == "# Hello Markdown"

        # Run another basic test and register its checkpoint as a version as well.
        # Validate the latest has been updated.
        exp_id = exp.run_basic_test(
            conf.fixtures_path("mnist_pytorch/const-pytorch11.yaml"),
            conf.tutorials_path("mnist_pytorch"),
            None,
        )
        checkpoint = d.get_experiment(exp_id).top_checkpoint()
        model_version = mnist.register_version(checkpoint.uuid)
        assert model_version.model_version == 2

        latest_version = mnist.get_version()
        assert latest_version is not None
        assert latest_version.checkpoint.uuid == checkpoint.uuid

        # Ensure the correct number of versions are present.
        all_versions = mnist.get_versions()
        assert len(all_versions) == 2

        for v in all_versions:
            mv = mnist.get_version(v.model_version)
            assert mv is not None
            assert mv.model_version == v.model_version

        # Test deletion of model version
        latest_version.delete()
        all_versions = mnist.get_versions()
        assert len(all_versions) == 1

        # Create some more models and validate listing models.
        tform = d.create_model("transformer", "all you need is attention")
        objectdetect = d.create_model("ac - Dc", "a test name model")

        models = d.get_models(sort_by=ModelSortBy.NAME)
        model_names = [m.name for m in models if m.name not in existing_models]
        assert model_names == ["ac - Dc", "mnist", "transformer"]

        # Test model labels combined
        mnist.set_labels(["hello", "world"])
        tform.set_labels(["world", "test", "zebra"])
        labels = d.get_model_labels()
        assert labels == ["world", "hello", "test", "zebra"]

        # Test deletion of model
        tform.delete()
        tform = None
        models = d.get_models(sort_by=ModelSortBy.NAME)
        model_names = [m.name for m in models if m.name not in existing_models]
        assert model_names == ["ac - Dc", "mnist"]
    finally:
        # Clean model registry of test models
        for model in [mnist, objectdetect, tform]:
            if model is not None:
                model.delete()
