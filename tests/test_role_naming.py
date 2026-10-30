"""Role names expose the existing modules without changing checkpoint keys."""
import torch

from jets import JETSGenerator, TimeTargetEncoder, FeatureTargetEncoder
from jets.teachers import TimeTeacher, FeatureTeacher
from jets.model import JETSGenerator as LegacyJETSGenerator


def test_target_encoder_names_are_backward_compatible():
    assert TimeTeacher is TimeTargetEncoder
    assert FeatureTeacher is FeatureTargetEncoder
    assert LegacyJETSGenerator is JETSGenerator
    assert TimeTargetEncoder.__name__ == "TimeTargetEncoder"
    assert FeatureTargetEncoder.__name__ == "FeatureTargetEncoder"


def test_context_and_predictor_properties_do_not_change_state_dict():
    model = JETSGenerator(5, 2, 10., code_dim=8, width=16, layers=1,
                          heads=2, hidden=16, components=2, diffusion_steps=10)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    assert model.history_encoder is model.history_gru
    assert model.context_encoder is model.prefix_gru
    assert model.time_predictor is model.prediction
    assert model.feature_predictor is model.feature_prediction
    after = model.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(value, after[key]) for key, value in before.items())
    assert not any(key.startswith(('history_encoder.', 'context_encoder.',
                                   'time_predictor.', 'feature_predictor.')) for key in after)
