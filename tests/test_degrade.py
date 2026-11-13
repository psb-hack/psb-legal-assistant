from lex_degrade import PipelineStage, choose_stage


def test_choose_stage_thresholds() -> None:
    assert choose_stage(20.0) is PipelineStage.FULL
    assert choose_stage(19.9) is PipelineStage.NO_RERANK
    assert choose_stage(8.0) is PipelineStage.NO_RERANK
    assert choose_stage(7.9) is PipelineStage.NO_CLASSIFIER
    assert choose_stage(2.0) is PipelineStage.NO_CLASSIFIER
    assert choose_stage(1.9) is PipelineStage.TEMPLATE
    assert choose_stage(0.0) is PipelineStage.TEMPLATE
