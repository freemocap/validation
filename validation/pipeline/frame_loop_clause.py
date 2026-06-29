from validation.pipeline.context import PipelineContext
class FrameLoopClause:
    @classmethod
    def enabled(cls, ctx:PipelineContext, cls_name:str) -> bool:
        config = ctx.get(f"{cls_name}.config")
        if not config or not ctx.project_config.conditions:
            return False
        return bool(config.get("loop_over_conditions", False))
    