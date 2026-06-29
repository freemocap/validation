from abc import ABC, abstractmethod
from pathlib import Path
import logging
from validation.datatypes.data_component import DataComponent
from typing import List, Type
from validation.pipeline.context import PipelineContext
from validation.pipeline.frame_loop_clause import FrameLoopClause



logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class StepLoggerAdapter(logging.LoggerAdapter):
    """Prefixes log messages with step name."""
    def process(self, msg, kwargs):
        return f"[{self.extra.get('step','UnknownStep')}] {msg}", kwargs

class ValidationStep(ABC):
    REQUIRES: list[DataComponent] = []
    PRODUCES: list[DataComponent] = []
    CONFIG = None

    @classmethod
    def expected_requirements(cls, ctx:PipelineContext) -> list[DataComponent]:
        return cls.REQUIRES
    
    @classmethod
    def expected_products(cls, ctx:PipelineContext) -> list[DataComponent]:
        if not FrameLoopClause.enabled(ctx, cls.__name__):
            return cls.PRODUCES
        return [p.clone_with_prefix(condition) if condition not in (None,"") else p
                for p in cls.PRODUCES 
                for condition in ctx.conditions.keys()]

    def __init__(self, context: PipelineContext, logger=None):
        self.ctx = context
        self.logger = logger or logging.getLogger(__name__)
        self.data = {}
        self.outputs = {}

        self.cfg = self._check_config(self.ctx)
        self.loop_enabled = FrameLoopClause.enabled(self.ctx, self.__class__.__name__)
        self._resolve_requirements()

    def _check_config(self, ctx:PipelineContext):
        if self.CONFIG is not None:
            config = ctx.get(f"{self.__class__.__name__}.config")

            if config is None:
                raise RuntimeError(
                    f"{self.__class__.__name__} requires a {self.CONFIG.__name__}, "
                    "but none was provided in the context. Check the pipeline_config.yaml")
            if isinstance(config, dict):
                config = self.CONFIG(**config)
            self.logger.info(f"Step {self.__class__.__name__} using config: {config}")  
        else:
            config = None
        return config

    def _resolve_requirements(self):
        for requirement in self.REQUIRES:
            val = self.ctx.get(requirement.name)
            if val is None:
                raise RuntimeError(
                    f"{self.__class__.__name__} needs {requirement.name}, "
                    "but it isn’t in the context.")
            self.data[requirement.name] = val

    @abstractmethod
    def calculate(self):
        "Perform calculation and pass results"
        pass
                
    def store(self, condition_name: str | None = None):
        for data_comp in self.PRODUCES:
            output = self.outputs.get(data_comp.name)
            if output is None:
                self.logger.warning(f'No output found for {data_comp.name}, skipping save to disk')
                continue

            dc_to_save = data_comp.clone_with_prefix(condition_name) if condition_name not in (None,"") else data_comp

            if dc_to_save.saver is not None:
                dc_to_save.save(self.ctx.recording_dir, output, **self.ctx.data_component_context)
                self.logger.info(f"Saved {dc_to_save.name} to {dc_to_save.full_path(self.ctx.recording_dir, **self.ctx.data_component_context)}")
            else:
                self.logger.warning(f"No saver found for {dc_to_save.name}, skipping disk save")
            self.ctx.put(dc_to_save.name, output)
            
    def calculate_and_store(self):
        if not self.loop_enabled:
            self.calculate()
            self.store()
            return 

        conditions = self.ctx.conditions
        items = list(conditions.items())
        if len(items) == 1 and items[0][0] in (None,""):
            _,frames = items[0]
            self.logger.info(f"Running {self.__class__.__name__} with frames {frames}")
            self.calculate(frames['frames'])
            self.store()
            self.outputs.clear()
            return

        for condition_name, frames in self.ctx.conditions.items():
            self.logger.info(f"Running {self.__class__.__name__} for condition '{condition_name}' with frames {frames}")
            self.calculate(frames['frames'])
            self.store(condition_name)
            self.outputs.clear()



ValidationStepClass = Type[ValidationStep]
ValidationStepList = List[ValidationStepClass]

class ValidationPipeline:
    def __init__(
            self,
            context:PipelineContext,
            steps: ValidationStepList,
            logger: logging.Logger | None = None,
    ):  
        self.ctx = context
        self.logger = logger or logging.getLogger(__name__)
        self.step_classes = steps

    def load_reqs(self, step_cls:type[ValidationStep]):
        added_comp_list = []
        for data_comp in step_cls.expected_requirements(self.ctx):
            if not data_comp.exists(self.ctx.recording_dir, **self.ctx.data_component_context):
                raise FileNotFoundError(f"{data_comp.name} is required but not found at {data_comp.full_path(self.ctx.recording_dir, **self.ctx.data_component_context)}")
            self.ctx.put(data_comp.name, data_comp.load(self.ctx.recording_dir, **self.ctx.data_component_context))
            added_comp_list.append(data_comp.name)
        self.logger.info(f"Pre-loaded {added_comp_list} for {step_cls.__name__}")

    def _preflight_check(self, start_at:int):
        self.load_reqs(self.step_classes[start_at]) #load requirements for first step being run

        produced = set(self.ctx.backpack.keys())

        for step_cls in self.step_classes[start_at:]:
            for data_comp in step_cls.expected_requirements(self.ctx):
                if data_comp.name in produced:
                    continue
                if not data_comp.exists(self.ctx.recording_dir, **self.ctx.data_component_context):
                    raise FileNotFoundError(
                        f"{data_comp.name} is required by {step_cls.__name__} but not found at "
                        f"{data_comp.full_path(self.ctx.recording_dir, **self.ctx.data_component_context)}")                    
                self.ctx.put(data_comp.name, data_comp.load(self.ctx.recording_dir, **self.ctx.data_component_context))
                self.logger.info(f"Pre-loaded {data_comp.name} for {step_cls.__name__}")
            
            produced.update([p.name for p in step_cls.expected_products(self.ctx)])
        self.logger.info("Preflight check passed")

    def run(self, *, start_at: int =0):
        if not (0 <= start_at < len(self.step_classes)):
            raise IndexError(f"start_at={start_at} is outside valid step range (0–{len(self.step_classes) - 1})")

        self._preflight_check(start_at=start_at)
        
        #run the pipeline
        for step_cls in self.step_classes[start_at:]:
            step_logger = StepLoggerAdapter(self.logger, {"step": step_cls.__name__})
            step = step_cls(self.ctx, logger=step_logger)

            step_logger.info(f"Running {step_cls.__name__}")
            step.calculate_and_store()

            if hasattr(step, 'visualize'):
                step.visualize()
