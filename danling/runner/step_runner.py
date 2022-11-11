from .base_runner import BaseRunner


class StepRunner(BaseRunner):
    """
    Set up everything for running a job
    """

    step_begin: int
    step_end: int

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if "step_begin" not in self:
            self.step_begin = 0
        if "step_end" not in self:
            raise ValueError('"step_end" must be specified for StepRunner')
