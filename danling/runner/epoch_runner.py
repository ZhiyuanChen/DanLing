from .runner import Runner


class EpochRunner(Runner):
    """
    Set up everything for running a job
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if "epoch_begin" not in self:
            self.epoch_begin = 0
        if "epoch_end" not in self:
            raise ValueError('"epoch_end" must be specified for EpochRunner')
