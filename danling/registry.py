from chanfig import Config


class Registry(Config):
    """
    Registry for components
    """

    register = Config.set
    lookup = Config.get
