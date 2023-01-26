from functools import wraps
from typing import Any, Callable, Mapping, Optional

from chanfig import NestedDict


class Registry(NestedDict):
    """
    `Registry` for components.

    Notes
    -----

    `Registry` inherits from [`NestedDict`](https://chanfig.danling.org/nested_dict/).

    Therefore, `Registry` comes in a nested structure by nature.
    You could create a sub-registry by simply calling `registry.sub_registry = Registry`,
    and access through `registry.sub_registry.register()`.

    Examples
    --------
    ```python
    >>> registry = Registry("test")
    >>> @registry.register
    ... @registry.register("Module1")
    ... class Module:
    ...     def __init__(self, a, b):
    ...         self.a = a
    ...         self.b = b
    >>> module = registry.register(Module, "Module2")
    >>> registry
    Registry(
      (Module1): <class 'danling.registry.registry.Module'>
      (Module): <class 'danling.registry.registry.Module'>
      (Module2): <class 'danling.registry.registry.Module'>
    )
    >>> registry.lookup("Module")
    <class 'danling.registry.registry.Module'>
    >>> config = {"module": {"name": "Module", "a": 1, "b": 2}}
    >>> # registry.register(Module)
    >>> module = registry.build(config["module"])
    >>> type(module)
    <class 'danling.registry.registry.Module'>
    >>> module.a
    1
    >>> module.b
    2

    ```
    """

    override: bool = False

    def __init__(self, override: bool = False):
        super().__init__()
        self.setattr("override", override)

    def register(self, component: Optional[Callable] = None, name: Optional[str] = None) -> Callable:
        r"""
        Register a new component

        Parameters
        ----------
        component: Optional[Callable] = None
            The component to register.
        name: Optional[str] = component.__name__
            The name of the component.

        Returns
        -------
        component: Callable
            The registered component.

        Raises
        ------
        ValueError
            If the component with the same name already exists and `Registry.override=False`.

        Examples
        --------
        ```python
        >>> registry = Registry("test")
        >>> @registry.register
        ... @registry.register("Module1")
        ... class Module:
        ...     def __init__(self, a, b):
        ...         self.a = a
        ...         self.b = b
        >>> module = registry.register(Module, "Module2")
        >>> registry
        Registry(
          (Module1): <class 'danling.registry.registry.Module'>
          (Module): <class 'danling.registry.registry.Module'>
          (Module2): <class 'danling.registry.registry.Module'>
        )

        ```
        """

        if isinstance(name, str) and name in self and not self.override:
            raise ValueError(f"Component with name {name} already exists")

        # Registry.register()
        if name is not None:
            self.set(name, component)

        # @Registry.register()
        @wraps(self.register)
        def register(component, name=None):
            if name is None:
                name = component.__name__
            self.set(name, component)
            return component

        # @Registry.register
        if callable(component) and name is None:
            return register(component)

        return lambda x: register(x, component)

    def lookup(self, name: str) -> Any:
        r"""
        Lookup for a component.

        Parameters
        ----------
        name: str
            The name of the component.

        Returns
        -------
        value: Any

        Raises
        ------
        KeyError
            If the component is not registered.

        Examples
        --------
        ```python
        >>> registry = Registry("test")
        >>> @registry.register
        ... class Module:
        ...     def __init__(self, a, b):
        ...         self.a = a
        ...         self.b = b
        >>> registry.lookup("Module")
        <class 'danling.registry.registry.Module'>

        ```
        """

        return self.get(name)

    def build(self, name: str, *args, **kwargs):
        r"""
        Build a component.

        Parameters
        ----------
        name: str
            The name of the component.
        *args
            The arguments to pass to the component.
        **kwargs
            The keyword arguments to pass to the component.

        Returns
        -------
        component: Callable

        Raises
        ------
        KeyError
            If the component is not registered.

        Examples
        --------
        ```python
        >>> registry = Registry("test")
        >>> @registry.register
        ... class Module:
        ...     def __init__(self, a, b):
        ...         self.a = a
        ...         self.b = b
        >>> config = {"module": {"name": "Module", "a": 1, "b": 2}}
        >>> # registry.register(Module)
        >>> module = registry.build(config["module"])
        >>> type(module)
        <class 'danling.registry.registry.Module'>
        >>> module.a
        1
        >>> module.b
        2

        ```
        """

        if isinstance(name, Mapping) and not args and not kwargs:
            name, kwargs = name.pop("name"), name  # type: ignore
        return self.get(name)(*args, **kwargs)

    def __wrapped__(self):
        pass


GlobalRegistry = Registry()
