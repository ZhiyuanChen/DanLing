import pytest

from danling.utils import decorators


class Test:
    def test_catch_error(self):
        @decorators.catch()
        def func():
            raise Exception("test")

        func()

    def test_catch_interrupt(self):
        @decorators.catch()
        def func():
            raise KeyboardInterrupt("test")

        with pytest.raises(KeyboardInterrupt):
            func()

    def test_catch_exit(self):
        @decorators.catch()
        def func():
            raise SystemExit("test")

        with pytest.raises(SystemExit):
            func()

    def test_catch_raise(self):
        @decorators.catch(FileExistsError)
        def func():
            raise FileNotFoundError("test")

        with pytest.raises(FileNotFoundError):
            func()

    def test_catch_exclude(self):
        @decorators.catch(exclude=FileNotFoundError)
        def func():
            raise FileNotFoundError("test")

        with pytest.raises(FileNotFoundError):
            func()
