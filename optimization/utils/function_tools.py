import functools
from threading import Thread

from typing import Callable


def timeout(timeout_time: int) -> Callable[..., object]:
    def deco(func: Callable[..., object]) -> Callable[..., object]:
        @functools.wraps(func)
        def wrapper(*args: tuple[object], **kwargs: dict[object, object]) -> object:
            res: list[Exception | object] = [
                TimeoutError(
                    f"function [{func.__name__}] timeout"
                    f" [{timeout_time} seconds] exceeded!"
                )
            ]

            def new_func() -> None:
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:  # pylint: disable=broad-except
                    res[0] = e

            t = Thread(target=new_func)
            t.daemon = True
            try:
                t.start()
                t.join(timeout_time)
            except Exception as je:
                print("error starting thread")
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret

        return wrapper

    return deco
