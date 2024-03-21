import functools
from threading import Thread


def timeout(timeout):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [
                TimeoutError(
                    f"function [{func.__name__}] timeout [{timeout} seconds] exceeded!"
                )
            ]

            def new_func():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:  # pylint: disable=broad-except
                    res[0] = e

            t = Thread(target=new_func)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                print("error starting thread")
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret

        return wrapper

    return deco
