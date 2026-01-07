import time

def retry(
        num_attempts : int = 1,
        pause_between_retries_ms : int = 1000,
        logger = None
        ):
    def decorator(method):
        def wrapper(*args, **kw):
            for i in range(num_attempts):
                try:
                    result = method(*args, **kw)
                    if i>0:
                        print(f"retry_util.retry done {method.__name__} on #{i+1} call. {args} {kw}")
                    return result
                except Exception as retry_error:
                    if i==(num_attempts-1):
                        err_msg = f"retry_util.retry gave up {method.__name__} after {num_attempts} calls. {args} {kw}. {retry_error}"
                        if logger:
                            logger.error(err_msg)
                        else:
                            print(err_msg)
                        raise Exception(err_msg) from retry_error
                finally:
                    time.sleep(int(pause_between_retries_ms/1000))
                    
        return wrapper
    return decorator