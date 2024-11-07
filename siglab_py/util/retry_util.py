def retry(num_attempts : int = 1):
    def decorator(method):
        def wrapper(*args, **kw):
            for i in range(num_attempts):
                try:
                    result = method(*args, **kw)
                    if i>0:
                        print(f"retry_gizmo.retry succeeded: {method.__name__} on #{i+1} invocation. {args} {kw}")
                    return result
                except Exception as retry_error:
                    if i==(num_attempts-1):
                        err_msg = f"retry_gizmo.retry failed: {method.__name__} after {num_attempts} invocations. {args} {kw}. {retry_error}"
                        raise Exception(err_msg) from retry_error
        return wrapper
    return decorator