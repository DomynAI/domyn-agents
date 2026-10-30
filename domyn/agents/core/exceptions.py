class RestHookError(Exception):
    def __init__(self, error_code: str | int, error_message: str):
        self.error_code = error_code
        self.error_message = error_message
        super().__init__(f"{error_code}: {error_message}")
