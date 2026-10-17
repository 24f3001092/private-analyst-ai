class PrivateAnalystError(Exception):
    """Base error. Messages must not include raw customer payloads."""


class ValidationError(PrivateAnalystError):
    pass


class UnsupportedFormatError(ValidationError):
    pass


class FileTooLargeError(ValidationError):
    pass


class ResourceLimitError(PrivateAnalystError):
    pass


class ModelUnavailableError(PrivateAnalystError):
    pass


class ModelResponseError(PrivateAnalystError):
    pass


class AuthenticationError(PrivateAnalystError):
    pass


class AuthorizationError(PrivateAnalystError):
    pass


class IsolationError(PrivateAnalystError):
    pass


class AuditLogError(PrivateAnalystError):
    pass
