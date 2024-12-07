class easyremoteError(Exception):
    """Base exception for easyremote"""
    print("Base exception for easyremote")

class ConversionError(easyremoteError):
    """Raised when data conversion fails"""
    print("Raised when data conversion fails")

class ProcessingError(easyremoteError):
    """Raised when data processing fails"""
    print("Raised when data processing fails")

class BridgeError(easyremoteError):
    """Raised when bridge operations fail"""
    print("Raised when bridge operations fail")