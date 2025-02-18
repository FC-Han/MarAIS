"""
Error Functions
"""


class InvalidInputError(Exception):
    """Exception raised for invalid input values."""

    def __init__(self, value, allowed_values):
        self.value = value
        self.allowed_values = allowed_values
        super().__init__(f"Invalid input: {value}. Allowed values are {allowed_values}.")
