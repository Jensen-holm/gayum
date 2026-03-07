class XTrainTypeError(Exception):
    msg = 'X has an invalid datatype |{invalid_type}| must be data a dataframe or a jax ndarray'

    def __init__(self, invalid_type: str) -> None:
        super().__init__(self.msg.format(invalid_type=invalid_type))
