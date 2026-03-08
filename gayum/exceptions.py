class DataTypeError(Exception):
    msg = 'data has an invalid datatype |{invalid_type}| must be narwhals supported dataframe'

    def __init__(self, invalid_type: str):
        super().__init__(self.msg.format(invalid_type=invalid_type))
