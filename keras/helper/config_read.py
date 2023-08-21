import pytest

class init_config:
    def __init__(self,
                 file_path):
        self.file_path = file_path

    def load(self):
        self.__config_file_validation__()

    def __config_file_validation__(self) -> None:
        """
        validate the config file
        :return:
        """
        pass




