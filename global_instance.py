from Utils.read_yaml_file import read_yaml
from Database.database import database

def __read_yaml_file():
    return read_yaml("../config_yaml.yaml")

__read_data__ = __read_yaml_file()

database_context = database(
    server_domain=__read_data__['DATABASE']['SERVER_DOMAIN'],
    database_name= __read_data__['DATABASE']['DATABASE']
)