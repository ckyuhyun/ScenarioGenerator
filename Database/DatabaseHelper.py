import pyodbc
import pandas as pd
from csv import reader
import warnings
import global_instance


class dbContext:
    # Some other example server values are
    # server = 'localhost\sqlexpress' # for a named instance
    # server = 'myserver,port' # to specify an alternate port
    '''
    server = 'tcp:myserver.database.windows.net'
    database = 'AutoTesterDB'
    username = 'plexusadmin'
    password = 'D6mE_uqu'
    '''


    def __init__(self):
        self.table_values = None
        self.cursor = None
        self.connection = None

    def connect(self):

        try:
            # Trusted Connection to Named Instance
            self.connection = pyodbc.connect(
                r"DRIVER={SQL Server};"
                fr'SERVER={global_instance.database_context.server_domain};'
                fr'DATABASE={global_instance.database_context.database_name};'
                fr'Trusted_Connection={global_instance.database_context.trusted_connect};'
            )
            self.cursor = self.connection.cursor()
        except:
            print("not connected?")

    def select(self,
               table_name,
               orderBy="",
               whereQuery=""):
        warnings.simplefilter("ignore")
        _orderBy = "" if orderBy == "" else "Order by {0}".format(orderBy)
        _where = "" if whereQuery == "" else "where {0}".format(whereQuery)
        _columns = self.__get_table_columns__(table_name)
        # self.cursor.execute("select * from {0} {1}".format(table_name, _orderBy))
        # rows = self.cursor.fetchone()

        _query = "select * from {0} {1} {2}".format(table_name, _where, _orderBy)
        self.table_values = pd.read_sql(_query, self.connection)

    def run_query(self, query):
        _query = query
        self.table_values = pd.read_sql(_query, self.connection)

    def get_table_values(self) -> list:
        return self.table_values

    def __get_table_columns__(self, table_name) -> list:
        self.cursor.execute("select COLUMN_NAME from information_schema.columns where table_name = '{0}'".format(table_name))
        column = self.cursor.fetchone()
        table_columns = []
        while column:
            table_columns.append(column[0])
            column = self.cursor.fetchone()

        return table_columns
