class database:
    def __init__(self,
                 server_domain,
                 database_name,
                 trusted_connect=True):
        self.__server_domain = server_domain
        self.__database_name = database_name
        self.__trusted_connect = "Yes" if trusted_connect else "No"

    @property
    def server_domain(self):
        return self.__server_domain

    @server_domain.setter
    def server_domain(self, value):
        self.__server_domain = value

    @property
    def database_name(self):
        return self.__database_name

    @database_name.setter
    def database_name(self, value):
        self.__database_name = value

    @property
    def trusted_connect(self):
        return self.__trusted_connect

    @trusted_connect.setter
    def trust_connect(self, value):
        self.__trusted_connect =  "Yes" if trusted_connect else "No"





