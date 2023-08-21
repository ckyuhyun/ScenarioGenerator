from Database.DatabaseHelper import dbContext


class dbContext_helper:
    def __init__(self):
        self.db = dbContext()
        self.db.connect()

    def get_action_name_by_guid(self, project_id, guid: str):
        self.db.select("TestAction", whereQuery=f"ActiveProject={project_id} and guid = '{guid}'")
        db_data = self.db.get_table_values()
        return db_data.ActionName[0]
