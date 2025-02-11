from pymongo import MongoClient

client = MongoClient("mongodb+srv://oscartesniere:oscar@users.g0tmm.mongodb.net")
print(client.list_database_names())
