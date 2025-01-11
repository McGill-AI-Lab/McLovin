import csv
import json
from datetime import datetime

API_COUNT_FILE = "utils/ProfileGenerator/api_calls_count.json"

def load_count():
    try:
        with open(API_COUNT_FILE, mode='r') as file:
            data = json.load(file)
            return data['api_calls_count']

    except FileNotFoundError:
        return -1


def increment_count():
    try:
        with open(API_COUNT_FILE, mode='r+') as file:
            data = json.load(file)
            data['api_calls_count'] += 1

            file.seek(0)
            file.truncate()

            json.dump(data, file, indent=4)

    except FileNotFoundError:
        return -1


def reset_count():
    try:
        with open(API_COUNT_FILE, mode='r+') as file:
            data = json.load(file)

            if str(datetime.today()).split()[0] == data['date_time']:
                return 'Same Day'

            else:
                data['date_time'] = str(datetime.today()).split()[0]
                data['api_calls_count'] = 0

                file.seek(0)
                file.truncate()

                json.dump(data, file, indent=4)

    except FileNotFoundError:
        return -1
