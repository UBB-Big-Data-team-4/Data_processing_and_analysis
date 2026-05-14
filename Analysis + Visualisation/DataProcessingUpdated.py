import datetime
import random

# global state (simulates continuous stream)
current_time = datetime.datetime.now()
current_people = 5


def getProcessedData():
    global current_time
    global current_people

    # simulate realistic office movement
    change = random.choice([-2, -1, 0, 1, 2])
    current_people += change

    # keep values realistic
    current_people = max(0, min(current_people, 25))

    # advance time (simulate stream)
    current_time = current_time + datetime.timedelta(seconds=5)

    return {
        "timestamp": current_time,
        "people": current_people
    }