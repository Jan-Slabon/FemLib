from enum import IntEnum
from datetime import datetime

class Log_Level(IntEnum):
    Debug = 0
    Info = 1
    Warning = 2
    Error = 3
class Logging:
    log_level : Log_Level = Log_Level.Warning
    @staticmethod
    def set_log_level(new_level : Log_Level) -> None:
        Logging.log_level = new_level
    @staticmethod
    def Log(level : Log_Level, message) -> None:
        if(level >= Logging.log_level):
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("LOGGING:", current_time, message)

