from datetime import datetime

print(str(datetime.now()).replace(":", "-").split(".")[0])
