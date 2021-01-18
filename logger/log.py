import logging
import datetime

date = datetime.datetime.now()
date = date.strftime("%d_%m_%Y")
logging.basicConfig(
    filename='log/loggin_'+date+'.log',
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logging.getLogger('tensorflow').disabled = True
