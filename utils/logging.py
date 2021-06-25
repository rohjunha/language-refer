from logging import Formatter, DEBUG, getLogger, StreamHandler

logger = None


class MyFormatter(Formatter):
    width = 50

    def format(self, record):
        width = 50
        datefmt = '%H:%M:%S'
        cpath = '%s:%s:%s' % (record.module, record.funcName, record.lineno)
        cpath = cpath[-width:].ljust(width)
        record.message = record.getMessage()
        s = "[%s - %s] %s" % (self.formatTime(record, datefmt), cpath, record.getMessage())
        if record.exc_info:
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if s[-1:] != "\n":
                s = s + "\n"
            s = s + record.exc_text
        return s


def get_logger(name):
    global logger
    if logger is None:
        LEVEL = DEBUG
        logger = getLogger(name)
        logger.setLevel(LEVEL)
        ch = StreamHandler()
        ch.setLevel(LEVEL)
        formatter = MyFormatter()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger
