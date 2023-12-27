import logging

def init_log(log_name, log_file=None, mode='w+'):
    log_format = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
    log = logging.getLogger(log_name)
    log.setLevel(logging.DEBUG)
    if log_file is not None:
        log.handlers = []
        fh = logging.FileHandler(log_file, mode)
        fh.setFormatter(log_format)
        log.addHandler(fh)
        return log
    ch = logging.StreamHandler()
    ch.setFormatter(log_format)
    log.addHandler(ch)
    return log

def dispose_log(log):
    for handler in log.handlers:
        handler.close()
        log.removeFilter(handler)