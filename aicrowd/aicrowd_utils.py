import os


def is_on_aicrowd_server():
    on_aicrowd_server = os.getenv('AICROWD_IS_GRADING', False)
    on_aicrowd_server = True if on_aicrowd_server != False else on_aicrowd_server
    return on_aicrowd_server
