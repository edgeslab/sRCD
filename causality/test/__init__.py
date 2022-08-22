import logging
import logging.config
import os

#
# This file exists solely to set up logging during unit test runs. NB: For this to work, following fileConfig() call
# MUST BE DONE PRIOR to any logging calls such as logging.info(). To use logging in your code:
#
# 1) Put this at the top of each module file:
#
#   logger = logging.getLogger(__name__)
#
# 2) Make calls like this:
#
#   logger.info('your info message')
#   logger.debug('your debug message')
#
# 3) Set the TEST_LOG_FILE environment variable to the logging configuration file to use as specified in
# http://docs.python.org/3.2/library/logging.config.html#configuration-file-format . By default test/logging.ini
# is used. NB: The file is located relative to this module's directory. In IntelliJ IDEA you can set this variable
# using the Run/Debug Configuration dialog box's Environment Variables editor. On the command line use your shell's
# feature, e.g., in bash:
#
#   $ export TEST_LOG_FILE=logging-custom.ini
#
# 4) To configure a custom config file to allow output from specific modules you care about, say causality.pc.PC,
# a) add a logger to [loggers]:
#
#   [loggers]
#   keys=root,testLoggingConfig,causality.pc.PC
#
# And b) add a [logger_] entry for it:
#
#   [logger_causality.pc.PC]
#   level=INFO
#   handlers=consoleHandler
#   qualname=causality.pc.PC
#   propagate=0
#
# NB: The qualname must match the name of the logger being configured. If you use __name__ as above, then you'll use
# the module name for qualname.
#

LOG_FILE_VAR = 'TEST_LOG_FILE'
DEFAULT_LOG_FILE = 'logging.ini'

def initLogging():
    envFile = os.environ.get(LOG_FILE_VAR)
    envFileAbs = os.path.join(os.path.dirname(__file__), envFile) if envFile else None
    logFileEnvFound = envFile and os.path.exists(envFileAbs)
    logFileAbs = envFileAbs if logFileEnvFound else os.path.join(os.path.dirname(__file__), DEFAULT_LOG_FILE)
    logging.config.fileConfig(logFileAbs)

    logger = logging.getLogger(__name__)    # 'testLoggingConfig'. NB: must be after fileConfig()
    logger.info("configured test logging from file {logFile}".format(logFile=logFileAbs))
    if envFileAbs and not logFileEnvFound:
        logger.warn('could not find file specified in {}: {}'.format(LOG_FILE_VAR, envFileAbs))


initLogging()
