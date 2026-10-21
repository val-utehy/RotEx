class ConsoleLogger:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def info(msg): print(f"{ConsoleLogger.GREEN}{ConsoleLogger.BOLD}[INFO]{ConsoleLogger.ENDC} {msg}")
    @staticmethod
    def step(msg): print(f"{ConsoleLogger.CYAN}[STEP]{ConsoleLogger.ENDC} {msg}")
    @staticmethod
    def warn(msg): print(f"{ConsoleLogger.WARNING}{ConsoleLogger.BOLD}[WARN]{ConsoleLogger.ENDC} {msg}")
    @staticmethod
    def section(msg): print(f"\n{ConsoleLogger.HEADER}{ConsoleLogger.BOLD}{'='*60}\n{msg}\n{'='*60}{ConsoleLogger.ENDC}")
