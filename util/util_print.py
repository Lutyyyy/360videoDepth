class Bcolors:
    def __init__(self):
        self.HEADER = "\033[95m"
        self.OKBLUE = "\033[94m"
        self.OKGREEN = "\033[92m"
        self.WARNING = "\033[93m"
        self.DEBUG = "\033[97m"
        self.FAIL = "\033[91m"
        self.ENDC = "\033[0m"

    def disable(self):
        self.HEADER = ""
        self.OKBLUE = ""
        self.OKGREEN = ""
        self.WARNING = ""
        self.FAIL = ""
        self.ENDC = ""


def print_grad_stats(grad):
    grad_ = grad.detach()
    print(
        "\nmin, max, mean, std: %e, %e, %e, %e"
        % (
            grad_.min().item(),
            grad_.max().item(),
            grad_.mean().item(),
            grad_.std().item(),
        )
    )


bcolors = Bcolors()
# bcolors.disable()
str_stage = bcolors.OKBLUE + "==>" + bcolors.ENDC
str_verbose = bcolors.OKGREEN + "[Verbose]" + bcolors.ENDC
str_warning = bcolors.WARNING + "[Warning]" + bcolors.ENDC
str_error = bcolors.FAIL + "[Error]" + bcolors.ENDC
str_debug = bcolors.DEBUG + "[Debug...]" + bcolors.ENDC
