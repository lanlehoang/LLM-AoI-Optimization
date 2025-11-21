import multiprocessing


class Params:
    def __init__(self):
        # General settings
        self.method = "eoh"
        self.problem = "satellite_routing"
        self.selection = None
        self.management = None

        # EC settings
        self.ec_pop_size = 5  # number of algorithms in each population, default = 10
        self.ec_n_pop = 5  # number of populations, default = 10
        self.ec_operators = None  # evolution operators: ['e1','e2','m1','m2', 'm3']
        self.ec_m = 2  # number of parents for 'e1' and 'e2' operators, default = 2
        self.ec_operator_weights = None  # weights for operators, i.e., the probability of use the operator in each iteration, default = [1,1,1,1]

        # LLM settings
        self.llm_api_endpoint = None  # endpoint for remote LLM, e.g., api.deepseek.com
        self.llm_api_key = None  # API key for remote LLM, e.g., sk-xxxx
        self.llm_model = None  # model type for remote LLM, e.g., deepseek-chat

        # Exp settings
        self.exp_debug_mode = False  # if debug
        self.exp_output_path = "./"  # default folder for ael outputs
        self.exp_use_seed = False
        self.exp_seed_path = "./seeds/seeds.json"
        self.exp_use_continue = False
        self.exp_continue_id = 0
        self.exp_continue_path = "./results/pops/population_generation_0.json"
        self.exp_n_proc = 1

        # Evaluation settings
        self.eva_timeout = 30
        self.eva_numba_decorator = False

    def set_parallel(self):
        num_processes = multiprocessing.cpu_count()
        if self.exp_n_proc == -1 or self.exp_n_proc > num_processes:
            self.exp_n_proc = num_processes
            print(f"Set the number of proc to {num_processes} .")

    def set_ec(self):
        if self.management == None:
            self.management = "pop_greedy"

        if self.selection == None:
            self.selection = "prob_rank"

        if self.ec_operators == None:
            self.ec_operators = ["e1", "e2", "m1", "m2"]

        if self.ec_operator_weights == None:
            self.ec_operator_weights = [1 for _ in range(len(self.ec_operators))]
        elif len(self.ec_operator) != len(self.ec_operator_weights):
            print(
                "Warning! Lengths of ec_operator_weights and ec_operator shoud be the same."
            )
            self.ec_operator_weights = [1 for _ in range(len(self.ec_operators))]

        if self.method in ["ls", "sa"] and self.ec_pop_size > 1:
            self.ec_pop_size = 1
            self.exp_n_proc = 1
            print("> single-point-based, set pop size to 1. ")

    def set_evaluation(self):
        # Initialize evaluation settings
        if self.problem == "bp_online":
            self.eva_timeout = 20
            self.eva_numba_decorator = True
        elif self.problem == "tsp_construct":
            self.eva_timeout = 20

    def set_params(self, *args, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Identify and set parallel
        self.set_parallel()

        # Initialize method and ec settings
        self.set_ec()

        # Initialize evaluation settings
        self.set_evaluation()


if __name__ == "__main__":
    # Create an instance of the Params class
    params_instance = Params()

    # Setting parameters using the set_params method
    params_instance.set_params(
        llm_use_local=True, llm_local_url="http://example.com", ec_pop_size=8
    )

    # Accessing the updated parameters
    print(params_instance.llm_use_local)  # Output: True
    print(params_instance.llm_local_url)  # Output: http://example.com
    print(params_instance.ec_pop_size)  # Output: 8
