import multiprocessing


class Params:
    def __init__(self):
        # EC settings
        self.ec_pop_size = 5
        self.ec_n_pop = 20
        self.ec_operators = ["e1", "e2", "m1", "m2", "m3"]
        self.ec_operator_weights = [1, 1, 1, 1, 1]
        self.ec_m = 2

        # LLM settings
        self.llm_use_local = False
        self.llm_local_url = None
        self.llm_api_endpoint = None
        self.llm_api_key = None
        self.llm_model = None

        # Experiment settings
        self.exp_debug_mode = False
        self.exp_output_path = "./eoh_results"
        self.exp_n_proc = 1

        # Evaluation settings
        self.eva_timeout = 60
        self.eva_numba_decorator = False

    def set_params(self, **kwargs):
        """Update parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: {key} is not a valid parameter")

        # Auto-adjust processes
        if self.exp_n_proc == -1 or self.exp_n_proc > multiprocessing.cpu_count():
            self.exp_n_proc = multiprocessing.cpu_count()
