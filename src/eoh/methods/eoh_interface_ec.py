import re
import time
import warnings
import concurrent.futures

import numpy as np
from joblib import Parallel, delayed

from .eoh_evolution import Evolution
from .evaluator_accelerate import add_numba_decorator


class InterfaceEC:
    def __init__(
        self,
        pop_size,
        m,
        api_endpoint,
        api_key,
        llm_model,
        llm_use_local,
        llm_local_url,
        debug_mode,
        interface_prob,
        n_p,
        timeout,
        use_numba,
        **kwargs,
    ):
        self.pop_size = pop_size
        self.interface_eval = interface_prob
        prompts = interface_prob.prompts

        self.evol = Evolution(
            api_endpoint, api_key, llm_model, llm_use_local, llm_local_url, debug_mode, prompts, **kwargs
        )

        self.m = m  # Number of parents for e1 and e2
        self.debug = debug_mode

        if not self.debug:
            warnings.filterwarnings("ignore")

        self.n_p = n_p
        self.timeout = timeout
        self.use_numba = use_numba

    def check_duplicate(self, population, code):
        """Check if code already exists in the population."""
        return any(code == ind["code"] for ind in population)

    def _simple_parent_selection(self, pop, n):
        """Simple parent selection: pick top n individuals by objective"""
        # Sort by objective (higher is better)
        sorted_pop = sorted(pop, key=lambda x: x["objective"] if x["objective"] else -float("inf"), reverse=True)
        return sorted_pop[:n]

    def population_generation(self):
        """Generate the initial population using operator i1."""
        n_create = 2
        population = []

        for _ in range(n_create):
            _, pop = self.get_algorithm([], "i1")
            population.extend(pop)

        return population

    def _get_alg(self, pop, operator):
        """Dispatch for evolution operators i1/e1/e2/m1/m2/m3."""
        offspring = {
            "algorithm": None,
            "code": None,
            "objective": None,
            "other_inf": None,
        }

        if operator == "i1":
            parents = None
            offspring["code"], offspring["algorithm"] = self.evol.i1()

        elif operator in ("e1", "e2"):
            parents = self._simple_parent_selection(pop, self.m)
            method = getattr(self.evol, operator)
            offspring["code"], offspring["algorithm"] = method(parents)

        elif operator in ("m1", "m2", "m3"):
            parents = self._simple_parent_selection(pop, 1)
            method = getattr(self.evol, operator)
            offspring["code"], offspring["algorithm"] = method(parents[0])

        else:
            print(f"Evolution operator [{operator}] has not been implemented!\n")
            parents = None

        return parents, offspring

    def get_offspring(self, pop, operator):
        """Create an offspring + evaluate its fitness (with timeout & retry)."""
        try:
            parents, offspring = self._get_alg(pop, operator)

            # Optional: inject @numba decorator
            if self.use_numba:
                pattern = r"def\s+(\w+)\s*\(.*\):"
                match = re.search(pattern, offspring["code"])
                function_name = match.group(1)
                code = add_numba_decorator(program=offspring["code"], function_name=function_name)
            else:
                code = offspring["code"]

            # Duplicate handling
            n_retry = 1
            while self.check_duplicate(pop, offspring["code"]):
                n_retry += 1

                if self.debug:
                    print("duplicated code, wait 1 second and retrying ... ")

                parents, offspring = self._get_alg(pop, operator)

                if self.use_numba:
                    pattern = r"def\s+(\w+)\s*\(.*\):"
                    match = re.search(pattern, offspring["code"])
                    function_name = match.group(1)
                    code = add_numba_decorator(program=offspring["code"], function_name=function_name)
                else:
                    code = offspring["code"]

                if n_retry > 1:
                    break

            # Evaluate with timeout
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self.interface_eval.evaluate, code)
                fitness = future.result(timeout=self.timeout)
                future.cancel()

            offspring["objective"] = np.round(fitness, 5).item()

        except Exception:
            parents = None
            offspring = {
                "algorithm": None,
                "code": None,
                "objective": None,
                "other_inf": None,
            }

        return parents, offspring

    def get_algorithm(self, pop, operator):
        """Parallel generation of multiple offspring."""
        try:
            results = Parallel(n_jobs=self.n_p, timeout=self.timeout + 15)(
                delayed(self.get_offspring)(pop, operator) for _ in range(self.pop_size)
            )
        except Exception as e:
            if self.debug:
                print(f"Error: {e}")
            print("Parallel time out.")
            results = []

        time.sleep(1)

        out_p = []
        out_off = []
        for parents, off in results:
            out_p.append(parents)
            out_off.append(off)

            if self.debug:
                print(f">>> check offsprings:\n {off}")

        return out_p, out_off
