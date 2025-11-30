from .utils.create_folders import create_folders
from .problems.run import SatelliteRouting
from .methods.eoh import EOH


class Evol:
    def __init__(self, params, model_path, **kwargs):
        print("----------------------------------------- ")
        print("---              Start EoH            ---")
        print("-----------------------------------------")

        self.params = params
        print("- parameters loaded -")
        self.model_path = model_path

        # Create output folder
        create_folders(params.exp_output_path)
        print("- output folder created -")

    def run(self):
        print("- loading problem -")
        problem = SatelliteRouting(model_path=self.model_path)

        print("- initializing EoH -")
        method = EOH(self.params, problem)

        print("- starting evolution -")
        method.run()

        print("> End of Evolution! ")
        print("----------------------------------------- ")
        print("---     EoH successfully finished !   ---")
        print("-----------------------------------------")
