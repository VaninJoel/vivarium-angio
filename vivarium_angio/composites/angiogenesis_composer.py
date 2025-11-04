from vivarium.core.composer import Composer
from vivarium_angio.processes.angiogenesis_process import AngiogenesisProcess


class AngiogenesisComposer(Composer):
    """
    Standard composer for Angiogenesis simulations.
    """
    
    defaults = {
        'angiogenesis_process': AngiogenesisProcess.defaults
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.config['angiogenesis_process'] = self.defaults['angiogenesis_process'].copy()
        if config and config.get('angiogenesis_process'):
            self.config['angiogenesis_process'].update(config['angiogenesis_process'])
    
    def generate_processes(self, config):
        """Generate the Angiogenesis process."""
        return {
            'angiogenesis': AngiogenesisProcess(config['angiogenesis_process'])
        }
    
    def generate_topology(self, config):
        """Generate standard single-process topology."""
        return {
            'angiogenesis': {
                'inputs': ('inputs',),
                'outputs': ('outputs',),
            }
        }
