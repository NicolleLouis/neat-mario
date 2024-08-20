from neatlab.experiences.experience import Experience


class XOR(Experience):
    CONFIG_NAME = 'config-xor'
    DIRECTORY_NAME = 'xor'
    NODE_NAMES = {-1: 'A', -2: 'B', 0: 'A XOR B'}

    XOR_INPUTS = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    XOR_OUTPUTS = [(0.0,), (1.0,), (1.0,), (0.0,)]

    def generate_environment(self):
        pass

    def get_score(self):
        score = 4.0
        for xi, xo in zip(self.XOR_INPUTS, self.XOR_OUTPUTS):
            output = self.net.activate(xi)
            score -= (output[0] - xo[0]) ** 2
        return score
