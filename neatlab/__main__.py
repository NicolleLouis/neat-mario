from neatlab.experiences.blackjack import Blackjack
from neatlab.experiences.xor import XOR
from neatlab.train import Train

Train(Blackjack, backup_filename='test').run(51)
