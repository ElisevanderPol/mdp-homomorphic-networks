
from rlpyt.agents.dqn.dqn_agent import DqnAgent
from rlpyt.models.dqn.cartpole_dqn_model import CartpoleDqnModel
from rlpyt.agents.dqn.cartpole.mixin import CartpoleMixin


class CartpoleDqnAgent(CartpoleMixin, DqnAgent):

    def __init__(self, ModelCls=CartpoleDqnModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
