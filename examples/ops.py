
def get_agent_cls_grid(agent_type, algo="a2c"):
    """
    Get agent wrapper for grid world
    """
    if agent_type in ["equivariant", "nullspace", "random"]:
        from rlpyt.agents.pg.grid import GridBasisAgent
        return GridBasisAgent, agent_type
    elif agent_type == "cnn":
        from rlpyt.agents.pg.grid import GridFfAgent
        return GridFfAgent, None
    else:
        raise TypeError("No agent of type {agent_type} known")

def get_agent_cls_cartpole(agent_type, algo="ppo"):
    """
    Get agent wrapper for cartpole
    """
    if agent_type in ["equivariant", "nullspace", "random"]:
        from rlpyt.agents.pg.cartpole import CartpoleBasisAgent
        return CartpoleBasisAgent, agent_type
    elif agent_type == "mlp":
        from rlpyt.agents.pg.cartpole import CartpoleFfAgent
        return CartpoleFfAgent, None
    else:
        raise TypeError("No agent of type {agent_type} known")
